"""Online adaptive conformal inference (ACI) with a delayed update.

Following Asch et al. (arXiv:2606.19642): the ensemble's nominal
[LOWER_QUANTILE, UPPER_QUANTILE] interval is corrected by a running scalar
c_t, updated as

    c <- c + eta * (err - alpha)

where ``err`` is 1 if the outcome fell outside the emitted interval and 0
otherwise. Because the outcome of a forecast issued at step t is only observed
at step t + tau, the update from step t is applied to c at step t + tau.

Two things make this module non-trivial and both are deliberate:

1.  **Vectorised over the grid.** c is an array with the shape of the grid
    (lat, lon), not a scalar. Every gridpoint runs its own independent
    controller; there are no Python loops over gridpoints. The only Python
    loop is over time, which is inherently sequential.

2.  **Adaptation space is pluggable.** See :class:`Adapter` below.

    TODO(lucas): [Fri] ==== ADAPTATION SPACE IS UNDECIDED ====
    Both VariableSpaceAdapter and QuantileSpaceAdapter implement the same
    interface and are drop-in interchangeable via config.ACI_SPACE. They are
    NOT equivalent: variable-space padding is unbounded and has units of
    Kelvin, quantile-space adaptation saturates once alpha - c leaves [0, 1]
    and so cannot widen an interval past the ensemble min/max. That saturation
    is exactly what may bite in the high-p_t bins, which is the paper's point,
    so run both before committing. Decide on the burn-in window only.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import config

__all__ = [
    "Adapter",
    "VariableSpaceAdapter",
    "QuantileSpaceAdapter",
    "make_adapter",
    "DelayedACI",
    "AciResult",
]


# --------------------------------------------------------------------------
# Adaptation space
# --------------------------------------------------------------------------


class Adapter:
    """Maps (ensemble, correction c) -> (lower, upper) interval endpoints.

    Subclasses define *what space* the ACI correction lives in. The controller
    in :class:`DelayedACI` is agnostic to it: it only knows that a larger c
    means a wider interval.

    Contract for every subclass:
      - ``interval(ensemble, c)`` where ``ensemble`` has shape
        ``(n_members, *grid)`` and ``c`` has shape ``grid`` (or is a scalar),
        returns ``(lower, upper)``, each of shape ``grid``.
      - The mapping must be monotone non-decreasing in c for ``upper`` and
        monotone non-increasing in c for ``lower``. ACI's convergence argument
        needs this; if you add a third adapter, keep it true.
      - c == 0 must reproduce the raw ensemble interval at the nominal
        quantiles, so that the raw baseline is just this class with c fixed at 0.
    """

    def interval(self, ensemble: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "in: ensemble (n_members, *grid), c (*grid); out: (lower, upper) each (*grid)"
        )


@dataclass
class VariableSpaceAdapter(Adapter):
    """Pad the nominal ensemble quantiles by +/- c, in the units of the variable.

    lower = quantile(ensemble, lower_q) - c
    upper = quantile(ensemble, upper_q) + c
    """

    lower_q: float = config.LOWER_QUANTILE
    upper_q: float = config.UPPER_QUANTILE

    def interval(self, ensemble: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "in: ensemble (n_members, *grid), c (*grid); out: "
            "(np.quantile(ens, lower_q, axis=0) - c, np.quantile(ens, upper_q, axis=0) + c)"
        )


@dataclass
class QuantileSpaceAdapter(Adapter):
    """Re-read the ensemble at an adapted quantile level.

    The effective miscoverage level is ``alpha_eff = clip(alpha - c, 0, 1)`` and
    the interval is the empirical ensemble interval at that level, i.e.
    ``[quantile(ens, alpha_eff/2), quantile(ens, 1 - alpha_eff/2)]``.

    Note the saturation: with n_members members the widest achievable interval
    is [min, max], so c cannot widen beyond the ensemble support. Track how
    often alpha_eff clips to 0 -- see :attr:`AciResult.saturated_frac`.
    """

    alpha: float = config.ALPHA

    def interval(self, ensemble: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "in: ensemble (n_members, *grid), c (*grid); out: (lower, upper) each (*grid), "
            "read at alpha_eff = clip(alpha - c, 0, 1) via per-gridpoint empirical quantiles"
        )


def make_adapter(space: str = config.ACI_SPACE, **kwargs) -> Adapter:
    """Build the adapter named by ``space`` ('variable' or 'quantile')."""
    if space == "variable":
        return VariableSpaceAdapter(**kwargs)
    if space == "quantile":
        return QuantileSpaceAdapter(**kwargs)
    raise ValueError(f"unknown adaptation space {space!r}; expected 'variable' or 'quantile'")


# --------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------


@dataclass
class AciResult:
    """Output of a full ACI pass over a stream of inits.

    lower, upper : (n_time, *grid) emitted interval endpoints
    c_history    : (n_time, *grid) value of c USED at each step (pre-update)
    saturated_frac : fraction of (time, gridpoint) pairs where the adapter hit
                     its representable limit; None for adapters that cannot
                     saturate.
    """

    lower: np.ndarray
    upper: np.ndarray
    c_history: np.ndarray
    saturated_frac: float | None = None


@dataclass
class DelayedACI:
    """Stateful, per-gridpoint ACI controller with a tau-step feedback delay.

    Args:
        grid_shape: shape of the spatial grid, e.g. ``(121, 240)``. c has this
            shape. Use ``()`` for a single scalar controller.
        alpha: target miscoverage.
        eta: learning rate.
        tau: feedback delay in steps. The (interval, outcome) pair from step t
            is applied to c at step t + tau. tau=0 is the undelayed ACI.
        adapter: adaptation space; defaults to ``make_adapter()``.
        c_init: starting value of c.

    Ordering convention (the test suite encodes this, so do not change it
    without changing tests/test_aci_synthetic.py):

        Within :meth:`step`, the pending update that has come due is applied to
        c FIRST, then the interval for the current init is emitted using the
        updated c. In words: use every piece of feedback that is already
        available at issue time. This means c does not move at all for the
        first tau steps.
    """

    grid_shape: tuple[int, ...] = ()
    alpha: float = config.ALPHA
    eta: float = config.ETA
    tau: int = config.TAU
    adapter: Adapter = field(default_factory=make_adapter)
    c_init: float = 0.0

    c: np.ndarray = field(init=False)
    _pending: deque = field(init=False)

    def __post_init__(self) -> None:
        if self.tau < 0:
            raise ValueError(f"tau must be non-negative, got {self.tau}")
        self.c = np.full(self.grid_shape, float(self.c_init), dtype=np.float64)
        self._pending = deque()

    def reset(self) -> None:
        """Return the controller to its initial state."""
        self.c = np.full(self.grid_shape, float(self.c_init), dtype=np.float64)
        self._pending.clear()

    def step(
        self, ensemble: np.ndarray, y_true: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process one init and return its emitted interval.

        Args:
            ensemble: (n_members, *grid_shape) member values at the target lead.
            y_true: (*grid_shape) verifying truth for THIS init. In a
                retrospective experiment we already hold every outcome; the
                tau-step delay is enforced internally rather than by the
                caller, so it cannot be got wrong at the call site.

        Returns:
            (lower, upper), each (*grid_shape), computed from the current c.

        The (interval, y_true) pair is buffered and used to update c exactly
        tau steps later. See the ordering convention in the class docstring.
        """
        raise NotImplementedError(
            "in: ensemble (n_members, *grid), y_true (*grid); out: (lower, upper) each (*grid). "
            "Applies the due buffered update to self.c, then emits adapter.interval(ensemble, c)."
        )

    def _apply_update(self, lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray) -> None:
        """Apply one ACI update: c += eta * (err - alpha), err = y outside [lower, upper]."""
        raise NotImplementedError(
            "in: lower (*grid), upper (*grid), y_true (*grid); out: None, mutates self.c in place"
        )

    def run(self, ensembles: np.ndarray, truths: np.ndarray) -> AciResult:
        """Run the controller over a whole stream and collect its trace.

        Args:
            ensembles: (n_time, n_members, *grid_shape)
            truths: (n_time, *grid_shape)

        Returns:
            AciResult with lower/upper/c_history each stacked along time.

        This is a thin loop over :meth:`step`; it exists so that 03_verify.py
        and the tests share one implementation of the time loop.
        """
        raise NotImplementedError(
            "in: ensembles (n_time, n_members, *grid), truths (n_time, *grid); out: AciResult"
        )


def raw_interval(
    ensembles: np.ndarray,
    lower_q: float = config.LOWER_QUANTILE,
    upper_q: float = config.UPPER_QUANTILE,
) -> tuple[np.ndarray, np.ndarray]:
    """Uncorrected ensemble interval -- the 'raw' baseline in the headline figure.

    in: ensembles (n_time, n_members, *grid); out: (lower, upper) each (n_time, *grid).
    """
    raise NotImplementedError(
        "in: ensembles (n_time, n_members, *grid); out: (lower, upper) each (n_time, *grid)"
    )
