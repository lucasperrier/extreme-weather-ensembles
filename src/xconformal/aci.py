"""Online adaptive conformal inference (ACI) with a delayed update.

Following Asch et al. (arXiv:2606.19642): the ensemble's nominal
[LOWER_QUANTILE, UPPER_QUANTILE] interval is corrected by a running c_t,
updated as

    c <- c + eta * (err - alpha)

where ``err`` is 1 if the outcome fell outside the emitted interval and 0
otherwise. The outcome of a forecast issued at t is only observed at t + tau,
so the update from t is applied at the first init at or after t + tau.

Three things make this module non-trivial and all three are deliberate:

1.  **Delay bookkeeping is by DATETIME, never by array index.** The queue holds
    ``(verification_time, err_field)`` and at each init every entry whose
    verification_time has passed is applied. tau is a physical delay in days
    set by the forecast lead, not a number of rows. The real archive is thinned
    to every 4th day in 2020 and daily in 2021, so a step-counting
    implementation would silently use 20 days of delay in one segment and 5 in
    the other, with nothing in the output looking wrong. The 2020->2021
    boundary is not special-cased anywhere: it is just a change in how often
    the queue drains.

2.  **Vectorised over the grid.** c is an array with the shape of the grid
    (lat, lon); every gridpoint runs its own independent controller. There are
    no Python loops over gridpoints. The only Python loop is over time, which
    is inherently sequential.

3.  **Adaptation space is pluggable.** RATIFIED 2026-08-27: the paper uses
    STANDARDIZED VARIABLE SPACE (config.ACI_SPACE, config.ACI_STANDARDIZE),

        lower = q05 - c * s,   upper = q95 + c * s

    with s the per-gridpoint std of the 2020 0Z t2m series
    (thresholds.load_aci_scale, calibration year only, so the verification year
    never informs the scaling).

    The decisive argument is not eta transfer, it is the guarantee itself. With
    M = 20 members, quantile-space adaptation cannot widen an interval past the
    ensemble min/max: once alpha - c reaches 0 the interval is [min, max] and
    further correction does nothing. On underdispersed extremes that breaks
    convergence outright, unless one is willing to emit degenerate infinite
    intervals. Variable space is unbounded and so is the guarantee-preserving
    choice at small M. QuantileSpaceAdapter remains behind
    ``--space quantile`` as the appendix ablation, where
    :attr:`AciResult.saturated_frac` is its required diagnostic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import config
from .coverage import miscoverage

__all__ = [
    "Adapter",
    "VariableSpaceAdapter",
    "QuantileSpaceAdapter",
    "make_adapter",
    "DelayedACI",
    "AciResult",
    "raw_interval",
]

# Only used to synthesize a daily grid when a caller passes no init times.
_DEFAULT_EPOCH = np.datetime64("2020-01-01T00", "h")


def _as_timedelta(tau) -> np.timedelta64:
    """Accept tau as a number of days or as a timedelta64."""
    if isinstance(tau, np.timedelta64):
        return tau.astype("timedelta64[h]")
    return np.timedelta64(int(tau), "D").astype("timedelta64[h]")


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
        non-increasing for ``lower``. ACI's convergence argument needs this.
      - c == 0 must reproduce the raw ensemble interval at the nominal
        quantiles, so the raw baseline is this class with c fixed at 0.
      - ``saturation_mask`` reports where the adapter has hit a representable
        limit, or None if it cannot saturate.
    """

    def interval(self, ensemble: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def saturation_mask(self, c: np.ndarray) -> np.ndarray | None:
        """Where further correction has no effect. None if not applicable."""
        return None


@dataclass
class VariableSpaceAdapter(Adapter):
    """Pad the nominal ensemble quantiles by +/- c*scale, in variable units.

    ``scale`` is the per-gridpoint standardization field -- in production,
    thresholds.load_aci_scale(), the 2020 0Z t2m std. It makes c dimensionless
    so one eta works everywhere. None means 1.0 (raw Kelvin), which is what the
    synthetic tests use since their stream is already unit-variance.

    This adapter cannot saturate: c is unbounded above, so any required width
    is reachable.
    """

    lower_q: float = config.LOWER_QUANTILE
    upper_q: float = config.UPPER_QUANTILE
    scale: np.ndarray | float | None = None

    def interval(self, ensemble: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale = 1.0 if self.scale is None else self.scale
        lower = np.quantile(ensemble, self.lower_q, axis=0)
        upper = np.quantile(ensemble, self.upper_q, axis=0)
        pad = c * scale
        return lower - pad, upper + pad


@dataclass
class QuantileSpaceAdapter(Adapter):
    """Re-read the ensemble at an adapted quantile level.

    The effective miscoverage level is ``alpha_eff = clip(alpha - c, 0, 1)`` and
    the interval is the empirical ensemble interval at that level.

    Note the saturation: with M members the widest achievable interval is
    [min, max], so c cannot widen beyond the ensemble support. See
    :attr:`AciResult.saturated_frac`.
    """

    alpha: float = config.ALPHA

    def interval(self, ensemble: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        alpha_eff = np.clip(self.alpha - np.asarray(c, dtype=float), 0.0, 1.0)
        ordered = np.sort(ensemble, axis=0)
        return (
            _quantile_at(ordered, alpha_eff / 2.0),
            _quantile_at(ordered, 1.0 - alpha_eff / 2.0),
        )

    def saturation_mask(self, c: np.ndarray) -> np.ndarray:
        raw = self.alpha - np.asarray(c, dtype=float)
        return (raw <= 0.0) | (raw >= 1.0)


def _quantile_at(ordered: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-gridpoint linear-interpolated quantile at a per-gridpoint level.

    in: ordered (n_members, *grid) sorted ascending along axis 0, q (*grid) or
    scalar in [0, 1]; out: (*grid).

    np.quantile cannot take a different q per gridpoint, and looping over
    gridpoints would defeat the whole vectorisation. This reproduces
    np.quantile's default "linear" method exactly, which matters because the
    c == 0 case must equal the raw interval to the last bit.
    """
    n = ordered.shape[0]
    position = np.asarray(q, dtype=float) * (n - 1)
    low = np.floor(position).astype(np.intp)
    high = np.ceil(position).astype(np.intp)
    weight = position - low
    lower = np.take_along_axis(ordered, low[np.newaxis, ...], axis=0)[0]
    upper = np.take_along_axis(ordered, high[np.newaxis, ...], axis=0)[0]
    return lower * (1.0 - weight) + upper * weight


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

    lower, upper     : (n_time, *grid) emitted interval endpoints
    c_history        : (n_time, *grid) the c USED at each init, i.e. after the
                       updates due at that init were applied
    init_times       : (n_time,) datetime64 init times
    pending_depth    : (n_time,) updates still in flight after each init
    updates_applied  : (n_time,) updates applied at each init
    update_log       : (applied_at_init_time, verification_time) in application
                       order -- the audit trail for the no-lookahead guarantee
    saturated_frac   : fraction of (init, gridpoint) pairs where the adapter hit
                       its representable limit; None for adapters that cannot
                       saturate
    """

    lower: np.ndarray
    upper: np.ndarray
    c_history: np.ndarray
    init_times: np.ndarray | None = None
    pending_depth: np.ndarray | None = None
    updates_applied: np.ndarray | None = None
    update_log: list = field(default_factory=list)
    saturated_frac: float | None = None


@dataclass
class DelayedACI:
    """Stateful, per-gridpoint ACI controller with a datetime feedback delay.

    Args:
        grid_shape: shape of the spatial grid, e.g. ``(121, 240)``. c has this
            shape. Use ``()`` for a single scalar controller.
        alpha: target miscoverage.
        eta: learning rate.
        tau: feedback delay, as a number of DAYS or a numpy timedelta64. The
            outcome of a forecast issued at t verifies at t + tau and is
            applied at the first init at or after that.
        adapter: adaptation space; defaults to ``make_adapter()``.
        c_init: starting value of c.

    Ordering convention (the test suite encodes this):

        Within :meth:`step`, every update that has come due is applied to c
        FIRST, then the interval for the current init is emitted using the
        updated c. In words: use every piece of feedback that is already
        available at issue time. A forecast's own outcome is never available at
        its own init, so c cannot move before the first init at or after
        ``first_init + tau``.
    """

    grid_shape: tuple[int, ...] = ()
    alpha: float = config.ALPHA
    eta: float = config.ETA
    tau: object = config.TAU
    adapter: Adapter = field(default_factory=make_adapter)
    c_init: float = 0.0

    c: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._tau = _as_timedelta(self.tau)
        if self._tau < np.timedelta64(0, "h"):
            raise ValueError(f"tau must be non-negative, got {self.tau}")
        self.reset()

    def reset(self) -> None:
        """Return the controller to its initial state."""
        self.c = np.full(self.grid_shape, float(self.c_init), dtype=np.float64)
        self._pending: deque = deque()
        self._update_log: list = []

    @property
    def pending(self) -> int:
        """How many updates are currently in flight."""
        return len(self._pending)

    def step(self, ensemble, y_true, init_time) -> tuple[np.ndarray, np.ndarray]:
        """Process one init and return its emitted interval.

        Args:
            ensemble: (n_members, *grid_shape) member values at the target lead.
            y_true: (*grid_shape) verifying truth for THIS init. In a
                retrospective experiment we already hold every outcome; the
                delay is enforced internally rather than by the caller, so it
                cannot be got wrong at the call site.
            init_time: datetime64 of this init.

        Returns:
            (lower, upper), each (*grid_shape), from the c in force after this
            init's due updates were applied.
        """
        init_time = np.datetime64(init_time).astype("datetime64[h]")
        n_applied = self._drain(init_time)

        lower, upper = self.adapter.interval(ensemble, self.c)
        # Asch's convention: an empty or crossed interval counts as a miss.
        # covered() is `lower <= y <= upper`, which is False whenever
        # lower > upper, so err == 1 falls out without a special case -- and
        # NaN anywhere also compares False, so a corrupt field is a miss rather
        # than a silent pass.
        err = miscoverage(y_true, lower, upper)
        self._pending.append((init_time + self._tau, err))
        return lower, upper, n_applied

    def _drain(self, init_time: np.datetime64) -> int:
        """Apply every queued update whose verification time has passed."""
        n_applied = 0
        while self._pending and self._pending[0][0] <= init_time:
            verification_time, err = self._pending.popleft()
            self._apply_update(err)
            self._update_log.append((init_time, verification_time))
            n_applied += 1
        return n_applied

    def _apply_update(self, err: np.ndarray) -> None:
        """One ACI update: c += eta * (err - alpha), in place, over the grid."""
        self.c += self.eta * (err - self.alpha)

    def run(self, ensembles, truths, init_times=None) -> AciResult:
        """Run the controller over a whole stream and collect its trace.

        Args:
            ensembles: (n_time, n_members, *grid_shape)
            truths: (n_time, *grid_shape)
            init_times: (n_time,) datetime64. Defaults to a daily grid, which
                is what the synthetic tests use; the real archive passes its
                actual init times, gaps and all.

        Returns:
            AciResult with the emitted intervals, the c trajectory and the
            delay audit trail.

        Does not reset first: call :meth:`reset` to rerun from scratch.
        """
        ensembles = np.asarray(ensembles)
        truths = np.asarray(truths)
        n_time = ensembles.shape[0]
        if truths.shape[0] != n_time:
            raise ValueError(
                f"{n_time} ensembles but {truths.shape[0]} truths"
            )
        if init_times is None:
            init_times = _DEFAULT_EPOCH + np.arange(n_time).astype("timedelta64[D]")
        init_times = np.asarray(init_times).astype("datetime64[h]")
        if init_times.shape[0] != n_time:
            raise ValueError(f"{n_time} ensembles but {init_times.shape[0]} init times")
        if np.any(np.diff(init_times) <= np.timedelta64(0, "h")):
            raise ValueError("init_times must be strictly increasing")

        lower = np.empty((n_time, *self.grid_shape), dtype=np.float64)
        upper = np.empty_like(lower)
        c_history = np.empty_like(lower)
        pending_depth = np.empty(n_time, dtype=int)
        updates_applied = np.empty(n_time, dtype=int)

        saturated = 0
        n_saturation_checks = 0
        log_start = len(self._update_log)

        for t in range(n_time):
            lo, hi, n_applied = self.step(ensembles[t], truths[t], init_times[t])
            c_history[t] = self.c
            lower[t] = lo
            upper[t] = hi
            updates_applied[t] = n_applied
            pending_depth[t] = len(self._pending)

            mask = self.adapter.saturation_mask(self.c)
            if mask is not None:
                saturated += int(np.count_nonzero(mask))
                n_saturation_checks += int(np.size(mask))

        if not np.all(np.isfinite(c_history)):
            bad = int(np.count_nonzero(~np.isfinite(c_history)))
            raise FloatingPointError(
                f"c became non-finite at {bad} (init, gridpoint) pairs -- this is a "
                f"bug, not a result. Check for NaNs in the ensemble or the truth."
            )

        return AciResult(
            lower=lower,
            upper=upper,
            c_history=c_history,
            init_times=init_times,
            pending_depth=pending_depth,
            updates_applied=updates_applied,
            update_log=self._update_log[log_start:],
            saturated_frac=(saturated / n_saturation_checks) if n_saturation_checks else None,
        )


def raw_interval(
    ensembles: np.ndarray,
    lower_q: float = config.LOWER_QUANTILE,
    upper_q: float = config.UPPER_QUANTILE,
) -> tuple[np.ndarray, np.ndarray]:
    """Uncorrected ensemble interval -- the 'raw' baseline in the headline figure.

    in: ensembles (n_time, n_members, *grid); out: (lower, upper) each
    (n_time, *grid). The member axis is 1.
    """
    ensembles = np.asarray(ensembles)
    return (
        np.quantile(ensembles, lower_q, axis=1),
        np.quantile(ensembles, upper_q, axis=1),
    )
