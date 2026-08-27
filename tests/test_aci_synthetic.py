"""The load-bearing test: ACI must converge to its nominal coverage.

This file is the contract that src/xconformal/aci.py is implemented against. It
is written in full and marked xfail until aci.py exists; when the
implementation lands, these turn XPASS and the markers come off.

TODO(lucas): [Thu] Once aci.py is implemented, delete `pending_aci` and its
decorators. An XPASS in the pytest summary is the signal that a test is ready
to be un-marked.

Why the numbers below are what they are, so you can change them safely:

* The deterministic ACI guarantee bounds the empirical coverage error over T
  steps by roughly (range of c) / (T * eta). With eta = 0.02, T = 20000 and c
  ranging over about 1.1, that bound is ~0.003 -- comfortably inside the 0.01
  tolerance. Halving eta or T doubles the bound. If you shorten the stream to
  make the suite faster, raise the tolerance to match, or the test becomes
  flaky rather than wrong.
* The measured worst deviation over 8 seeds of a reference implementation was
  0.0026, so TOL = 0.01 has ~4x headroom. It is not a tolerance tuned until the
  test passed.
"""

from __future__ import annotations

import numpy as np
import pytest

from xconformal import aci, config

pending_aci = pytest.mark.xfail(
    raises=NotImplementedError,
    strict=False,
    reason="TODO(lucas): [Thu] aci.py is a stub; this test defines its contract",
)

# ---- synthetic stream parameters -----------------------------------------
T = 20_000          # stream length
M = 20              # ensemble members, matching config.N_MEMBERS
ETA = 0.02          # faster than config.ETA so the test converges in T steps
TAU = 5             # matches config.TAU
ALPHA = config.ALPHA
TARGET = 1.0 - ALPHA
TOL = 0.01

# One controller per "gridpoint", each with a differently mis-dispersed
# ensemble, so a scalar-only implementation cannot pass. sigma < 1 is
# under-dispersed (the real failure mode); sigma > 1 is over-dispersed.
SIGMAS = (0.5, 0.7, 1.0, 2.0)


def synthetic_stream(sigmas=SIGMAS, n_time=T, n_members=M, seed=0):
    """A stream where the truth is N(0, 1) and each member is N(0, sigma).

    Returns (ensembles, truths) shaped (n_time, n_members, n_grid) and
    (n_time, n_grid). The marginal distribution is known exactly, so the
    correct answer is known exactly: whatever c does, the emitted intervals
    must cover 90% of outcomes.
    """
    rng = np.random.default_rng(seed)
    sig = np.asarray(sigmas, dtype=float)
    ensembles = rng.normal(0.0, 1.0, size=(n_time, n_members, sig.size)) * sig
    truths = rng.normal(0.0, 1.0, size=(n_time, sig.size))
    return ensembles, truths


def empirical_coverage(truths, lower, upper):
    """Per-gridpoint fraction of outcomes inside [lower, upper]."""
    return ((truths >= lower) & (truths <= upper)).mean(axis=0)


# ---------------------------------------------------------------------------
# Sanity: the stream really is miscalibrated. No ACI involved, so this runs
# green today and proves the fixture is testing something.
# ---------------------------------------------------------------------------


def test_raw_ensemble_interval_is_miscalibrated():
    ensembles, truths = synthetic_stream()
    lower = np.quantile(ensembles, config.LOWER_QUANTILE, axis=1)
    upper = np.quantile(ensembles, config.UPPER_QUANTILE, axis=1)
    cov = empirical_coverage(truths, lower, upper)

    # Under-dispersed members under-cover badly, over-dispersed over-cover.
    assert cov[0] < 0.60, f"sigma=0.5 should badly under-cover, got {cov[0]:.3f}"
    assert cov[1] < 0.75, f"sigma=0.7 should under-cover, got {cov[1]:.3f}"
    assert cov[3] > 0.95, f"sigma=2.0 should over-cover, got {cov[3]:.3f}"
    # Even the correctly-dispersed case under-covers: a 20-member empirical
    # 5th/95th percentile is not a 90% interval for a fresh draw.
    assert cov[2] < 0.87, f"finite-ensemble bias should show, got {cov[2]:.3f}"


# ---------------------------------------------------------------------------
# The contract.
# ---------------------------------------------------------------------------


@pending_aci
def test_aci_converges_to_target_coverage():
    """Per-gridpoint empirical coverage converges to 1 - alpha within TOL."""
    ensembles, truths = synthetic_stream()
    controller = aci.DelayedACI(
        grid_shape=(len(SIGMAS),),
        alpha=ALPHA,
        eta=ETA,
        tau=TAU,
        adapter=aci.VariableSpaceAdapter(),
    )
    result = controller.run(ensembles, truths)

    assert result.lower.shape == truths.shape
    assert result.upper.shape == truths.shape
    assert np.all(result.upper >= result.lower), "emitted intervals must be non-empty"

    cov = empirical_coverage(truths, result.lower, result.upper)
    for sigma, rate in zip(SIGMAS, cov):
        assert abs(rate - TARGET) < TOL, (
            f"sigma={sigma}: coverage {rate:.4f} is {abs(rate - TARGET):.4f} "
            f"from the {TARGET} target (tol {TOL})"
        )


@pending_aci
def test_aci_corrects_a_deliberately_miscalibrated_input():
    """A badly under-dispersed ensemble is pulled up to target; c goes positive.

    This is the direction that matters for the paper: the raw interval covers
    about 51% and ACI has to more than double the gap it spans.
    """
    ensembles, truths = synthetic_stream(sigmas=(0.5,))
    raw_lower = np.quantile(ensembles, config.LOWER_QUANTILE, axis=1)
    raw_upper = np.quantile(ensembles, config.UPPER_QUANTILE, axis=1)
    raw_cov = float(empirical_coverage(truths, raw_lower, raw_upper)[0])
    assert raw_cov < 0.60  # fixture check

    controller = aci.DelayedACI(
        grid_shape=(1,), alpha=ALPHA, eta=ETA, tau=TAU,
        adapter=aci.VariableSpaceAdapter(),
    )
    result = controller.run(ensembles, truths)
    cov = float(empirical_coverage(truths, result.lower, result.upper)[0])

    assert abs(cov - TARGET) < TOL, f"corrected coverage {cov:.4f}, want {TARGET}+-{TOL}"
    assert controller.c[0] > 0.5, (
        f"c must grow to widen an under-dispersed interval, ended at {controller.c[0]:.3f}"
    )
    assert np.all(result.upper - result.lower >= raw_upper - raw_lower - 1e-12), (
        "every corrected interval must be at least as wide as the raw one here"
    )


@pending_aci
def test_aci_shrinks_an_over_dispersed_input():
    """An over-dispersed ensemble is pulled down to target; c goes negative."""
    ensembles, truths = synthetic_stream(sigmas=(2.0,))
    controller = aci.DelayedACI(
        grid_shape=(1,), alpha=ALPHA, eta=ETA, tau=TAU,
        adapter=aci.VariableSpaceAdapter(),
    )
    result = controller.run(ensembles, truths)
    cov = float(empirical_coverage(truths, result.lower, result.upper)[0])

    assert abs(cov - TARGET) < TOL, f"corrected coverage {cov:.4f}, want {TARGET}+-{TOL}"
    assert controller.c[0] < 0.0, f"c must go negative to narrow, ended at {controller.c[0]:.3f}"


@pending_aci
def test_tau_delay_freezes_c_for_the_first_tau_steps():
    """Feedback from step t reaches c at step t + tau, and no sooner.

    Convention (see DelayedACI docstring): within step(), the update that has
    come due is applied BEFORE the interval is emitted. So the c used at steps
    0 .. tau-1 is still c_init, and the first step whose c can differ is tau.
    tau=0 means the unavoidable one-step delay only -- the pair from step t
    reaches c at step t+1, since the outcome of an interval cannot be known
    before it is emitted.
    """
    ensembles, truths = synthetic_stream(sigmas=(0.5,), n_time=200)

    for tau in (0, 1, 5, 20):
        controller = aci.DelayedACI(
            grid_shape=(1,), alpha=ALPHA, eta=ETA, tau=tau,
            adapter=aci.VariableSpaceAdapter(), c_init=0.0,
        )
        result = controller.run(ensembles, truths)
        frozen = max(tau, 1)
        assert np.all(result.c_history[:frozen] == 0.0), (
            f"tau={tau}: c moved during the first {frozen} steps: "
            f"{result.c_history[:frozen].ravel()}"
        )
        assert result.c_history[frozen] != 0.0, (
            f"tau={tau}: c should have moved at step {frozen} but did not"
        )


@pending_aci
def test_delay_costs_coverage_accuracy_but_not_convergence():
    """A longer delay adds lag, not bias: both taus still land on target."""
    ensembles, truths = synthetic_stream(sigmas=(0.5, 2.0))
    rates = {}
    for tau in (1, TAU):
        controller = aci.DelayedACI(
            grid_shape=(2,), alpha=ALPHA, eta=ETA, tau=tau,
            adapter=aci.VariableSpaceAdapter(),
        )
        result = controller.run(ensembles, truths)
        rates[tau] = empirical_coverage(truths, result.lower, result.upper)

    for tau, cov in rates.items():
        assert np.all(np.abs(cov - TARGET) < TOL), f"tau={tau}: coverage {cov}"


# ---------------------------------------------------------------------------
# Adaptation space: both must satisfy the same interface. They do NOT have the
# same behaviour, and that difference is a result, not a bug.
# ---------------------------------------------------------------------------


@pending_aci
def test_both_adaptation_spaces_satisfy_the_adapter_interface():
    ensembles, _ = synthetic_stream(n_time=10)
    c = np.zeros(len(SIGMAS))
    for space in ("variable", "quantile"):
        adapter = aci.make_adapter(space)
        lower, upper = adapter.interval(ensembles[0], c)
        assert lower.shape == upper.shape == (len(SIGMAS),)
        assert np.all(upper >= lower)
        # c == 0 must reproduce the raw nominal interval, so that "raw" is just
        # this adapter frozen at zero.
        raw_lo = np.quantile(ensembles[0], config.LOWER_QUANTILE, axis=0)
        raw_hi = np.quantile(ensembles[0], config.UPPER_QUANTILE, axis=0)
        assert np.allclose(lower, raw_lo, atol=1e-9), f"{space}: c=0 must give the raw lower"
        assert np.allclose(upper, raw_hi, atol=1e-9), f"{space}: c=0 must give the raw upper"

        # Monotone in c: more correction is a wider interval, never narrower.
        wide_lo, wide_hi = adapter.interval(ensembles[0], c + 0.05)
        assert np.all(wide_lo <= lower + 1e-12) and np.all(wide_hi >= upper - 1e-12)


@pending_aci
def test_quantile_space_reports_saturation_on_an_underdispersed_stream():
    """Quantile-space adaptation cannot widen past the ensemble min/max.

    A reference implementation of this exact stream reaches only ~0.63
    coverage at sigma=0.5 while c runs away past 100, because alpha - c clips
    at 0 and the interval is already [min, max]. Variable space reaches 0.900
    on the same stream. The controller must therefore REPORT that it saturated
    rather than silently returning a number that looks like a coverage result.

    TODO(lucas): [Fri] This is the deciding evidence for config.ACI_SPACE.
    Confirm it holds on the real ensemble before writing it into the paper --
    on real data the ensemble is less pathologically under-dispersed than
    sigma=0.5, so quantile space may saturate only in the high-p_t bins, which
    would itself be a result worth reporting.
    """
    ensembles, truths = synthetic_stream(sigmas=(0.5,))
    controller = aci.DelayedACI(
        grid_shape=(1,), alpha=ALPHA, eta=ETA, tau=TAU,
        adapter=aci.QuantileSpaceAdapter(alpha=ALPHA),
    )
    result = controller.run(ensembles, truths)

    assert result.saturated_frac is not None, (
        "QuantileSpaceAdapter must report a saturation fraction; a silent None "
        "here would hide the failure mode this test exists to catch"
    )
    assert result.saturated_frac > 0.1, (
        f"expected heavy saturation on an under-dispersed stream, "
        f"got {result.saturated_frac:.3f}"
    )
    cov = float(empirical_coverage(truths, result.lower, result.upper)[0])
    assert cov < TARGET - TOL, (
        f"quantile space should fail to reach target here (reference ~0.63), got {cov:.4f}"
    )


@pending_aci
def test_reset_restores_initial_state():
    ensembles, truths = synthetic_stream(sigmas=(0.5,), n_time=100)
    controller = aci.DelayedACI(
        grid_shape=(1,), alpha=ALPHA, eta=ETA, tau=TAU,
        adapter=aci.VariableSpaceAdapter(), c_init=0.25,
    )
    first = controller.run(ensembles, truths)
    controller.reset()
    assert np.all(controller.c == 0.25)
    second = controller.run(ensembles, truths)
    assert np.array_equal(first.c_history, second.c_history), "run() must be reproducible"
