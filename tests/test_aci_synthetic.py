"""The load-bearing test: ACI must converge to its nominal coverage.

This file is the contract that src/xconformal/aci.py is implemented against. It
is written in full and marked xfail until aci.py exists; when the
implementation lands, these turn XPASS and the markers come off.

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
    # ACI widens on AGGREGATE, and cannot promise it step by step: the first
    # update on a COVERED outcome sets c = eta*(0 - alpha) < 0, so at least one
    # early interval is necessarily narrower than raw. Measured on this stream:
    # exactly 1 step of 20000, at step 5, c = -0.002. Asserting the per-step
    # version would be asserting something the update rule contradicts.
    assert (result.upper - result.lower).mean() > (raw_upper - raw_lower).mean(), (
        "the corrected interval must be wider than raw on average for an "
        "under-dispersed ensemble"
    )


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


def test_quantile_space_reports_saturation_on_an_underdispersed_stream():
    """Quantile-space adaptation cannot widen past the ensemble min/max.

    A reference implementation of this exact stream reaches only ~0.63
    coverage at sigma=0.5 while c runs away past 100, because alpha - c clips
    at 0 and the interval is already [min, max]. Variable space reaches 0.900
    on the same stream. The controller must therefore REPORT that it saturated
    rather than silently returning a number that looks like a coverage result.

    This was the deciding evidence for config.ACI_SPACE, which is now locked to
    standardized variable space. The test remains as the contract that
    saturation is REPORTED rather than hidden, since quantile space is still
    reachable via `03_verify.py --space quantile` for the appendix.
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


# ---------------------------------------------------------------------------
# Irregular init spacing.
#
# The real archive is not evenly spaced: 2020 is thinned to every 4th day and
# 2021 is daily, so the stream changes cadence mid-run at the year boundary.
# tau is 5 DAYS throughout -- a physical delay set by the forecast lead, not a
# number of rows in an array. An implementation that counts steps instead of
# days silently uses 20 days of delay in the thinned segment and 5 in the dense
# one, and nothing about the output looks wrong.
#
# This test pins the datetime-queue implementation: hold
# (verification_time, err) and, at each init, apply every entry whose
# verification_time has passed. The year boundary is then not a special case;
# it is just a change in how often the queue is drained.
# ---------------------------------------------------------------------------

WIDE_SPACING_DAYS = 4
TAU_DAYS = 5
T0 = np.datetime64("2020-01-02T00", "h")


def spaced_init_times(n_time: int, wide_fraction: float = 0.25) -> np.ndarray:
    """Init times: the first `wide_fraction` at 4-day spacing, then daily."""
    n_wide = int(n_time * wide_fraction)
    deltas = np.array([WIDE_SPACING_DAYS] * n_wide + [1] * (n_time - n_wide))
    offsets = np.concatenate([[0], np.cumsum(deltas)[:-1]])
    return T0 + offsets.astype("timedelta64[D]")


def daily_init_times(n_time: int) -> np.ndarray:
    return T0 + np.arange(n_time).astype("timedelta64[D]")


def _run(times, ensembles, truths, n_grid):
    controller = aci.DelayedACI(
        grid_shape=(n_grid,), alpha=ALPHA, eta=ETA,
        tau=np.timedelta64(TAU_DAYS, "D"), adapter=aci.VariableSpaceAdapter(),
    )
    return controller, controller.run(ensembles, truths, init_times=times)


def test_irregular_spacing_still_converges():
    """(a) Coverage reaches 1 - alpha despite the cadence change."""
    ensembles, truths = synthetic_stream(sigmas=(0.5, 1.0))
    times = spaced_init_times(len(truths))
    _, result = _run(times, ensembles, truths, 2)

    cov = empirical_coverage(truths, result.lower, result.upper)
    for sigma, rate in zip((0.5, 1.0), cov):
        assert abs(rate - TARGET) < TOL, (
            f"sigma={sigma}: coverage {rate:.4f} is {abs(rate - TARGET):.4f} from "
            f"{TARGET} under irregular spacing (tol {TOL})"
        )


def test_no_update_is_applied_before_its_verification_time():
    """(b) The no-lookahead guarantee, checked against the audit trail.

    Every update must be applied at an init time at or after the verification
    time of the forecast it came from. This is the property that makes the
    experiment honest: applying feedback early is using an outcome we could not
    have known, and it would improve coverage for free.
    """
    ensembles, truths = synthetic_stream(sigmas=(0.5, 1.0), n_time=2000)
    times = spaced_init_times(len(truths))
    _, result = _run(times, ensembles, truths, 2)

    assert result.update_log, "the controller must expose which updates it applied when"
    for applied_at, verification_time in result.update_log:
        assert applied_at >= verification_time, (
            f"update verifying at {verification_time} was applied at {applied_at} -- "
            f"that is {(verification_time - applied_at) / np.timedelta64(1, 'D'):.0f} "
            f"days of lookahead"
        )

    # ...and nothing due is left unapplied: the queue is drained eagerly, not lazily.
    tau = np.timedelta64(TAU_DAYS, "D")
    due = int(np.sum((times + tau) <= times[-1]))
    assert len(result.update_log) == due, (
        f"{len(result.update_log)} updates applied but {due} were due by the last init"
    )


def test_spacing_change_leaves_no_trace_beyond_updates_in_flight():
    """(c) The cadence change is not a special case, only a different queue depth.

    c itself legitimately differs between the two runs and is NOT asserted
    equal: at 4-day spacing with tau = 5 days, feedback arrives 2 steps after
    issue rather than 5, so the thinned segment runs on strictly more
    information per step. What must match is the *mechanism* -- once the stream
    has been daily for longer than tau, a controller that saw a cadence change
    must be indistinguishable from one that never did.
    """
    ensembles, truths = synthetic_stream(sigmas=(0.5, 1.0), n_time=2000)
    n_time = len(truths)
    switched, res_switched = _run(spaced_init_times(n_time), ensembles, truths, 2)
    uniform, res_uniform = _run(daily_init_times(n_time), ensembles, truths, 2)

    # In a steady daily stream, exactly tau updates are in flight at any time.
    n_wide = int(n_time * 0.25)
    tail = slice(n_wide + TAU_DAYS + 1, None)
    assert np.all(res_switched.pending_depth[tail] == TAU_DAYS), (
        f"after the change to daily, in-flight updates should settle to {TAU_DAYS}, "
        f"saw {np.unique(res_switched.pending_depth[tail])}"
    )
    assert np.all(res_uniform.pending_depth[tail] == TAU_DAYS)

    # In the 4-day segment the same rule gives a shallower queue: an entry
    # issued at day d is due at d+5 and drained at the next init, day d+8.
    steady_wide = slice(4, n_wide)
    expected = int(np.ceil(TAU_DAYS / WIDE_SPACING_DAYS))
    assert np.all(res_switched.pending_depth[steady_wide] == expected), (
        f"at {WIDE_SPACING_DAYS}-day spacing with tau={TAU_DAYS}d the queue should "
        f"hold {expected}, saw {np.unique(res_switched.pending_depth[steady_wide])}"
    )

    # Both drain one update per step once daily, so neither loses or repeats work.
    for name, result in (("switched", res_switched), ("uniform", res_uniform)):
        applied = np.asarray(result.updates_applied)
        assert np.all(applied[tail] == 1), (
            f"{name}: a steady daily stream must apply exactly one update per step, "
            f"saw {np.unique(applied[tail])}"
        )

    # And the cadence change costs nothing in the end: both converge.
    for name, result in (("switched", res_switched), ("uniform", res_uniform)):
        cov = empirical_coverage(truths, result.lower, result.upper)
        assert np.all(np.abs(cov - TARGET) < 0.05), f"{name}: coverage {cov}"


def test_tau_is_physical_days_not_array_rows():
    """A thinned stream must not silently multiply the delay by the stride.

    With 4-day inits and tau = 5 days, feedback is due after 5 days and drained
    at the next init (8 days). A step-counting implementation would wait 5
    INITS, i.e. 20 days -- four times the intended delay, with no visible
    symptom.
    """
    ensembles, truths = synthetic_stream(sigmas=(0.5,), n_time=400)
    times = T0 + (np.arange(400) * WIDE_SPACING_DAYS).astype("timedelta64[D]")
    controller = aci.DelayedACI(
        grid_shape=(1,), alpha=ALPHA, eta=ETA,
        tau=np.timedelta64(TAU_DAYS, "D"), adapter=aci.VariableSpaceAdapter(),
    )
    result = controller.run(ensembles, truths, init_times=times)

    # c must move at step 2 (day 8 >= day 5), not step 5.
    assert np.all(result.c_history[:2] == 0.0), "no feedback is due before day 5"
    assert result.c_history[2] != 0.0, (
        "feedback issued at day 0 is due at day 5 and must be applied at the "
        "first init at or after it (day 8, step 2)"
    )
    assert np.all(result.pending_depth[2:] == 2)


# ---------------------------------------------------------------------------
# Warm start: cycling the calibration year so evaluation opens settled.
# ---------------------------------------------------------------------------


def test_warm_start_converges_and_clears_the_queue_between_passes():
    """c carries across passes; the in-flight queue does not.

    Carrying the queue would be a real bug rather than an inefficiency: the
    December verification times still pending at a pass boundary would all come
    due at the next pass's first January init, in the wrong order relative to
    that pass's own forecasts.
    """
    ensembles, truths = synthetic_stream(sigmas=(0.5, 2.0), n_time=200)
    times = daily_init_times(len(truths))
    controller = aci.DelayedACI(
        grid_shape=(2,), alpha=ALPHA, eta=ETA,
        tau=np.timedelta64(TAU_DAYS, "D"), adapter=aci.VariableSpaceAdapter(),
    )
    n_passes, means, result = aci.warm_start(
        controller, ensembles, truths, times, tol=0.01, verbose=False
    )

    assert 1 < n_passes <= 25, f"expected several passes, got {n_passes}"
    assert controller.pending == 0, "the queue must be empty at a pass boundary"
    assert abs(means[-1] - means[-2]) < 0.01 * abs(means[-2]), "did not meet the tolerance"

    # A warm-started controller opens the next stream much closer to target
    # than a cold one does.
    fresh = aci.DelayedACI(
        grid_shape=(2,), alpha=ALPHA, eta=ETA,
        tau=np.timedelta64(TAU_DAYS, "D"), adapter=aci.VariableSpaceAdapter(),
    )
    cold = fresh.run(ensembles, truths, init_times=times)
    warm = controller.run(ensembles, truths, init_times=times)
    cold_gap = np.abs(empirical_coverage(truths, cold.lower, cold.upper) - TARGET)
    warm_gap = np.abs(empirical_coverage(truths, warm.lower, warm.upper) - TARGET)
    assert np.all(warm_gap < cold_gap), (
        f"warm start should open closer to target: warm {warm_gap} vs cold {cold_gap}"
    )


def test_flush_preserves_the_no_lookahead_audit_trail():
    """Even a flushed update is logged at a time no earlier than it verified."""
    ensembles, truths = synthetic_stream(sigmas=(0.5,), n_time=50)
    times = daily_init_times(len(truths))
    controller = aci.DelayedACI(
        grid_shape=(1,), alpha=ALPHA, eta=ETA,
        tau=np.timedelta64(TAU_DAYS, "D"), adapter=aci.VariableSpaceAdapter(),
    )
    controller.run(ensembles, truths, init_times=times)
    in_flight = controller.pending
    assert in_flight == TAU_DAYS
    assert controller.flush() == in_flight
    assert controller.pending == 0
    for applied_at, verification_time in controller._update_log:
        assert applied_at >= verification_time
