#!/usr/bin/env python
"""Why is January-February ACI coverage below the rest of the year?

03_verify.py's CHECK 2 reports the symptom. This script is the diagnosis, and
it exists because the paper makes a claim about it: the residual is the
controller's adaptation lag under a real year-to-year shift in the padding the
data requires, not a defect of the warm start.

Four controls, each isolating one candidate cause:

  1  CONVERGENCE  -- rerun the warm start at tighter tolerances. If the dip is
     unconverged warm start, tightening removes it.
  2  DATA         -- freeze c at the field the warm start hands over and switch
     the controller off. If Jan-Feb is intrinsically harder, the dip survives
     with no dynamics at all.
  3  EQUILIBRIUM  -- compute the per-gridpoint equilibrium padding of each year
     directly. c*(gridpoint) is exactly the (1-alpha) quantile of the
     standardized required padding
         d_t = max(q05 - y, y - q95) / s
     because the outcome is covered exactly when c >= d_t. Static; no dynamics,
     no bisection.
  4  STARTUP      -- rerun the evaluation year from oracle starting fields. If
     the dip is the startup gap, it disappears when c_0 already knows 2021.

Controls 3 and 4 use verification-year data to build reference fields. They are
DIAGNOSTICS ONLY and must never feed a reported number -- they are the oracle
the honest configuration is measured against.

    source /workspace/env.sh
    python scripts/09_diagnose_transient.py

This is the slow script in the repo -- it runs the controller tens of thousands
of times over the grid. Expect tens of minutes, not the ~90 s of 03_verify.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
from importlib import import_module  # noqa: E402

from xconformal import aci, config, coverage, grid, thresholds  # noqa: E402

verify = import_module("03_verify")


def monthly(cov, months, weights):
    return np.array([coverage.marginal_coverage(cov[months == m], weights=weights)[0]
                     for m in range(1, 13)])


def verdict(rates: np.ndarray) -> tuple[bool, str]:
    jf, rest = rates[:2], rates[2:]
    inside = bool(((jf >= rest.min()) & (jf <= rest.max())).all())
    return inside, (f"Jan-Feb {jf.min():.4f}..{jf.max():.4f}  "
                    f"Mar-Dec {rest.min():.4f}..{rest.max():.4f}  "
                    f"-> {'PASS' if inside else 'FAIL'}")


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args(argv)
    config.validate_paths()

    present, _ = verify.archive_inits(config.INIT_START, config.INIT_END)
    init_times = np.array(present, dtype="datetime64[h]")
    valid_times = init_times + np.timedelta64(config.LEAD_DAYS, "D")
    print(f"loading {len(present)} inits ...", flush=True)
    ens = verify.load_archive(present)
    truth = verify.load_truth(valid_times)

    w2d = coverage.latitude_weights(grid.era5_latitudes())[:, None]
    weights = w2d[None, ...]
    scale = thresholds.load_aci_scale().values

    keep = init_times >= np.datetime64(config.EVAL_START, "h")
    calib = ~keep
    eval_times = init_times[keep]
    months = eval_times.astype("datetime64[M]").astype(int) % 12 + 1
    truth_eval = truth[keep]

    def wmean(field):
        ww = np.broadcast_to(w2d, field.shape)
        return float((field * ww).sum() / ww.sum())

    def controller():
        return aci.DelayedACI(
            grid_shape=truth.shape[1:], alpha=config.ALPHA, eta=config.ETA,
            tau=np.timedelta64(config.TAU, "D"),
            adapter=aci.VariableSpaceAdapter(scale=scale))

    lo, hi = aci.raw_interval(ens[keep])
    d_eval = np.maximum(lo - truth_eval, truth_eval - hi) / scale
    lo_c = np.quantile(ens[calib], config.LOWER_QUANTILE, axis=1)
    hi_c = np.quantile(ens[calib], config.UPPER_QUANTILE, axis=1)
    d_calib = np.maximum(lo_c - truth[calib], truth[calib] - hi_c) / scale
    q = 1.0 - config.ALPHA
    field_2020 = np.quantile(d_calib, q, axis=0)
    field_2021 = np.quantile(d_eval, q, axis=0)

    # ---- 1  convergence -------------------------------------------------
    print("\n=== CONTROL 1: is the warm start converged? ===")
    fields = {}
    for tol, max_passes, label in [(1e-2, 25, "shipped, tol 1e-2"),
                                   (1e-3, 200, "tol 1e-3"),
                                   (1e-4, 400, "tol 1e-4"),
                                   (0.0, 60, "60 passes, no early stop")]:
        ctrl = controller()
        n, means, _ = aci.warm_start(ctrl, ens[calib], truth[calib],
                                     init_times[calib], max_passes=max_passes,
                                     tol=tol, verbose=False)
        # Snapshot c BEFORE the evaluation run: ctrl.run mutates ctrl.c, so
        # reading it afterwards reports end-of-2021 c, not c_0.
        fields[label] = ctrl.c.copy()
        res = ctrl.run(ens[keep], truth[keep], init_times=eval_times)
        cov = coverage.covered(truth_eval, res.lower, res.upper)
        rates = monthly(cov, months, weights)
        _, line = verdict(rates)
        print(f"  {label:<26s} {n:>3d} passes  c_0 mean {means[-1]:+.5f} "
              f"(area-wtd {wmean(fields[label]):+.5f})  marginal "
              f"{coverage.marginal_coverage(cov, weights=weights)[0]:.4f}")
        print(f"  {'':<26s} {line}")
    warm_field = fields["60 passes, no early stop"]

    # ---- 2  data --------------------------------------------------------
    print("\n=== CONTROL 2: is Jan-Feb intrinsically harder? "
          "(controller OFF, c frozen at a field) ===")
    for label, field in (("2020 static equilibrium", field_2020),
                         ("warm-start field handed to 2021", warm_field),
                         ("2021 static equilibrium (ORACLE)", field_2021)):
        cov = d_eval <= field
        rates = monthly(cov, months, weights)
        _, line = verdict(rates)
        print(f"  {label:<34s} full-year {rates.mean():.4f}   {line}")

    # ---- 3  equilibrium -------------------------------------------------
    print("\n=== CONTROL 3: do the two years want the same padding? ===")
    print(f"  per-gridpoint equilibrium padding, area-weighted mean")
    print(f"    2020 calibration  ({int(calib.sum()):3d} inits) {wmean(field_2020):+.5f}")
    print(f"    2021 verification ({int(keep.sum()):3d} inits) {wmean(field_2021):+.5f}")
    print(f"    difference                        {wmean(field_2021) - wmean(field_2020):+.5f}")
    print(f"  spatial correlation of the two fields: "
          f"{np.corrcoef(field_2020.ravel(), field_2021.ravel())[0, 1]:.4f}")

    # ---- 4  startup -----------------------------------------------------
    print("\n=== CONTROL 4: is it the startup gap? ===")
    runs = {}
    # Reuse the converged field Control 1 already produced rather than paying
    # for a second 60-pass warm start; it is the same computation.
    ctrl = controller(); ctrl.c = warm_field.copy()
    runs["C  warm start on 2020, fully converged"] = ctrl

    ctrl = controller(); ctrl.c = field_2021.copy()
    runs["B  ORACLE start at the 2021 equilibrium field"] = ctrl

    # Cycling 2021 is the expensive control -- 360 inits per pass against the
    # calibration year's 92 -- so it stops on the tolerance rather than being
    # forced to a fixed pass count. Same fixed point; fewer passes spent
    # proving it.
    ctrl = controller()
    n_d, _, _ = aci.warm_start(ctrl, ens[keep], truth[keep], eval_times,
                               max_passes=40, tol=1e-4, verbose=False)
    runs[f"D  ORACLE 2021 cycled to its own equilibrium ({n_d} passes)"] = ctrl

    traces = {}
    for label, ctrl in runs.items():
        res = ctrl.run(ens[keep], truth[keep], init_times=eval_times)
        cov = coverage.covered(truth_eval, res.lower, res.upper)
        rates = monthly(cov, months, weights)
        traces[label] = rates
        _, line = verdict(rates)
        print(f"\n  {label}")
        print("    " + "  ".join(f"{m:02d}:{x:.4f}" for m, x in zip(range(1, 13), rates)))
        print(f"    marginal {coverage.marginal_coverage(cov, weights=weights)[0]:.4f}   {line}")

    reported = traces["C  warm start on 2020, fully converged"]
    oracle = next(v for k, v in traces.items() if k.startswith("D "))
    print("\n  cost of the honest start against the oracle, by month:")
    print("    " + "  ".join(f"{m:02d}:{d:+.4f}"
                             for m, d in zip(range(1, 13), reported - oracle)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
