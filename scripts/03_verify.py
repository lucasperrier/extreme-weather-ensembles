#!/usr/bin/env python
"""Run ACI over the forecast archive and write the coverage tables.

This is the experiment. It loads the archive, bins verification days by the
ensemble's own exceedance probability p_t, runs both the raw ensemble interval
and marginal ACI, and reports marginal and per-bin coverage for each.

    source /workspace/env.sh
    python scripts/03_verify.py --space variable
    python scripts/03_verify.py --space quantile

Outputs (paths from config):
    coverage_by_bin.csv    tidy: method, bin_label, coverage, count, target
    coverage_marginal.csv  one row per method
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from xconformal import aci, binning, config, coverage, thresholds  # noqa: E402


def load_archive(start: np.datetime64, end: np.datetime64) -> xr.Dataset:
    """Open every per-init forecast file in the window as one dataset.

    in: start, end init dates; out: xr.Dataset with dims
    (init_time, member, lat, lon), sorted ascending by init_time.

    Skip -- loudly -- any init whose file is absent, and any file whose
    init_time dimension is empty. An empty file that gets concatenated
    disappears without trace and silently shortens the stream, which would
    corrupt the ACI state sequence rather than just lose a day.

    TODO(lucas): [Fri] ACI is a sequential controller: a gap in the init
    sequence is not the same as a shorter sequence. Decide what a missing init
    means -- skip the step entirely (c does not move) or carry the delay
    forward -- and make load_archive report gaps so 03_verify can act on them.
    Record the choice in RUNBOOK.md under Decisions.
    """
    raise NotImplementedError(
        "in: start, end datetime64; out: xr.Dataset (init_time, member, lat, lon), gaps reported"
    )


def load_truth(valid_times: np.ndarray) -> xr.DataArray:
    """Load ERA5 truth for the given valid times.

    in: valid_times (n_time,) datetime64; out: DataArray (time, lat, lon) of
    the resolved target variable, aligned to valid_times in order.

    TODO(lucas): [Fri] The 2020 ERA5 download may still be in flight and 2022
    files may exist with time=0. Open files individually and drop any with an
    empty time dimension BEFORE concatenating -- see scripts/00_inspect_data.py,
    which flags exactly these.
    """
    raise NotImplementedError(
        "in: valid_times (n_time,); out: DataArray (time, lat, lon) aligned to valid_times"
    )


def evaluation_mask(init_times: np.ndarray) -> np.ndarray:
    """Which inits count towards the reported coverage.

    Burn-in inits (before config.EVAL_START) are still fed to the ACI update --
    that is the point of a burn-in -- but are excluded from every reported
    number. Returns a bool array over init_times.
    """
    return np.asarray(init_times) >= config.EVAL_START


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--space", default=config.ACI_SPACE, choices=("variable", "quantile"),
                        help=f"ACI adaptation space (default {config.ACI_SPACE})")
    parser.add_argument("--start", type=np.datetime64, default=config.INIT_START)
    parser.add_argument("--end", type=np.datetime64, default=config.INIT_END)
    parser.add_argument("--eta", type=float, default=config.ETA)
    parser.add_argument("--alpha", type=float, default=config.ALPHA)
    parser.add_argument("--tau", type=int, default=config.TAU)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.validate_paths()

    # ---- inputs -----------------------------------------------------------
    archive = load_archive(args.start, args.end)
    init_times = archive.init_time.values
    valid_times = init_times + np.timedelta64(config.LEAD_DAYS, "D")
    truth = load_truth(valid_times)

    thr = thresholds.load_thresholds()
    thr_t = thresholds.threshold_for_times(thr, valid_times)

    var = config.TARGET_VAR_SHORT
    ensembles = archive[var].transpose("init_time", "member", ...).values  # (T, M, lat, lon)
    y = truth.transpose("time", ...).values                                # (T, lat, lon)
    grid_shape = y.shape[1:]

    # ---- p_t and bins -----------------------------------------------------
    p = binning.exceedance_probability(ensembles, thr_t.values)  # (T, lat, lon)

    # ---- methods ----------------------------------------------------------
    raw_lo, raw_hi = aci.raw_interval(ensembles)
    controller = aci.DelayedACI(
        grid_shape=grid_shape,
        alpha=args.alpha,
        eta=args.eta,
        tau=args.tau,
        adapter=aci.make_adapter(args.space),
    )
    result = controller.run(ensembles, y)

    covered_by_method = {
        "raw": coverage.covered(y, raw_lo, raw_hi),
        f"aci-{args.space}": coverage.covered(y, result.lower, result.upper),
    }

    # ---- restrict to the evaluation window --------------------------------
    keep = evaluation_mask(init_times)
    print(f"inits: {keep.size} total, {int(keep.sum())} evaluated "
          f"({int((~keep).sum())} burn-in, used for updates only)")
    p_eval = p[keep]
    masks = binning.bin_masks(p_eval)
    labels = config.p_bin_labels()
    covered_by_method = {k: v[keep] for k, v in covered_by_method.items()}

    # ---- report -----------------------------------------------------------
    import pandas as pd

    marginal_rows = []
    for method, is_covered in covered_by_method.items():
        rate, count = coverage.marginal_coverage(is_covered)
        marginal_rows.append({"method": method, "coverage": rate, "count": count,
                              "target": config.TARGET_COVERAGE})
        print(f"marginal  {method:<16s} {rate:.4f}  (n={count})")

    table = coverage.coverage_table(covered_by_method, masks, labels)
    print()
    print(table.to_string(index=False))

    config.EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(marginal_rows).to_csv(config.MARGINAL_TABLE_PATH, index=False)
    table.to_csv(config.COVERAGE_TABLE_PATH, index=False)
    print(f"\nwrote {config.MARGINAL_TABLE_PATH}\nwrote {config.COVERAGE_TABLE_PATH}")
    if result.saturated_frac is not None:
        print(f"adapter saturated on {result.saturated_frac:.2%} of (init, gridpoint) pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
