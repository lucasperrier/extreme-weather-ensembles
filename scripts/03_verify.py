#!/usr/bin/env python
"""Run ACI over the forecast archive and write the coverage tables.

This is the experiment. It loads whatever inits exist, bins verification days by
the ensemble's own exceedance probability p_t, runs the raw ensemble interval
and standardized variable-space ACI, and reports marginal and per-bin coverage.

    source /workspace/env.sh
    python scripts/03_verify.py                      # the paper's configuration
    python scripts/03_verify.py --space quantile     # appendix ablation
    python scripts/03_verify.py --eval-start 2020-01-02   # inspect calibration

Gaps need no handling. The controller keys its delay queue on verification
DATETIME, so a missing init is simply a longer wait before queued updates come
due -- exactly what physically happened. Missing dates are skipped and counted;
nothing is interpolated, and the 2020->2021 cadence change is not a special
case anywhere.

Outputs (paths from config, all under $EVAL_ROOT):
    coverage_by_bin.csv       per method, per p_t bin: coverage + raw counts
    coverage_marginal.csv     per method: marginal coverage
    marginal_timeseries.csv   per init: marginal coverage of each method
    c_trajectories.csv        c_t at six named gridpoints
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from xconformal import aci, binning, config, coverage, grid, thresholds  # noqa: E402

# Six gridpoints whose c_t trajectories are tracked, chosen to span the
# regimes where the standardization scale differs most: the tropics have a
# small day-to-day spread, mid-latitude land a large one, and deserts a large
# diurnal-driven one. Nearest gridpoint is used.
PROBE_POINTS = {
    "tropics_ocean": (0.0, 200.0),    # central Pacific
    "tropics_land": (0.0, 25.0),      # Congo basin
    "midlat_ocean": (45.0, 330.0),    # North Atlantic
    "midlat_land": (45.0, 265.0),     # US Great Plains
    "high_lat": (75.0, 90.0),         # Siberian Arctic
    "desert": (25.0, 15.0),           # central Sahara
}


def archive_inits(start: np.datetime64, end: np.datetime64) -> tuple[list, list]:
    """Init dates present on disk in the window, and the ones missing from the grid.

    out: (present, missing), both ascending lists of datetime64[h].
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from importlib import import_module

    wanted = import_module("01_generate").iter_init_dates(start, end)
    present = [d for d in wanted if config.forecast_file(d).exists()]
    missing = [d for d in wanted if not config.forecast_file(d).exists()]
    return present, missing


def load_archive(inits: list, lead: int = config.LEAD_DAYS) -> np.ndarray:
    """Load per-member t2m at one lead for each init, on the ERA5 grid.

    in: ascending init dates, lead in days;
    out: ensembles (n_time, n_members, lat, lon) float32.

    The archive is stored on the MODEL grid (latitude descending, longitude
    offset by 180 degrees) and is reoriented here -- see xconformal.grid. The
    caller must run grid.check_alignment on the result; skipping the transform
    does not raise, it just produces believable nonsense.

    Files are opened individually rather than via open_mfdataset: an empty or
    truncated file vanishes silently in a concat, which would shift every
    downstream index rather than raise.
    """
    fields = []
    for init in inits:
        with xr.open_dataset(config.forecast_file(init), decode_timedelta=False) as ds:
            if ds.sizes.get("member", 0) == 0:
                raise ValueError(f"{config.forecast_file(init)} has no members")
            fields.append(grid.to_era5_grid(ds["t2m"].sel(lead=lead).values.astype("float32")))
    return np.stack(fields)


def load_truth(valid_times: np.ndarray) -> np.ndarray:
    """ERA5 2m temperature at the given valid times.

    in: valid_times (n_time,) datetime64; out: (n_time, lat, lon) float32,
    ordered to match valid_times.

    Files with an empty time dimension are dropped before concatenation -- an
    empty time dim does not raise in xr.concat, it silently shortens the record.
    """
    years = sorted({int(str(t)[:4]) for t in valid_times.astype("datetime64[Y]")})
    slices = []
    for year in years:
        path = config.era5_file(year, config.INIT_HOUR)
        if not path.is_file():
            raise FileNotFoundError(f"ERA5 truth missing for {year}: {path}")
        ds = xr.open_dataset(path, decode_timedelta=False)
        if ds.sizes.get("time", 0) == 0:
            ds.close()
            raise ValueError(f"{path} has an empty time dimension -- treat as missing data")
        slices.append(ds[config.TARGET_VAR])
    truth = xr.concat(slices, dim="time").sortby("time")
    missing = [t for t in valid_times if t not in truth.time.values]
    if missing:
        raise KeyError(f"{len(missing)} valid times have no ERA5 truth, first {missing[0]}")
    out = truth.sel(time=valid_times).transpose("time", "latitude", "longitude")
    return out.values.astype("float32")


def probe_indices(latitudes: np.ndarray, longitudes: np.ndarray) -> dict:
    """Nearest grid index for each named probe point."""
    out = {}
    for name, (lat, lon) in PROBE_POINTS.items():
        i = int(np.abs(latitudes - lat).argmin())
        j = int(np.abs(longitudes - (lon % 360)).argmin())
        out[name] = (i, j, float(latitudes[i]), float(longitudes[j]))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--space", default=config.ACI_SPACE, choices=("variable", "quantile"))
    parser.add_argument("--start", type=np.datetime64, default=config.INIT_START)
    parser.add_argument("--end", type=np.datetime64, default=config.INIT_END)
    parser.add_argument("--eval-start", type=np.datetime64, default=config.EVAL_START,
                        help="first init counted in the reported numbers")
    parser.add_argument("--eta", type=float, default=config.ETA)
    parser.add_argument("--alpha", type=float, default=config.ALPHA)
    parser.add_argument("--tau-days", type=int, default=config.TAU)
    parser.add_argument("--no-standardize", action="store_true",
                        help="disable the per-gridpoint ACI scale (diagnostic only)")
    parser.add_argument("--no-warm-start", dest="warm_start", action="store_false",
                        help="single sequential calibration pass instead of cycling")
    parser.add_argument("--warm-tol", type=float, default=0.01,
                        help="stop cycling when pass-mean c changes by less than this")
    parser.add_argument("--suffix", default="", help="appended to output filenames")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.validate_paths()
    import pandas as pd

    # ---- archive ---------------------------------------------------------
    present, missing = archive_inits(args.start, args.end)
    if not present:
        print(f"no ensemble files under {config.ENSEMBLE_DIR}")
        return 1
    print(f"archive : {len(present)} inits present, {len(missing)} of the grid missing")
    print(f"          {present[0]} .. {present[-1]}")

    init_times = np.array(present, dtype="datetime64[h]")
    valid_times = init_times + np.timedelta64(config.LEAD_DAYS, "D")
    ensembles = load_archive(present)
    truth = load_truth(valid_times)
    latitudes = grid.era5_latitudes()
    longitudes = grid.era5_longitudes()
    print(f"          ensembles {ensembles.shape}, truth {truth.shape}")

    # Prove the reorientation actually worked before any science runs on it.
    # A day-1 forecast is the sharpest available check: sub-Kelvin when aligned,
    # ~12.7 K when not, and nothing else about the arrays looks different.
    lead1 = load_archive(present[:1], lead=1)[0].mean(axis=0)
    truth1 = load_truth(np.array([present[0] + np.timedelta64(1, "D")]))[0]
    rmse = grid.check_alignment(lead1, truth1, label=f"init {present[0]} lead 1")
    print(f"          grid alignment ok: day-1 ensemble-mean RMSE {rmse:.2f} K")

    # ---- p_t and bins ----------------------------------------------------
    thr = thresholds.load_thresholds()
    thr_t = thresholds.threshold_for_times(thr, valid_times).values
    p = binning.exceedance_probability(ensembles, thr_t)

    # ---- controller ------------------------------------------------------
    if args.space == "variable":
        scale = None if args.no_standardize else thresholds.load_aci_scale().values
        adapter = aci.VariableSpaceAdapter(scale=scale)
    else:
        adapter = aci.QuantileSpaceAdapter(alpha=args.alpha)
    method = f"aci-{args.space}" + ("" if not args.no_standardize else "-unscaled")

    controller = aci.DelayedACI(
        grid_shape=truth.shape[1:], alpha=args.alpha, eta=args.eta,
        tau=np.timedelta64(args.tau_days, "D"), adapter=adapter,
    )

    # ---- evaluation window ----------------------------------------------
    keep = init_times >= np.datetime64(args.eval_start, "h")
    n_eval = int(keep.sum())
    print(f"eval    : {n_eval} of {keep.size} inits at or after {args.eval_start} "
          f"({keep.size - n_eval} calibration)")
    if n_eval == 0:
        print("!! no inits in the evaluation window -- nothing to report yet")
        return 2

    # ---- warm start on the calibration year only ------------------------
    # ACI's time constant here is ~80 inits, so a single 92-init calibration
    # pass leaves the controller still climbing when evaluation opens. Cycling
    # the calibration year removes that transient without touching eta and
    # without the evaluation year ever informing c.
    n_passes, pass_means = 0, []
    calibration = ~keep
    if args.warm_start and calibration.any():
        print(f"\nwarm start on {int(calibration.sum())} calibration inits "
              f"({init_times[calibration][0]} .. {init_times[calibration][-1]})")
        n_passes, pass_means, warm_result = aci.warm_start(
            controller, ensembles[calibration], truth[calibration],
            init_times[calibration], tol=args.warm_tol,
        )
        print(f"  converged after {n_passes} passes, mean c {pass_means[-1]:+.5f}, "
              f"queue empty: {controller.pending == 0}")
    elif calibration.any():
        # Single sequential pass, the un-warm-started behaviour.
        controller.run(ensembles[calibration], truth[calibration],
                       init_times[calibration])

    # The evaluation run starts from the settled c with an empty queue.
    result = controller.run(ensembles[keep], truth[keep], init_times=init_times[keep])
    raw_lo, raw_hi = aci.raw_interval(ensembles[keep])
    truth_eval = truth[keep]

    covered = {
        "raw": coverage.covered(truth_eval, raw_lo, raw_hi),
        method: coverage.covered(truth_eval, result.lower, result.upper),
    }

    weights = coverage.latitude_weights(latitudes)[None, :, None]
    eval_times = init_times[keep]

    # ---- (1) marginal ----------------------------------------------------
    print("\n--- (1) marginal coverage, area-weighted ---")
    marginal_rows = []
    for name, is_covered in covered.items():
        rate, count = coverage.marginal_coverage(is_covered, weights=weights)
        unweighted, _ = coverage.marginal_coverage(is_covered)
        marginal_rows.append({"method": name, "coverage": rate,
                              "coverage_unweighted": unweighted, "count": count,
                              "target": config.TARGET_COVERAGE,
                              "warm_start_passes": n_passes})
        print(f"  {name:<18s} {rate:.4f}  (unweighted {unweighted:.4f}, n={count:,})")

    # ---- (2) c_t transient ----------------------------------------------
    print("\n--- (2) c_t at the evaluation boundary ---")
    probes = probe_indices(latitudes, longitudes)
    c_rows = []
    for t, init in enumerate(eval_times):
        for name, (i, j, lat, lon) in probes.items():
            c_rows.append({"init_time": init, "probe": name, "latitude": lat,
                           "longitude": lon, "c": float(result.c_history[t, i, j])})
    c_frame = pd.DataFrame(c_rows)

    boundary = 0  # the evaluation run now starts at the boundary by construction
    window = min(30, max(1, len(eval_times) - 1))
    print(f"  {'probe':<14s} {'lat':>6s} {'lon':>6s} {'c@bound':>9s} {'c@end':>9s} "
          f"{'drift/init':>11s}")
    for name, (i, j, lat, lon) in probes.items():
        series = result.c_history[:, i, j]
        # Drift is the mean per-init change over the run-up to the boundary. If
        # the transient is dead this is ~0; if c is still climbing it is not,
        # and the calibration year was too short.
        drift = float((series[window] - series[0]) / window)
        print(f"  {name:<14s} {lat:>+6.1f} {lon:>6.1f} {series[boundary]:>+9.4f} "
              f"{series[-1]:>+9.4f} {drift:>+11.5f}")
    print(f"  all gridpoints @boundary: c in "
          f"[{result.c_history[boundary].min():+.4f}, {result.c_history[boundary].max():+.4f}], "
          f"mean {result.c_history[boundary].mean():+.4f}")
    print(f"  all gridpoints @end     : c in "
          f"[{result.c_history[-1].min():+.4f}, {result.c_history[-1].max():+.4f}], "
          f"mean {result.c_history[-1].mean():+.4f}")

    # How fast is c actually moving, and how far does it still have to go? With
    # only a short calibration run this is the difference between "converged"
    # and "barely started", and the marginal number alone does not show it.
    per_update = float(np.mean(np.diff(result.c_history[:, ...].mean(axis=(1, 2)))))
    print(f"  mean c increment per init: {per_update:+.5f}")

    # ---- (2b) is the calibration year long enough? -----------------------
    # The marginal number alone cannot distinguish "converged" from "still
    # climbing". c* is the CONSTANT padding that would hit the target on this
    # data -- a static calculation, no dynamics -- and comparing it to what the
    # controller can actually reach in the warm-up says whether the transient
    # is dead by the evaluation boundary.
    print("\n--- (2b) warm-up sufficiency ---")
    scale_field = getattr(adapter, "scale", None)
    if scale_field is None:
        scale_field = 1.0
    def _coverage_at(c_const):
        padded = coverage.covered(truth_eval, raw_lo - c_const * scale_field,
                                  raw_hi + c_const * scale_field)
        return coverage.marginal_coverage(padded, weights=weights)[0]

    low, high = 0.0, 5.0
    for _ in range(50):
        mid = (low + high) / 2.0
        if _coverage_at(mid) < config.TARGET_COVERAGE:
            low = mid
        else:
            high = mid
    c_star = (low + high) / 2.0
    eps = 0.02
    slope = (_coverage_at(c_star / 2 + eps) - _coverage_at(c_star / 2 - eps)) / (2 * eps)
    tau_inits = 1.0 / (args.eta * slope) if slope > 0 else float("inf")
    n_calibration = int(calibration.sum())
    reached = 1.0 - np.exp(-n_calibration / tau_inits) if n_calibration else 0.0
    print(f"  equilibrium c*          : {c_star:.4f} "
          f"(median padding {float(np.median(c_star * scale_field)):.2f} K)")
    print(f"  dcoverage/dc            : {slope:.3f} per unit c")
    print(f"  ACI time constant       : {tau_inits:.0f} inits at eta={args.eta}")
    print(f"  calibration inits        : {n_calibration}")
    if args.warm_start and n_passes:
        print(f"  warm start cycled the calibration year {n_passes} times, so the "
              f"controller opens the evaluation window at c = {pass_means[-1]:+.4f} "
              f"rather than {(1 - np.exp(-n_calibration / tau_inits)) * c_star:+.4f}")
        print(f"  NOTE c* = {c_star:.4f} is estimated on the EVALUATION data and is "
              f"seasonal; expect c_t to oscillate over a full year.")
    elif n_calibration:
        print(f"  c reaches {(1 - np.exp(-n_calibration / tau_inits)) * 100:.1f}% of c* "
              f"in a single calibration pass -> coverage ~"
              f"{_coverage_at(c_star * (1 - np.exp(-n_calibration / tau_inits))):.4f}")
        print("  !! the transient is NOT dead at the evaluation boundary")

    # ---- (3) per-bin -----------------------------------------------------
    print("\n--- (3) per-bin coverage, area-weighted ---")
    masks = binning.bin_masks(p[keep])
    labels = config.p_bin_labels()
    table = coverage.coverage_table(covered, masks, labels, weights=weights)
    print(table.to_string(index=False))

    # ---- per-init marginal time series ----------------------------------
    ts_rows = []
    for t, init in enumerate(eval_times):
        row = {"init_time": init,
               "pending_updates": int(result.pending_depth[t]),
               "updates_applied": int(result.updates_applied[t])}
        for name, is_covered in covered.items():
            row[name], _ = coverage.marginal_coverage(is_covered[t], weights=weights[0])
        ts_rows.append(row)

    # ---- write -----------------------------------------------------------
    sfx = args.suffix
    config.EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    out = {
        config.MARGINAL_TABLE_PATH.with_stem(config.MARGINAL_TABLE_PATH.stem + sfx):
            pd.DataFrame(marginal_rows),
        config.COVERAGE_TABLE_PATH.with_stem(config.COVERAGE_TABLE_PATH.stem + sfx): table,
        config.EVAL_ROOT / f"marginal_timeseries{sfx}.csv": pd.DataFrame(ts_rows),
        config.EVAL_ROOT / f"c_trajectories{sfx}.csv": c_frame,
    }
    for path, frame in out.items():
        frame.to_csv(path, index=False)
        print(f"wrote {path}")

    if result.saturated_frac is not None:
        print(f"adapter saturated on {result.saturated_frac:.2%} of (init, gridpoint) pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
