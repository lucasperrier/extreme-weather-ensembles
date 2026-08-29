#!/usr/bin/env python
"""Full-archive audit, run once before the paper's numbers are computed.

07_verify_outputs.py checks a handful of files while a run is live. This checks
EVERY file, plus the things that only mean something once the run is finished:
the init dates on disk against the schedule that was locked in config, and the
grid convention on a random sample spanning both years.

    source /workspace/env.sh
    python scripts/08_audit_archive.py                # phases 1-3 + du
    python scripts/08_audit_archive.py --repro-date 2021-06-15   # + regenerate one init

Exit status is 0 only if every phase passes; a non-zero status means the
downstream verification must not be run.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from xconformal import config, grid  # noqa: E402

T2M_MIN_K, T2M_MAX_K = 190.0, 330.0


def phase_schedule() -> tuple[bool, dict]:
    """Init dates on disk vs the locked schedule."""
    from importlib import import_module
    gen01 = import_module("01_generate")
    wanted = gen01.iter_init_dates()

    files = sorted(config.ENSEMBLE_DIR.glob("*.nc"))
    stems = [f.stem for f in files]
    on_disk = []
    unparsable = []
    for stem in stems:
        try:
            on_disk.append(np.datetime64(
                f"{stem[0:4]}-{stem[4:6]}-{stem[6:8]}T{stem[9:11]}", "h"))
        except Exception:
            unparsable.append(stem)

    duplicates = sorted({str(d) for d in on_disk if list(on_disk).count(d) > 1})
    wanted_set = {np.datetime64(d, "h") for d in wanted}
    disk_set = set(on_disk)
    missing = sorted(str(d) for d in wanted_set - disk_set)
    extra = sorted(str(d) for d in disk_set - wanted_set)

    y2020 = [d for d in on_disk if str(d).startswith("2020")]
    y2021 = [d for d in on_disk if str(d).startswith("2021")]

    # Cadence: 2020 every 4th day from 2020-01-02, 2021 daily to 2021-12-26.
    day = np.timedelta64(1, "D")
    gaps2020 = sorted({int(g / day) for g in np.diff(sorted(y2020))}) if len(y2020) > 1 else []
    gaps2021 = sorted({int(g / day) for g in np.diff(sorted(y2021))}) if len(y2021) > 1 else []

    ok = (len(files) == 452 and not missing and not extra and not duplicates
          and not unparsable and len(y2020) == 92 and len(y2021) == 360
          and gaps2020 == [4] and gaps2021 == [1])

    info = dict(n_files=len(files), n_2020=len(y2020), n_2021=len(y2021),
                first=str(min(on_disk)) if on_disk else None,
                last=str(max(on_disk)) if on_disk else None,
                missing=missing, extra=extra, duplicates=duplicates,
                unparsable=unparsable, gaps2020=gaps2020, gaps2021=gaps2021,
                n_scheduled=len(wanted))
    return ok, info


def phase_integrity(files: list[Path]) -> tuple[bool, dict]:
    """Every file opens, loads, has 20 members and lead 5, is finite and physical."""
    failures: list[tuple[str, str]] = []
    lo_all, hi_all = np.inf, -np.inf
    spread_lo, spread_hi = np.inf, -np.inf
    for n, path in enumerate(files, start=1):
        try:
            with xr.open_dataset(path, decode_timedelta=False) as ds:
                ds = ds.load()
                problems = []
                if "t2m" not in ds.data_vars:
                    failures.append((path.name, f"no t2m (has {list(ds.data_vars)})"))
                    continue
                da = ds["t2m"]
                if da.sizes.get("member") != config.N_MEMBERS:
                    problems.append(f"M={da.sizes.get('member')} != {config.N_MEMBERS}")
                leads = np.asarray(da["lead"].values).tolist()
                if config.LEAD_DAYS not in leads:
                    problems.append(f"lead {config.LEAD_DAYS} absent (has {leads})")
                v = da.values
                n_bad = int((~np.isfinite(v)).sum())
                if n_bad:
                    problems.append(f"{n_bad} non-finite values")
                lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
                lo_all, hi_all = min(lo_all, lo), max(hi_all, hi)
                if not (T2M_MIN_K <= lo and hi <= T2M_MAX_K):
                    problems.append(f"range [{lo:.1f}, {hi:.1f}] K outside "
                                    f"[{T2M_MIN_K}, {T2M_MAX_K}]")
                s = float(v[:, -1].std(axis=0).mean())
                spread_lo, spread_hi = min(spread_lo, s), max(spread_hi, s)
                if s <= 1e-6:
                    problems.append(f"member spread {s:.2e} K -- members identical")
                if problems:
                    failures.append((path.name, "; ".join(problems)))
        except Exception as exc:  # noqa: BLE001
            failures.append((path.name, f"{type(exc).__name__}: {exc}"))
        if n % 50 == 0:
            print(f"    ... {n}/{len(files)} checked", flush=True)
    info = dict(n=len(files), n_pass=len(files) - len(failures), failures=failures,
                min_K=lo_all, max_K=hi_all, spread_min=spread_lo, spread_max=spread_hi)
    return not failures, info


def _load_truth_one(valid_time: np.datetime64) -> np.ndarray:
    year = int(str(valid_time)[:4])
    path = config.era5_file(year, config.INIT_HOUR)
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        if ds.sizes.get("time", 0) == 0:
            raise ValueError(f"{path} has an empty time dimension")
        return ds[config.TARGET_VAR].sel(
            time=np.datetime64(valid_time, "ns")).transpose(
                "latitude", "longitude").values.astype("float32")


def phase_alignment(files: list[Path], n_sample: int, seed: int) -> tuple[bool, dict]:
    """grid.check_alignment on a random sample spanning both years."""
    rng = np.random.default_rng(seed)
    y2020 = [f for f in files if f.name.startswith("2020")]
    y2021 = [f for f in files if f.name.startswith("2021")]
    half = n_sample // 2
    picks = sorted(
        list(rng.choice(y2020, size=min(half, len(y2020)), replace=False))
        + list(rng.choice(y2021, size=min(n_sample - half, len(y2021)), replace=False)),
        key=lambda p: p.name,
    )
    rows, failures = [], []
    for path in picks:
        stem = path.stem
        init = np.datetime64(f"{stem[0:4]}-{stem[4:6]}-{stem[6:8]}T{stem[9:11]}", "h")
        with xr.open_dataset(path, decode_timedelta=False) as ds:
            f1 = grid.to_era5_grid(
                ds["t2m"].sel(lead=1).values.astype("float32")).mean(axis=0)
        truth = _load_truth_one(init + np.timedelta64(1, "D"))
        try:
            rmse = grid.check_alignment(f1, truth, label=f"init {init} lead 1")
        except ValueError as exc:
            failures.append((path.name, str(exc)))
            rmse = float("nan")
        rows.append((path.name, rmse))
    finite = [r for _, r in rows if np.isfinite(r)]
    info = dict(rows=rows, failures=failures,
                max_rmse=max(finite) if finite else float("nan"),
                min_rmse=min(finite) if finite else float("nan"),
                tolerance=grid.ALIGNMENT_TOLERANCE_K)
    return not failures, info


def phase_repro(date: str, device: str) -> tuple[bool, dict]:
    """Regenerate one init and compare it bit-for-bit with the archived file."""
    from xconformal import generate as gen
    gen01 = __import__("01_generate")

    init = np.datetime64(f"{date}T00", "h")
    archived = config.forecast_file(init)
    if not archived.is_file():
        return False, dict(error=f"no archived file for {init}: {archived}")

    module, cfg = gen.load_model(device)
    dataset = gen.build_dataset()
    batch, dataset = gen.load_batch(init, dataset=dataset, device=device)
    t2m = gen.sample_members(module, batch, dataset, init,
                             list(range(config.N_MEMBERS)),
                             batch_size=config.SAMPLING_BATCH_SIZE)
    with xr.open_dataset(archived, decode_timedelta=False) as ds:
        stored = ds["t2m"].values
    same_shape = stored.shape == t2m.shape
    identical = bool(same_shape and np.array_equal(stored, t2m.astype("float32")))
    diff = float(np.abs(stored - t2m.astype("float32")).max()) if same_shape else float("nan")
    info = dict(init=str(init), file=archived.name, shape_new=t2m.shape,
                shape_stored=stored.shape, identical=identical, max_abs_diff=diff,
                sha_stored=hashlib.sha256(stored.tobytes()).hexdigest()[:16],
                sha_new=hashlib.sha256(t2m.astype("float32").tobytes()).hexdigest()[:16])
    return identical, info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sample", type=int, default=20,
                        help="files to run the grid alignment guard on")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--repro-date", default=None,
                        help="YYYY-MM-DD init to regenerate and compare (needs the GPU)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-integrity", action="store_true")
    args = parser.parse_args(argv)
    config.validate_paths()

    files = sorted(config.ENSEMBLE_DIR.glob("*.nc"))
    all_ok = True

    print("=" * 72)
    print("PHASE 0.1  schedule")
    ok, info = phase_schedule()
    all_ok &= ok
    print(f"  files on disk        : {info['n_files']}  (schedule says {info['n_scheduled']})")
    print(f"  2020 / 2021          : {info['n_2020']} / {info['n_2021']}  (expect 92 / 360)")
    print(f"  first / last         : {info['first']} .. {info['last']}")
    print(f"  2020 gaps (days)     : {info['gaps2020']}   2021 gaps (days): {info['gaps2021']}")
    print(f"  missing              : {len(info['missing'])} {info['missing'][:10]}")
    print(f"  unscheduled extras   : {len(info['extra'])} {info['extra'][:10]}")
    print(f"  duplicates           : {len(info['duplicates'])} {info['duplicates'][:10]}")
    print(f"  unparsable names     : {info['unparsable']}")
    print(f"  -> {'PASS' if ok else 'FAIL'}")

    if not args.skip_integrity:
        print("=" * 72)
        print(f"PHASE 0.2  per-file integrity ({len(files)} files)")
        ok, info = phase_integrity(files)
        all_ok &= ok
        print(f"  passed               : {info['n_pass']}/{info['n']}")
        print(f"  t2m range over all   : {info['min_K']:.1f} .. {info['max_K']:.1f} K")
        print(f"  day-5 member spread  : {info['spread_min']:.2f} .. {info['spread_max']:.2f} K")
        for name, why in info["failures"]:
            print(f"  !! {name}: {why}")
        print(f"  -> {'PASS' if ok else 'FAIL'}")

    print("=" * 72)
    print(f"PHASE 0.3  grid alignment on {args.sample} random files")
    ok, info = phase_alignment(files, args.sample, args.seed)
    all_ok &= ok
    for name, rmse in info["rows"]:
        print(f"  {name:<18s} day-1 ens-mean RMSE {rmse:7.3f} K")
    print(f"  min / max RMSE       : {info['min_rmse']:.3f} / {info['max_rmse']:.3f} K "
          f"(guard raises above {info['tolerance']} K)")
    print(f"  -> {'PASS' if ok else 'FAIL'}")

    if args.repro_date:
        print("=" * 72)
        print(f"PHASE 0.4  reproducibility spot-check, {args.repro_date}")
        ok, info = phase_repro(args.repro_date, args.device)
        all_ok &= ok
        for k, v in info.items():
            print(f"  {k:<20s} : {v}")
        print(f"  -> {'PASS' if ok else 'FAIL'}")

    print("=" * 72)
    print(f"AUDIT {'PASSED' if all_ok else 'FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
