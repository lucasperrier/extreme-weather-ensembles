#!/usr/bin/env python
"""Batch ArchesWeatherGen inference over the configured init grid.

ONE OUTPUT FILE PER INIT, written to a temp name and renamed on completion. A
rename is atomic within a filesystem, so a `.nc` file is either absent or
whole: the job is killable at any instant (Ctrl-C, spot preemption, OOM) and
restarting costs at most the one init that was in flight.

    source /workspace/env.sh
    tmux new -s gen -d 'cd $PROJECT_ROOT && python scripts/01_generate.py \
        2>&1 | tee -a $EVAL_ROOT/generate.log'

Inits are visited chronologically (2020 first) so the calibration year lands
before the verification year and downstream pipeline work can start on partial
output.

Storage: 20 members x 5 leads x 121 x 240 x float32 = 11.6 MB per init.

Every file carries a provenance attrs block -- checkpoints, geoarches version,
inference settings, seed scheme, ERA5 source. A regenerated archive with no
provenance is unreproducible six weeks later when the paper is under review.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from xconformal import config, generate as gen  # noqa: E402


def iter_init_dates(
    start: np.datetime64 = config.INIT_START,
    end: np.datetime64 = config.INIT_END,
    stride_calibration: int = config.INIT_STRIDE_CALIBRATION,
    stride_verification: int = config.INIT_STRIDE_VERIFICATION,
) -> list[np.datetime64]:
    """The init grid: daily 0Z, thinned per year by the configured strides.

    Chronological. The calibration year may be thinned to fit the wall-clock
    budget; the verification year never is.
    """
    day = np.timedelta64(1, "D")
    start, end = np.datetime64(start, "h"), np.datetime64(end, "h")
    total = int((end - start) / day) + 1
    dates = [start + i * day for i in range(total)]

    out: list[np.datetime64] = []
    for year, stride in (
        (config.CALIBRATION_YEAR, stride_calibration),
        (config.VERIFICATION_YEAR, stride_verification),
    ):
        in_year = [d for d in dates if d.astype("datetime64[Y]").astype(int) + 1970 == year]
        out.extend(in_year[::stride])
    return sorted(out)


def provenance(cfg, args) -> dict:
    """Everything needed to regenerate this file, as netcdf-safe attrs."""
    from importlib.metadata import version

    inf = cfg.module.inference
    ckpts = {}
    for name in [config.GEN_MODEL_NAME] + [
        Path(p).name for p in cfg.module.module.load_deterministic_model
    ]:
        ckpt = config.MODEL_ROOT / name / "checkpoints" / "checkpoint.ckpt"
        ckpts[name] = f"{ckpt.stat().st_size}B" if ckpt.is_file() else "MISSING"
    return {
        "title": "ArchesWeatherGen 2m-temperature ensemble",
        "generated_by": "xconformal scripts/01_generate.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "geoarches_version": version("geoarches"),
        "torch_version": version("torch"),
        "generative_model": config.GEN_MODEL_NAME,
        "deterministic_models": ", ".join(
            Path(p).name for p in cfg.module.module.load_deterministic_model
        ),
        "checkpoint_sizes": json.dumps(ckpts),
        "num_steps": int(inf.num_steps),
        "cf_guidance": float(inf.cf_guidance),
        "s_churn": float(inf.s_churn),
        "scale_input_noise": str(getattr(inf, "scale_input_noise", None)),
        "scheduler": str(cfg.module.module.scheduler),
        "seed_scheme": (
            "geoarches composes seed = member + 1000*rollout_step + batch_nb*1e6; "
            "batch_nb = days since 1970-01-01 of the init date, so a resumed run "
            "reproduces the same noise regardless of visit order"
        ),
        "sampling_batch_size": int(args.batch_size),
        "era5_source": str(config.ERA5_FULL_DIR),
        "lead_time_hours": int(config.LEAD_TIME_HOURS),
        "rollout_iterations": int(config.ROLLOUT_ITERATIONS),
        "n_members": int(args.members),
    }


def to_dataset(t2m: np.ndarray, init: np.datetime64, dataset, attrs: dict) -> xr.Dataset:
    """Wrap sampled t2m as a labelled dataset.

    in: t2m (n_members, n_leads, lat, lon) Kelvin, init, dataset, attrs;
    out: xr.Dataset with data var 't2m' and coords member/lead/latitude/longitude.
    """
    n_members, n_leads = t2m.shape[:2]
    leads = list(config.ARCHIVED_LEADS)[:n_leads]
    grid = dataset.xr_datasets[0] if hasattr(dataset, "xr_datasets") else None
    coords = {
        "member": np.arange(n_members, dtype="int16"),
        "lead": np.array(leads, dtype="int8"),
        "init_time": np.datetime64(init, "ns"),
        "valid_time": ("lead", np.array(
            [np.datetime64(init, "ns") + np.timedelta64(l, "D") for l in leads])),
    }
    if grid is not None:
        coords["latitude"] = grid["latitude"].values
        coords["longitude"] = grid["longitude"].values
    da = xr.DataArray(
        t2m.astype("float32"),
        dims=("member", "lead", "latitude", "longitude"),
        coords=coords,
        name="t2m",
        attrs={"long_name": "2m temperature", "units": "K",
               "standard_name": "air_temperature"},
    )
    ds = da.to_dataset()
    # Deliberately NOT units="days": a timedelta-like units attribute makes
    # xarray decode `lead` into timedelta64 on open, so a later ds.sel(lead=5)
    # raises instead of selecting day 5. Keep the unit in the long_name.
    ds["lead"].attrs.update({"long_name": "forecast lead time in days"})
    ds.attrs.update(attrs)
    return ds


def write_atomic(ds: xr.Dataset, path: Path) -> None:
    """Write via a temp file in the same directory, then rename.

    The temp name carries the pid so concurrent workers cannot collide, and
    stays in the destination directory so the rename is within one filesystem
    (a cross-filesystem rename is a copy, and is not atomic).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    encoding = {"t2m": {"zlib": True, "complevel": 1, "dtype": "float32"}}
    try:
        ds.to_netcdf(tmp, encoding=encoding)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)  # KeyboardInterrupt included: leave no debris
        raise


def clean_stale_temps(directory: Path) -> int:
    stale = list(directory.glob("*.tmp-*")) if directory.is_dir() else []
    for path in stale:
        path.unlink(missing_ok=True)
    return len(stale)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", type=np.datetime64, default=config.INIT_START)
    parser.add_argument("--end", type=np.datetime64, default=config.INIT_END)
    parser.add_argument("--members", type=int, default=config.N_MEMBERS)
    parser.add_argument("--batch-size", type=int, default=config.SAMPLING_BATCH_SIZE,
                        help="members per denoising pass; see 02_benchmark.py")
    parser.add_argument("--stride-calibration", type=int,
                        default=config.INIT_STRIDE_CALIBRATION)
    parser.add_argument("--stride-verification", type=int,
                        default=config.INIT_STRIDE_VERIFICATION)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many inits (smoke test)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be generated and exit; loads no model")
    parser.add_argument("--clean-temps", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.validate_paths(require_model=not args.dry_run)

    if args.clean_temps:
        print(f"removed {clean_stale_temps(config.ENSEMBLE_DIR)} stale temp files")

    inits = iter_init_dates(args.start, args.end,
                            args.stride_calibration, args.stride_verification)
    done = [d for d in inits if config.forecast_file(d).exists()]
    todo = [d for d in inits if not config.forecast_file(d).exists()]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"grid     : {args.start} .. {args.end}, "
          f"{config.CALIBRATION_YEAR} every {args.stride_calibration}d, "
          f"{config.VERIFICATION_YEAR} every {args.stride_verification}d")
    print(f"inits    : {len(inits)} total, {len(done)} done, {len(todo)} to do")
    print(f"members  : {args.members} at batch size {args.batch_size}")
    print(f"output   : {config.ENSEMBLE_DIR}")
    if args.dry_run:
        for date in todo[:10]:
            print(f"  would write {config.forecast_file(date)}")
        if len(todo) > 10:
            print(f"  ... and {len(todo) - 10} more")
        return 0
    if not todo:
        print("nothing to do")
        return 0

    module, cfg = gen.load_model(args.device)
    dataset = gen.build_dataset()
    attrs = provenance(cfg, args)
    members = list(range(args.members))

    started = time.time()
    for i, init in enumerate(todo, start=1):
        out_path = config.forecast_file(init)
        if out_path.exists():
            continue  # another worker got here first
        batch, dataset = gen.load_batch(init, dataset=dataset, device=args.device)
        t2m = gen.sample_members(module, batch, dataset, init, members,
                                 batch_size=args.batch_size)
        write_atomic(to_dataset(t2m, init, dataset, attrs), out_path)
        elapsed = time.time() - started
        rate = elapsed / i
        print(f"[{i}/{len(todo)}] {out_path.name}  "
              f"{rate:.1f}s/init  {t2m.min():.1f}-{t2m.max():.1f}K  "
              f"eta {(len(todo) - i) * rate / 3600:.1f}h", flush=True)
    print(f"done: {len(todo)} inits in {(time.time() - started) / 3600:.2f}h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
