#!/usr/bin/env python
"""Batch ArchesWeatherGen inference over daily 0Z inits.

ONE OUTPUT FILE PER INIT DATE, written to a temp name and renamed on
completion. That makes the job idempotent and killable at any point with zero
corruption: a rename is atomic on the same filesystem, so a file either does
not exist or is complete. This mirrors geoarches' own dl_era restart pattern
(one file per (year, hour), skip if present).

    source /workspace/env.sh
    python scripts/01_generate.py --start 2020-01-01 --end 2020-01-31 --members 20
    python scripts/01_generate.py            # whole configured window, resumes

Storage. We keep only what the verification needs: every member's target
variable at the single lead of interest, full grid.
    20 members x 121 x 240 x float32   = 2.3 MB per init
    731 inits (2020-2021 daily)        = 1.7 GB total
That is small enough that keeping per-member values (rather than pre-reducing
to quantiles) costs nothing and leaves us free to recompute p_t, try other
quantile levels, and run quantile-space ACI without regenerating. Pre-reducing
to [q05, q95, p_t] would save ~85% of 1.7 GB, which is not worth the lost
optionality.

TODO(lucas): [Thu] Confirm the above is the storage you want before launching
the full run. If you later decide you need all 5 lead days (e.g. to show the
effect grows with lead), that is 8.5 GB -- still fine, but regenerating costs
GPU hours, so decide once, now. See 02_benchmark.py for the wall-clock estimate.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from xconformal import config  # noqa: E402


def iter_init_dates(start: np.datetime64, end: np.datetime64) -> list[np.datetime64]:
    """Daily inits at config.INIT_HOUR from start to end, inclusive."""
    start = np.datetime64(start, "h")
    end = np.datetime64(end, "h")
    n = int((end - start) / np.timedelta64(1, "D"))
    return [start + np.timedelta64(i, "D") for i in range(n + 1)]


def load_model(device: str = "auto"):
    """Load the ArchesWeatherGen lightning module from config.GEN_MODEL_DIR.

    in: device; out: (module, cfg) from geoarches.lightning_modules.load_module.

    Import geoarches lazily -- it drags in torch, and the CLI's --help and the
    resume scan must stay fast and importable on a machine with no GPU.

    TODO(lucas): [Thu] geoarches.lightning_modules.load_module resolves a bare
    name against a literal "modelstore" directory relative to CWD, so pass the
    ABSOLUTE config.GEN_MODEL_DIR, not the model name. Verify the module that
    comes back is an EnsembleDiffusionModule and that it found the deterministic
    archesweather-m-* checkpoints its config references -- those must be
    downloaded too, or sampling silently uses an uninitialised backbone.
    """
    raise NotImplementedError(
        "in: device str; out: (EnsembleDiffusionModule, cfg) loaded from config.GEN_MODEL_DIR"
    )


def load_batch(init: np.datetime64):
    """Build the model input batch for one init date from ERA5.

    in: init datetime64; out: batch dict with keys state / prev_state /
    timestamp / lead_time_hours, as geoarches.dataloaders.era5.Era5Forecast
    produces (normalised TensorDicts on the model's device).

    TODO(lucas): [Thu] Era5Forecast overrides its own timestamp bounds when
    domain is "val"/"test", so construct it with an explicit filename_filter
    and set_timestamp_bounds rather than a domain string, or it will quietly
    hand back 2020 only. Reuse ONE dataset object across inits -- it opens every
    netcdf on construction and that is slow.
    """
    raise NotImplementedError(
        "in: init datetime64; out: batch dict (state, prev_state, timestamp, lead_time_hours)"
    )


def generate_one_init(module, batch, n_members: int, batch_size: int) -> xr.Dataset:
    """Sample an ensemble for one init and reduce it to what we archive.

    in: module, batch, n_members, batch_size;
    out: xr.Dataset with data var config.TARGET_VAR_SHORT of dims
         (member, lat, lon) at lead config.LEAD_DAYS, plus coords
         member/lat/lon and scalar coords init_time, valid_time, lead_days.

    Call module.sample_rollout(batch, batch_nb, iterations=config.ROLLOUT_ITERATIONS,
    member=m) per member, or a batched equivalent -- see 02_benchmark.py for
    which batch size is worth using.

    TODO(lucas): [Thu] sample_rollout seeds as `member + 1000*i + batch_nb*1e6`.
    Pass a batch_nb derived from the init date (not a loop counter), so that a
    resumed run regenerates bit-identical members for any init it redoes.
    Otherwise "resumable" is only true for files, not for values.
    """
    raise NotImplementedError(
        "in: module, batch, n_members, batch_size; "
        "out: xr.Dataset[TARGET_VAR_SHORT] dims (member, lat, lon) at lead LEAD_DAYS"
    )


def write_atomic(ds: xr.Dataset, path: Path) -> None:
    """Write ``ds`` to ``path`` via a temp file and an atomic rename.

    The temp name carries the pid so two concurrent workers cannot collide, and
    lives in the same directory so the rename stays on one filesystem (a rename
    across filesystems is a copy, and is not atomic).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        ds.to_netcdf(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)  # includes KeyboardInterrupt: leave no debris
        raise


def clean_stale_temps(directory: Path) -> int:
    """Delete leftover .tmp-* files from a previously killed run."""
    stale = list(directory.glob("*.tmp-*")) if directory.is_dir() else []
    for path in stale:
        path.unlink(missing_ok=True)
    return len(stale)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", type=np.datetime64, default=config.INIT_START,
                        help=f"first init date (default {config.INIT_START})")
    parser.add_argument("--end", type=np.datetime64, default=config.INIT_END,
                        help=f"last init date, inclusive (default {config.INIT_END})")
    parser.add_argument("--members", type=int, default=config.N_MEMBERS,
                        help=f"ensemble members per init (default {config.N_MEMBERS})")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="members sampled per forward pass; see 02_benchmark.py")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many inits (smoke test)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be generated and exit; loads no model")
    parser.add_argument("--clean-temps", action="store_true",
                        help="delete stale .tmp-* files from a killed run, then continue")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.validate_paths(require_model=not args.dry_run)

    if args.clean_temps:
        print(f"removed {clean_stale_temps(config.FORECAST_DIR)} stale temp files")

    inits = iter_init_dates(args.start, args.end)
    todo = [d for d in inits if not config.forecast_file(d).exists()]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"window   : {args.start} .. {args.end}  ({len(inits)} inits)")
    print(f"present  : {len(inits) - len([d for d in inits if not config.forecast_file(d).exists()])}")
    print(f"to do    : {len(todo)}")
    print(f"output   : {config.FORECAST_DIR}")
    if args.dry_run:
        for date in todo[:10]:
            print(f"  would write {config.forecast_file(date)}")
        if len(todo) > 10:
            print(f"  ... and {len(todo) - 10} more")
        return 0
    if not todo:
        print("nothing to do")
        return 0

    module, _ = load_model(args.device)
    started = time.time()
    for i, init in enumerate(todo, start=1):
        out_path = config.forecast_file(init)
        if out_path.exists():
            continue  # another worker got here first
        batch = load_batch(init)
        ds = generate_one_init(module, batch, args.members, args.batch_size)
        write_atomic(ds, out_path)
        elapsed = time.time() - started
        rate = elapsed / i
        print(f"[{i}/{len(todo)}] {out_path.name}  "
              f"{rate:.1f}s/init  eta {(len(todo) - i) * rate / 3600:.1f}h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
