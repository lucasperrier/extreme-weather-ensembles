#!/usr/bin/env python
"""Print the ground truth about what is actually on this pod.

Several TODOs elsewhere in the repo say "run this script". This is why: the
climatology's variable list, whether the 2020 ERA5 download has finished, and
whether the checkpoints are real files rather than saved HTML error pages are
all facts we assumed rather than checked. Everything printed here is read-only.

    python scripts/00_inspect_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from xconformal import config  # noqa: E402

RULE = "=" * 78


def human(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def describe_dataset(path: Path, max_vars: int = 60) -> None:
    """Print data_vars, coords and time range without loading the payload."""
    try:
        ds = xr.open_dataset(path, decode_timedelta=False)
    except Exception as exc:  # noqa: BLE001 - diagnostic script, report anything
        print(f"  !! cannot open: {type(exc).__name__}: {exc}")
        return
    with ds:
        print(f"  dims        : {dict(ds.sizes)}")
        print(f"  coords      : {list(ds.coords)}")
        names = list(ds.data_vars)
        print(f"  data_vars   : {len(names)}")
        for name in names[:max_vars]:
            var = ds[name]
            print(f"      {name:<44s} {tuple(var.dims)} {var.dtype}")
        if len(names) > max_vars:
            print(f"      ... and {len(names) - max_vars} more")
        for tname in ("time", "hour", "dayofyear"):
            if tname in ds.coords:
                coord = ds[tname]
                n = coord.size
                if n == 0:
                    print(f"  {tname:<11s} : EMPTY (size 0) -- treat this file as MISSING")
                else:
                    print(f"  {tname:<11s} : n={n} first={coord.values[0]} last={coord.values[-1]}")


def era5_last_time(hour: int):
    """Latest valid time available in the ERA5 files for one analysis hour."""
    latest = None
    for path in sorted(config.ERA5_FULL_DIR.glob(f"era5_240_*_{hour}h.nc")):
        try:
            with xr.open_dataset(path, decode_timedelta=False) as ds:
                if ds.sizes.get("time", 0) == 0:
                    continue  # empty time dim: missing data, not an empty year
                last = ds.time.values[-1]
        except Exception:  # noqa: BLE001 - diagnostic script
            continue
        latest = last if latest is None else max(latest, last)
    return latest


def main() -> int:
    section("PATHS")
    for label in ("DATA_ROOT", "MODEL_ROOT", "EVAL_ROOT", "PROJECT_ROOT"):
        path = getattr(config, label)
        print(f"  {label:<12s} {path}  exists={path.exists()}")
    print(f"  {'ERA5_FULL_DIR':<12s} {config.ERA5_FULL_DIR}  exists={config.ERA5_FULL_DIR.is_dir()}")
    print(f"  {'FORECAST_DIR':<12s} {config.FORECAST_DIR}  exists={config.FORECAST_DIR.is_dir()}")

    # ---------------------------------------------------------------- clim
    section("CLIMATOLOGY  (resolves: config.TARGET_VAR fallback, thresholds.py source)")
    print(f"  path: {config.ERA5_CLIM_PATH}")
    if not config.ERA5_CLIM_PATH.is_file():
        print("  !! MISSING. Fetch with:")
        print(f"     python -m geoarches.download.dl_era --folder {config.ERA5_FULL_DIR} --clim --years")
        print("  Until then thresholds.build_thresholds(source='era5') is the only option.")
    else:
        print(f"  size: {human(config.ERA5_CLIM_PATH.stat().st_size)}")
        describe_dataset(config.ERA5_CLIM_PATH)
        print("\n  -> Does it carry a 95th-percentile field, or only means?")
        print("     If only means, thresholds.build_thresholds(source='era5').")

    # ---------------------------------------------------------------- era5
    section("ERA5 TRUTH  (resolves: which init dates 01_generate.py can cover)")
    files = sorted(config.ERA5_FULL_DIR.glob("era5_240_*.nc")) if config.ERA5_FULL_DIR.is_dir() else []
    if not files:
        print(f"  !! no era5_240_*.nc under {config.ERA5_FULL_DIR}")
    for path in files:
        size = path.stat().st_size
        try:
            ds = xr.open_dataset(path, decode_timedelta=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  {path.name:<28s} {human(size):>9s}  !! unreadable: {type(exc).__name__}")
            continue
        with ds:
            n = ds.sizes.get("time", 0)
            if n == 0:
                # An empty time dimension is the failure mode to watch for: the
                # file exists, has a plausible size, and would silently vanish
                # from an xr.concat. Never concatenate one.
                print(f"  {path.name:<28s} {human(size):>9s}  time=0  !! EMPTY -> TREAT AS MISSING")
            else:
                t0, t1 = ds.time.values[0], ds.time.values[-1]
                print(f"  {path.name:<28s} {human(size):>9s}  time={n:<5d} {t0} .. {t1}")
                if config.TARGET_VAR not in ds.data_vars:
                    print(f"      !! {config.TARGET_VAR} ABSENT; has {list(ds.data_vars)[:6]} ...")

    print("\n  Init window wanted by config: "
          f"{config.INIT_START} .. {config.INIT_END} at {config.INIT_HOUR:02d}Z")
    last_valid = config.INIT_END + np.timedelta64(config.LEAD_DAYS, "D")
    print(f"  Verifying times needed      : init + {config.LEAD_DAYS} days, "
          f"so truth is needed out to {last_valid}")
    truth_end = era5_last_time(config.INIT_HOUR)
    if truth_end is None:
        print(f"  !! no readable {config.INIT_HOUR:02d}Z ERA5 file -- cannot verify anything")
    elif truth_end < last_valid:
        short = int((last_valid - truth_end) / np.timedelta64(1, "D"))
        print(f"  !! ERA5 {config.INIT_HOUR:02d}Z ends at {truth_end}: the last {short} "
              f"init(s) of the window have NO verifying truth.")
        print(f"     Either download the next year, or move config.INIT_END back "
              f"{short} day(s). Do not let 03_verify.py silently drop them.")
    else:
        print(f"  ERA5 {config.INIT_HOUR:02d}Z ends at {truth_end}: window fully covered.")

    # --------------------------------------------------------- checkpoints
    section("CHECKPOINTS  (resolves: whether 01_generate.py / 02_benchmark.py can run)")
    print(f"  MODEL_ROOT: {config.MODEL_ROOT}")
    entries = sorted(p for p in config.MODEL_ROOT.glob("*") if p.is_dir()) if config.MODEL_ROOT.is_dir() else []
    if not entries:
        print("  !! MODEL_ROOT is EMPTY -- no checkpoints. Fetch all five with:")
        print(f"     python -m geoarches.download.dl_aw_models --output-directory {config.MODEL_ROOT}")
    for model_dir in entries:
        print(f"\n  {model_dir.name}/")
        cfg = model_dir / "config.yaml"
        print(f"      config.yaml            exists={cfg.is_file()}"
              + (f" ({human(cfg.stat().st_size)})" if cfg.is_file() else ""))
        ckpts = sorted((model_dir / "checkpoints").glob("*.ckpt"))
        if not ckpts:
            print("      checkpoints/*.ckpt     !! NONE")
        for ckpt in ckpts:
            size = ckpt.stat().st_size
            with open(ckpt, "rb") as fh:
                magic = fh.read(4)
            # A wget that hit a redirect or a login page leaves an HTML file
            # with a .ckpt name. Real torch checkpoints are zip archives (PK\x03\x04).
            looks_html = magic[:1] == b"<"
            flag = "  !! LOOKS LIKE HTML, NOT A CHECKPOINT" if looks_html else ""
            print(f"      {ckpt.name:<22s} {human(size):>9s} magic={magic!r}{flag}")

    # ------------------------------------------------------------ forecasts
    section("FORECAST ARCHIVE  (resolves: how much of 01_generate.py remains)")
    print(f"  path: {config.FORECAST_DIR}")
    if config.FORECAST_DIR.is_dir():
        done = sorted(config.FORECAST_DIR.glob("*.nc"))
        partial = sorted(config.FORECAST_DIR.glob("*.tmp-*"))
        wanted = int((config.INIT_END - config.INIT_START) / np.timedelta64(1, "D")) + 1
        print(f"  complete: {len(done)} / {wanted} inits")
        if done:
            print(f"  first={done[0].name}  last={done[-1].name}")
            print(f"  size/init={human(done[0].stat().st_size)}  "
                  f"projected total={human(done[0].stat().st_size * wanted)}")
        if partial:
            print(f"  !! {len(partial)} stale .tmp-* files (killed mid-write); safe to delete")
    else:
        print("  not created yet -- 01_generate.py has not run")

    # ------------------------------------------------------------ capacity
    section("DISK")
    print("  Container disk is 20 GB and EPHEMERAL. Everything must live on /workspace.")
    print("  `df` on MooseFS reports the whole cluster, not our quota -- use `du -sh` on")
    print("  a directory to learn what we are actually using.")
    for label in ("DATA_ROOT", "MODEL_ROOT", "EVAL_ROOT"):
        path = getattr(config, label)
        if path.is_dir():
            total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            print(f"  du {label:<11s} {human(total):>10s}  {path}")

    print(f"\n{RULE}\nRecord anything surprising in RUNBOOK.md under Gotchas.\n{RULE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
