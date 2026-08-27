#!/usr/bin/env python
"""Check generated ensemble files before trusting a long run.

Run this on the first handful of files after launching 01_generate.py. Every
check here catches something that would otherwise be discovered on Saturday:

  shapes     -- wrong member/lead count means the archive is not what config says
  NaNs       -- one NaN gridpoint poisons every quantile that touches it
  range      -- non-physical Kelvin means denormalization is wrong
  spread     -- zero member spread means the "ensemble" is one forecast repeated,
                which would make every coverage number in the paper meaningless
  provenance -- a file with no provenance attrs is unreproducible later

    python scripts/07_verify_outputs.py --limit 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from xconformal import config  # noqa: E402

T2M_MIN_K, T2M_MAX_K = 180.0, 330.0
REQUIRED_ATTRS = (
    "geoarches_version", "generative_model", "deterministic_models",
    "checkpoint_sizes", "num_steps", "scale_input_noise", "seed_scheme",
    "era5_source", "sampling_batch_size",
)


def check(path: Path, members: int, leads: int) -> list[str]:
    """Return a list of problems with one file; empty means it passed."""
    problems = []
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        if "t2m" not in ds.data_vars:
            return [f"no t2m variable (has {list(ds.data_vars)})"]
        da = ds["t2m"]
        expected = ("member", "lead", "latitude", "longitude")
        if da.dims != expected:
            problems.append(f"dims {da.dims} != {expected}")
        if da.sizes.get("member") != members:
            problems.append(f"{da.sizes.get('member')} members, expected {members}")
        if da.sizes.get("lead") != leads:
            problems.append(f"{da.sizes.get('lead')} leads, expected {leads}")
        if da.dtype != np.float32:
            problems.append(f"dtype {da.dtype}, expected float32")

        values = da.values
        if np.isnan(values).any():
            problems.append(f"{int(np.isnan(values).sum())} NaNs")
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        if not (T2M_MIN_K <= lo and hi <= T2M_MAX_K):
            problems.append(f"range [{lo:.1f}, {hi:.1f}] K outside physical band")

        # Member spread, at the verification lead, averaged over the grid.
        last = values[:, -1]
        spread = float(last.std(axis=0).mean())
        if spread <= 1e-6:
            problems.append(f"member spread {spread:.2e} K -- members are identical")

        missing = [a for a in REQUIRED_ATTRS if a not in ds.attrs]
        if missing:
            problems.append(f"missing provenance attrs: {missing}")

        return problems, lo, hi, spread
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--members", type=int, default=config.N_MEMBERS)
    args = parser.parse_args(argv)

    files = sorted(config.ENSEMBLE_DIR.glob("*.nc"))[: args.limit]
    if not files:
        print(f"no files under {config.ENSEMBLE_DIR}")
        return 1

    leads = len(config.ARCHIVED_LEADS)
    print(f"{'file':<18s} {'min K':>7s} {'max K':>7s} {'spread K':>9s}  status")
    print("-" * 60)
    failed = 0
    for path in files:
        result = check(path, args.members, leads)
        problems, lo, hi, spread = result if isinstance(result, tuple) else (result, 0, 0, 0)
        status = "ok" if not problems else "; ".join(problems)
        failed += bool(problems)
        print(f"{path.name:<18s} {lo:>7.1f} {hi:>7.1f} {spread:>9.2f}  {status}")

    print(f"\n{len(files) - failed}/{len(files)} files passed")
    if failed:
        print("!! do NOT leave generation running on these outputs")
    else:
        with xr.open_dataset(files[0], decode_timedelta=False) as ds:
            size = files[0].stat().st_size / 1e6
            print(f"per-file size {size:.1f} MB; provenance: geoarches "
                  f"{ds.attrs['geoarches_version']}, num_steps {ds.attrs['num_steps']}, "
                  f"batch_size {ds.attrs['sampling_batch_size']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
