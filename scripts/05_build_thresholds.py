#!/usr/bin/env python
"""Build the climatological threshold field and the ACI scale field.

Network-bound, not GPU-bound: run it in tmux session `clim` alongside
generation.

    tmux new -s clim -d 'source /workspace/env.sh && cd $PROJECT_ROOT && \
        python scripts/05_build_thresholds.py 2>&1 | tee thresholds.log'

Streams ONLY 2m_temperature at 0Z from the WeatherBench2 zarr (~1.3 GB through
memory for 1990-2019) and persists only the two derived fields (~43 MB and
~0.1 MB). The raw slice is never written to disk -- `geoarches.download.dl_era`
would have pulled ~165 GB to answer the same question because it fetches every
variable and pressure level.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xconformal import config, thresholds  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-scale", action="store_true",
                        help="build only the threshold field")
    parser.add_argument("--skip-threshold", action="store_true",
                        help="build only the ACI scale field")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.THRESHOLD_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_scale:
        t0 = time.time()
        thresholds.build_aci_scale(overwrite=args.overwrite)
        print(f"  ACI scale in {time.time() - t0:.0f}s\n")

    if not args.skip_threshold:
        t0 = time.time()
        thresholds.build_thresholds(overwrite=args.overwrite)
        print(f"  thresholds in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
