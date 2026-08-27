#!/usr/bin/env python
"""Time ArchesWeatherGen sampling at several batch sizes and project the full run.

The number this prints decides whether the 731-init x 20-member archive fits in
the days remaining. Run it before launching 01_generate.py, not after.

    source /workspace/env.sh
    python scripts/02_benchmark.py --batch-sizes 1 4 16 32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from xconformal import config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

_gen = import_module("01_generate")


def time_batch_size(module, batch, batch_size: int, repeats: int) -> float:
    """Median seconds to sample ONE member at this batch size.

    in: module, batch, batch_size, repeats; out: seconds per member.

    Sample ``batch_size`` members in one call, divide the wall time by
    ``batch_size``, and take the median over ``repeats`` calls.

    TODO(lucas): [Thu] Discard the first call from the timing -- it pays for
    CUDA context creation and kernel autotuning and is not representative. Call
    torch.cuda.synchronize() before reading the clock, or you time the launch
    queue rather than the work. Watch for OOM at 32 and report it as a result
    (the largest batch size that fits IS the answer) rather than crashing.
    """
    raise NotImplementedError(
        "in: module, batch, batch_size, repeats; out: median seconds per member (float)"
    )


def project(seconds_per_member: float, n_inits: int, n_members: int) -> dict:
    """Turn a per-member timing into members/hour and a full-run wall clock."""
    members_per_hour = 3600.0 / seconds_per_member if seconds_per_member > 0 else float("inf")
    total_members = n_inits * n_members
    return {
        "seconds_per_member": seconds_per_member,
        "members_per_hour": members_per_hour,
        "total_members": total_members,
        "wall_clock_hours": total_members / members_per_hour,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init", type=np.datetime64, default=config.INIT_START,
                        help="which init date to benchmark on")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.validate_paths(require_model=True)

    n_inits = len(_gen.iter_init_dates(config.INIT_START, config.INIT_END))
    print(f"target archive: {n_inits} inits x {config.N_MEMBERS} members "
          f"= {n_inits * config.N_MEMBERS} member-forecasts "
          f"({config.ROLLOUT_ITERATIONS} rollout steps each)")

    load_start = time.time()
    module, _ = _gen.load_model(args.device)
    batch = _gen.load_batch(args.init)
    print(f"model + batch loaded in {time.time() - load_start:.1f}s\n")

    rows = []
    header = f"{'batch':>6s} {'s/member':>10s} {'members/h':>11s} {'full run (h)':>13s}"
    print(header)
    print("-" * len(header))
    for batch_size in args.batch_sizes:
        try:
            per_member = time_batch_size(module, batch, batch_size, args.repeats)
        except NotImplementedError:
            raise
        except RuntimeError as exc:  # most likely CUDA OOM: that is a result
            print(f"{batch_size:>6d} {'OOM':>10s}  ({exc})")
            rows.append({"batch_size": batch_size, "seconds_per_member": float("nan"),
                         "members_per_hour": 0.0, "wall_clock_hours": float("inf")})
            continue
        stats = project(per_member, n_inits, config.N_MEMBERS)
        rows.append({"batch_size": batch_size, **stats})
        print(f"{batch_size:>6d} {stats['seconds_per_member']:>10.2f} "
              f"{stats['members_per_hour']:>11.1f} {stats['wall_clock_hours']:>13.1f}")

    config.BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(rows).to_csv(config.BENCHMARK_PATH, index=False)
    print(f"\nwrote {config.BENCHMARK_PATH}")
    print("If the full run does not fit, the levers in order of preference are: "
          "fewer members, shorter init window, coarser init spacing (every other day). "
          "Do NOT shorten the lead -- it is a fixed experimental decision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
