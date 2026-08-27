#!/usr/bin/env python
"""Time member sampling serially and batched, and project the full run.

The number this prints decides the init grid. Run it before launching
generation, not after.

    python scripts/02_benchmark.py --batch-sizes 1 2 4 8

Batching correctness. geoarches draws the initial noise for a whole batch from
ONE generator in a single stream, so a member's noise depends on how many
members share its pass. Two things are therefore checked, not assumed:
  (a) members within a batch differ from each other -- if they did not, the
      "ensemble" would be one forecast repeated and every coverage number in
      the paper would be meaningless;
  (b) batched member 0 reproduces the serial member 0 at the same seed.
(b) is the strict test. If it fails, batching still gives a valid ensemble but
a different one, and the archive is only resumable at a pinned batch size.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from xconformal import config, generate as gen  # noqa: E402

WALL_CLOCK_BUDGET_H = 40.0  # sizing rule: thin the calibration year above this


def timed(fn, *args, **kwargs):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def n_inits(stride_cal: int, stride_ver: int) -> tuple[int, int]:
    """Inits in the configured window at the given per-year strides."""
    day = np.timedelta64(1, "D")
    total = int((config.INIT_END - config.INIT_START) / day) + 1
    dates = [config.INIT_START + i * day for i in range(total)]
    cal = [d for d in dates if d.astype("datetime64[Y]").astype(int) + 1970 == config.CALIBRATION_YEAR]
    ver = [d for d in dates if d.astype("datetime64[Y]").astype(int) + 1970 == config.VERIFICATION_YEAR]
    return len(cal[::stride_cal]), len(ver[::stride_ver])


def project(seconds_per_member: float, stride_cal: int = 1, stride_ver: int = 1) -> dict:
    cal, ver = n_inits(stride_cal, stride_ver)
    inits = cal + ver
    members = inits * config.N_MEMBERS
    hours = members * seconds_per_member / 3600.0
    disk_gb = (inits * config.N_MEMBERS * len(config.ARCHIVED_LEADS) * 121 * 240 * 4) / 1e9
    return {"stride_cal": stride_cal, "stride_ver": stride_ver, "inits_cal": cal,
            "inits_ver": ver, "inits": inits, "members": members,
            "wall_clock_h": hours, "disk_gb": disk_gb}


def choose_grid(seconds_per_member: float) -> dict:
    """SIZING RULE: <=40 h keeps the full plan; above it, thin 2020 only."""
    for stride in (1, 2, 3, 4, 5):
        plan = project(seconds_per_member, stride_cal=stride, stride_ver=1)
        if plan["wall_clock_h"] <= WALL_CLOCK_BUDGET_H:
            return plan
    return project(seconds_per_member, stride_cal=5, stride_ver=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init", type=np.datetime64, default=config.INIT_START)
    args = parser.parse_args(argv)
    config.validate_paths(require_model=True)

    module, _ = gen.load_model(args.device)
    batch, dataset = gen.load_batch(args.init, device=args.device)
    batch_nb = gen.init_batch_nb(args.init)
    print(f"init {args.init} (batch_nb {batch_nb}), "
          f"{config.ROLLOUT_ITERATIONS} rollout steps to day {config.LEAD_DAYS}\n")

    # Warm-up, discarded: the first call pays for CUDA context creation and
    # kernel autotuning and is not representative.
    gen.sample_members(module, batch, dataset, args.init, [0], batch_size=1)

    reference = None
    rows = []
    header = (f"{'batch':>6s} {'s/pass':>8s} {'s/member':>9s} {'members/h':>10s} "
              f"{'VRAM GB':>8s} {'distinct':>9s} {'==serial':>9s}")
    print(header)
    print("-" * len(header))

    for bs in args.batch_sizes:
        members = list(range(bs))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            times, out = [], None
            for _ in range(args.repeats):
                out, dt = timed(gen.sample_members, module, batch, dataset,
                                args.init, members, batch_size=bs)
                times.append(dt)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{bs:>6d} {'OOM':>8s}  (largest usable batch is below this)")
            rows.append({"batch_size": bs, "oom": True})
            break

        per_pass = float(np.median(times))
        per_member = per_pass / bs
        vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")

        # (a) members within the pass must differ from each other
        if bs == 1:
            distinct = "n/a"
            reference = out[0].copy()
        else:
            spread = float(np.abs(out - out.mean(axis=0, keepdims=True)).max())
            distinct = "yes" if spread > 1e-4 else "NO"
        # (b) batched member 0 vs serial member 0 at the same seed
        if bs == 1 or reference is None:
            matches = "n/a"
        else:
            same = np.allclose(out[0], reference, atol=1e-4)
            matches = "yes" if same else f"no({np.abs(out[0]-reference).max():.2f}K)"

        rows.append({"batch_size": bs, "seconds_per_pass": per_pass,
                     "seconds_per_member": per_member,
                     "members_per_hour": 3600 / per_member, "vram_gb": vram,
                     "members_distinct": distinct, "matches_serial": matches})
        print(f"{bs:>6d} {per_pass:>8.2f} {per_member:>9.2f} {3600/per_member:>10.1f} "
              f"{vram:>8.1f} {distinct:>9s} {matches:>9s}")

    ok = [r for r in rows if not r.get("oom")]
    best = min(ok, key=lambda r: r["seconds_per_member"])
    serial = next(r for r in ok if r["batch_size"] == 1)
    print(f"\nfastest batch size: {best['batch_size']} at "
          f"{best['seconds_per_member']:.2f} s/member "
          f"({serial['seconds_per_member'] / best['seconds_per_member']:.2f}x serial)")

    print(f"\n{'grid':<34s} {'inits':>6s} {'members':>8s} {'wall-clock h':>13s} {'disk GB':>9s}")
    print("-" * 74)
    for label, secs in (("serial", serial["seconds_per_member"]),
                        (f"batched x{best['batch_size']}", best["seconds_per_member"])):
        for stride in (1, 2, 3):
            plan = project(secs, stride_cal=stride, stride_ver=1)
            tag = f"{label}, 2020 every {stride}d" if stride > 1 else f"{label}, daily"
            flag = "" if plan["wall_clock_h"] <= WALL_CLOCK_BUDGET_H else "  OVER"
            print(f"{tag:<34s} {plan['inits']:>6d} {plan['members']:>8d} "
                  f"{plan['wall_clock_h']:>13.1f} {plan['disk_gb']:>9.1f}{flag}")

    chosen = choose_grid(best["seconds_per_member"])
    print(f"\nSIZING RULE (<= {WALL_CLOCK_BUDGET_H:g} h, never thin "
          f"{config.VERIFICATION_YEAR}) -> "
          f"{config.CALIBRATION_YEAR} every {chosen['stride_cal']}d, "
          f"{config.VERIFICATION_YEAR} every {chosen['stride_ver']}d: "
          f"{chosen['inits']} inits ({chosen['inits_cal']}+{chosen['inits_ver']}), "
          f"{chosen['wall_clock_h']:.1f} h, {chosen['disk_gb']:.1f} GB")

    import pandas as pd
    config.EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.BENCHMARK_PATH, index=False)
    print(f"wrote {config.BENCHMARK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
