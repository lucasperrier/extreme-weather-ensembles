#!/usr/bin/env python
"""GPU bring-up gate: variable indices, one sample, one rollout, physical ranges.

Everything downstream assumes t2m is a surface variable at a known index, that
sampling runs on this GPU, and that outputs denormalize to physical Kelvin.
This script proves all three and prints the inference config that goes in the
paper. Run it before 02_benchmark.py and before launching generation.

    python scripts/06_smoke_gpu.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from xconformal import config  # noqa: E402

RULE = "=" * 78
# Physical plausibility band for global 2m temperature, in Kelvin. Anything
# outside this is a denormalization bug, not weather.
T2M_MIN_K, T2M_MAX_K = 180.0, 330.0


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def timed(fn, *args, **kwargs) -> tuple[object, float]:
    """Run fn with CUDA synchronization on both sides; return (result, seconds)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--init", type=np.datetime64, default=config.INIT_START)
    parser.add_argument("--repeats", type=int, default=2,
                        help="timed runs after the discarded warm-up")
    args = parser.parse_args(argv)

    from xconformal import generate as gen  # noqa: E402  (imports geoarches; slow)
    from geoarches.dataloaders import era5  # noqa: E402

    # ---------------------------------------------------------------- vars
    section("VARIABLES  (gate: t2m must be a surface variable)")
    print(f"  surface_variables      : {era5.surface_variables}")
    print(f"  surface_variables_short: "
          f"{[era5.surface_variables_short[v] for v in era5.surface_variables]}")
    print(f"  level_variables        : {era5.level_variables}")
    print(f"  pressure_levels        : {era5.pressure_levels}")

    if config.TARGET_VAR not in era5.surface_variables:
        print(f"\n  !! STOP: {config.TARGET_VAR} is NOT a surface variable.")
        return 2
    idx = era5.surface_variables.index(config.TARGET_VAR)
    short = era5.surface_variables_short[config.TARGET_VAR]
    print(f"\n  CONFIRMED: {config.TARGET_VAR} is surface index {idx} (short {short!r})")
    print(f"  config.TARGET_SURFACE_INDEX = {config.TARGET_SURFACE_INDEX}")
    if idx != config.TARGET_SURFACE_INDEX:
        print(f"  !! STOP: config says {config.TARGET_SURFACE_INDEX}, dataloader says {idx}")
        return 2
    print(f"  get_surface_variable_indices()[{short!r}] = "
          f"{era5.get_surface_variable_indices()[short]}")

    # ---------------------------------------------------------------- load
    section("MODEL")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}  "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'})")
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM  : {total:.1f} GB")
    (module, cfg), load_s = timed(gen.load_model, device)
    print(f"  loaded {config.GEN_MODEL_NAME} in {load_s:.1f}s -> {type(module).__name__}")

    inf = cfg.module.inference
    scale_input_noise = getattr(inf, "scale_input_noise", None)
    print("\n  Inference config actually in force (these go in the paper):")
    print(f"    num_steps          : {inf.num_steps}")
    print(f"    cf_guidance        : {inf.cf_guidance}")
    print(f"    s_churn            : {inf.s_churn}")
    print(f"    scale_input_noise  : {scale_input_noise!r}"
          + ("  (absent from config -> sample() applies no rescaling)"
             if scale_input_noise is None else ""))
    print(f"    scheduler          : {cfg.module.module.scheduler}")
    print(f"    prediction_type    : {cfg.module.module.prediction_type}")
    print(f"    learn_residual     : {cfg.module.module.learn_residual}")
    print(f"    conditional        : {cfg.module.module.conditional}")
    det = list(cfg.module.module.load_deterministic_model)
    print(f"    deterministic models ({len(det)}):")
    for name in det:
        print(f"      {name}")
    print(f"    det backbone       : {type(module.det_model).__name__} "
          f"averaging {len(module.det_model.core)} models")
    print(f"    seed scheme        : member + 1000*step + batch_nb*1e6, "
          f"batch_nb = days since 1970-01-01 (init {args.init} -> "
          f"{gen.init_batch_nb(args.init)})")

    # ---------------------------------------------------------------- batch
    section("BATCH")
    (batch, ds), batch_s = timed(gen.load_batch, args.init)
    print(f"  init {args.init}, built in {batch_s:.1f}s "
          f"(dataset spans {gen.init_dates(ds)[0]} .. {gen.init_dates(ds)[-1]}, "
          f"{len(gen.init_dates(ds))} inits)")
    for key in ("state", "prev_state"):
        td = batch[key]
        print(f"    {key:<11s} {dict((k, tuple(v.shape)) for k, v in td.items())}")
    print(f"    timestamp        {batch['timestamp']}")
    print(f"    lead_time_hours  {batch['lead_time_hours']}")

    # --------------------------------------------------------------- steps
    section("SAMPLING  (warm-up discarded; CUDA-synchronized both sides)")
    _, warm_s = timed(module.sample, batch, seed=0, disable_tqdm=True)
    print(f"  warm-up single 24h step : {warm_s:.2f}s  (DISCARDED)")

    step_times = []
    for i in range(args.repeats):
        sample, dt = timed(module.sample, batch, seed=100 + i, disable_tqdm=True)
        step_times.append(dt)
        print(f"  single 24h step  run {i}: {dt:.2f}s")
    step_s = float(np.median(step_times))

    roll_times = []
    for i in range(args.repeats):
        rollout, dt = timed(
            module.sample_rollout, batch,
            batch_nb=gen.init_batch_nb(args.init),
            iterations=config.ROLLOUT_ITERATIONS, member=0, disable_tqdm=True,
        )
        roll_times.append(dt)
        print(f"  {config.LEAD_DAYS}-day rollout run {i}: {dt:.2f}s "
              f"({dt / config.ROLLOUT_ITERATIONS:.2f}s/step)")
    roll_s = float(np.median(roll_times))

    print(f"\n  median single step   : {step_s:.2f}s")
    print(f"  median 5-day rollout : {roll_s:.2f}s per member")
    if roll_s > 60.0:
        print(f"  !! STOP: per-member 5-day time {roll_s:.1f}s exceeds the 60s budget cap.")
        return 2
    print(f"  within the 60s/member escalation cap ({roll_s:.1f}s)")

    # ------------------------------------------------------- denormalize
    section("DENORMALIZATION  (gate: physical t2m in Kelvin)")
    print(f"  rollout tensordict: "
          f"{dict((k, tuple(v.shape)) for k, v in rollout.items())}")
    t2m = gen.extract_t2m(rollout, ds)[0]  # drop the batch-of-1
    print(f"  extracted t2m     : shape {t2m.shape} dtype {t2m.dtype}")
    for lead, field in zip(config.ARCHIVED_LEADS, t2m):
        print(f"    lead {lead}d: min {field.min():7.2f}  mean {field.mean():7.2f}  "
              f"max {field.max():7.2f} K")
    lo, hi = float(t2m.min()), float(t2m.max())
    if not (T2M_MIN_K <= lo and hi <= T2M_MAX_K):
        print(f"\n  !! STOP: t2m range [{lo:.1f}, {hi:.1f}] K is outside the physical "
              f"band [{T2M_MIN_K}, {T2M_MAX_K}] -- denormalization is wrong.")
        return 2
    print(f"\n  CONFIRMED physical: [{lo:.2f}, {hi:.2f}] K inside "
          f"[{T2M_MIN_K}, {T2M_MAX_K}]")
    if np.isnan(t2m).any():
        print("  !! STOP: NaNs in output")
        return 2
    print("  no NaNs")

    section("GATE PASSED")
    print(f"  s/member (5-day, serial) : {roll_s:.2f}")
    print(f"  members/hour             : {3600 / roll_s:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
