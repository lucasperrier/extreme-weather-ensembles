# LOG

Dated lab notebook. One entry per working session. Keep it terse: what was run,
what came out, what it changed. Durable operational facts belong in
`RUNBOOK.md`; this file is the narrative of how we got there.

Entry template:

```
## YYYY-MM-DD (Day)

**Goal.**
**Ran.**
**Result.**
**Decided.**
**Next.**
```

---

## 2026-08-27 (Thu)

**Goal.** Stand up the repo scaffold so the science can go straight into it.

**Ran.** Scaffolded `src/xconformal/{config,aci,thresholds,binning,coverage}.py`,
`scripts/00`–`04`, `tests/`. `pytest -q`.

**Result.** Scaffold in place; stubs raise `NotImplementedError` with an
inputs/outputs spec. `tests/test_aci_synthetic.py` is written in full and marked
`xfail` — it is the contract `aci.py` gets implemented against.

Checked against the pod (see `00_inspect_data.py` for the live version):
- `$MODEL_ROOT` is **empty** — no checkpoints yet.
- `$DATA_ROOT/era5_240/era5_240_clim.nc` **does not exist** yet.
- ERA5 2020 and 2021 present (4 files each, 0/6/12/18Z); 2020 was still being
  written at the time of checking. No 2022 files.

A reference implementation of the ACI loop on a synthetic under-dispersed
stream (σ=0.5, 20 members, 20 000 steps, τ=5) reached 0.8974 coverage in
**variable space** but stalled at **0.634** in **quantile space**, with the
effective level clipped at zero on 68% of steps — the interval was already
[min, max] and could not widen. Worth confirming on real data; if it survives,
it is a result, not a nuisance.

**Decided.** Nothing scientific. Adaptation space, target variable, threshold
source and day-of-year smoothing all remain open — see RUNBOOK "Decisions".

**Next.** Fetch checkpoints and climatology, run `00_inspect_data.py`, implement
`aci.py` against the test.


## 2026-08-27 (Thu) — GPU bring-up, sizing, batch launch

**Goal.** Get ensemble generation launched and verified.

**Environment.** The pod is an **RTX 4090 (24 GB)**, not an A100 — every sizing
number below is measured on it, not scaled from A100 figures. `tmux` is present
but lives on the ephemeral disk, so it is a per-pod reinstall. `$HF_TOKEN` was
not exported; a placeholder was added to `env.sh` and flagged. Nothing is
blocked by it — the ArchesWeather repo and the stats files are public, and an
empty value keeps `hf_hub` anonymous rather than sending a bad Authorization
header.

**Quantiles file — verdict: unusable.**
`era5-quantiles-2016_2022.nc` (133.8 MB, resolved via
`geoarches.stats.resolve_quantiles_file`) has dims
`(quantile: 6, longitude: 240, latitude: 121, level: 13)`. It fails **both**
arms of the decision rule: there is **no day-of-year dimension**, and the
quantile levels are `[1e-4, 1e-3, 0.01, 0.99, 0.999, 0.9999]` with **no 0.95**.
It pools all seasons together, so its upper quantiles are effectively a summer
threshold no winter day can reach.

**Threshold plan — built today, not deferred to Friday.** `dl_era` was the
wrong tool: it pulls every variable and pressure level, so 2011–2019 at four
synoptic hours would have been ~165 GB on disk (total projection 222 GB, over
the 200 GB escalation ceiling) to produce a 42 MB field. Instead, streamed only
`2m_temperature` at 0Z from the WB2 zarr:

| field | value |
|---|---|
| source | `gs://weatherbench2/.../1959-2022-6h-240x121_...zarr` |
| years / hour | 1990–2019, 0Z only |
| through memory | 10 957 timesteps, 1.27 GB (raw slice never persisted) |
| window | circular ±15 days → **907–930 samples** per gridpoint per day-of-year |
| output | `t2m_p95_clim_1990-2019_0z.nc`, 42.5 MB, 158 s |
| sanity | annual range 23.3 K at 30N/0E vs 4.0 K at the equator — seasonality present |

Conditioning on valid hour is not optional: t2m has a large diurnal cycle, so
pooling synoptic hours samples a *mixture* distribution rather than giving a
bigger sample of the same one.

Also built `t2m_std_2020_0z.nc` (0.13 MB), the per-gridpoint std of the
calibration year, as the denominator for standardized variable-space ACI. It
spans **0.46–23.44 K** — a 50× range, which is exactly why a single unstandardized
`eta` would be hopeless in the tropics.

**GPU gate — passed.**

| item | value |
|---|---|
| t2m | surface index **2** (`T2m`), confirmed in `era5.surface_variables` |
| det backbone | `AvgModule` over the 4 `archesweather-m-*` checkpoints |
| `num_steps` | **25** |
| `cf_guidance` / `s_churn` | **1** / **0.0** |
| `scale_input_noise` | **absent from the config → `None`**, so `sample()` applies no rescaling |
| scheduler / prediction_type | `flow` / `sample` |
| single 24 h step | **2.35 s** (median of 2, warm-up discarded, CUDA-synced) |
| 5-day rollout, 1 member | **14.72 s** |
| output range | 220.4–313.7 K, no NaNs |

**Benchmark — batching abandoned.**

| batch | s/pass | s/member | VRAM GB | members distinct | member 0 == serial |
|------:|-------:|---------:|--------:|---|---|
| 1 | 15.49 | **15.49** | 3.5 | n/a | n/a |
| 2 | 32.27 | 16.14 | 5.5 | yes | no (0.01 K) |
| 4 | 66.45 | 16.61 | 9.8 | yes | no (0.01 K) |
| 8 | 132.37 | 16.55 | 18.6 | yes | no (12.45 K) |

Batching is **not faster** — the GPU is already saturated at batch 1 (100 %
utilisation on 3.5 GB of 24 GB), so replicating the state just serialises the
same work with more memory traffic. It also fails the reproducibility check:
geoarches draws a batch's initial noise from one generator in a single stream,
so a member's noise depends on how many members share its pass. Two independent
reasons to drop it, so `SAMPLING_BATCH_SIZE = 1` and the run is serial.

**Sizing.** At 15.49 s/member over 20 members:

| grid | inits | members | wall-clock h | disk GB |
|---|---:|---:|---:|---:|
| 2020 daily | 725 | 14 500 | 62.4 | 8.4 |
| 2020 every 2d | 543 | 10 860 | 46.7 | 6.3 |
| 2020 every 3d | 482 | 9 640 | 41.5 | 5.6 |
| **2020 every 4d** | **452** | **9 040** | **38.9** | **5.3** |

Sizing rule (≤ 40 h, never thin 2021) selects **2020 every 4 days, 2021 daily**.
Thinning costs warm-up samples, not evaluation samples; 92 calibration inits
still give the controller ~18× tau of settling before the reported window opens.

**Launched.** tmux session `gen`, chronological (2020 first). Measured
**314 s/init**, ETA **39.2 h** → finishes ~Sat 00:30. Files are 6.8 MB
compressed (11.6 MB raw), so the archive lands at **~3.1 GB**, well under the
5.3 GB projection.

Restarted once, three minutes in, to fix `lead`'s `units: days` attribute —
xarray decodes a timedelta-like `units` into `timedelta64` on open, so a later
`ds.sel(lead=5)` would have raised. Cheap to fix now, a Saturday landmine
otherwise. Seeds are date-derived, so the two discarded files regenerate
bit-identically.

**Decided.** Init window, target variable, adaptation space, threshold method
and init grid — all recorded in RUNBOOK "Decisions" with the rule that
triggered each.

**Next (Fri).** Implement `aci.py` against `tests/test_aci_synthetic.py`; wire
`03_verify.py` to the real archive; decide what a missing init means to a
sequential controller.
