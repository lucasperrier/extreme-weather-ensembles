# RUNBOOK

Operational memory for this pod. Read **Session start** first thing every session.
Add to **Gotchas** the moment something costs you more than ten minutes.

---

## Facts

**Hardware / storage**
- `/workspace` is a persistent MooseFS network volume. The container disk is
  **20 GB and ephemeral**. Nothing — data, caches, checkpoints, pip wheels —
  may be written outside `/workspace`.
- `source /workspace/env.sh` activates the venv (python 3.11) and exports
  `DATA_ROOT`, `MODEL_ROOT`, `EVAL_ROOT`, `PROJECT_ROOT`, `HF_HOME`,
  `PIP_CACHE_DIR`. Do this before anything else.

**Data**
- ERA5 truth: `$DATA_ROOT/era5_240/full/era5_240_{year}_{hour}h.nc`, 6-hourly,
  240×121, 13 pressure levels, one file per (year, hour).
- Climatology: `$DATA_ROOT/era5_240/era5_240_clim.nc` (WeatherBench2 1990–2019,
  same grid). Its variable list is **unverified** — run `00_inspect_data.py`.
- Checkpoints: `$MODEL_ROOT/{archesweather-m-seed0,archesweather-m-seed1,
  archesweather-m-skip-seed0,archesweather-m-skip-seed1,archesweathergen}/`,
  each with `config.yaml` and `checkpoints/checkpoint.ckpt`.

**Experiment** — all constants live in `src/xconformal/config.py` and nowhere else.
- ArchesWeatherGen via the installed `geoarches` package. **Consume it, do not
  vendor or modify it.**
- 2m temperature (fallback T850), lead 5 days, ~20 members, daily 0Z inits over
  **2020–2021** (not 2023 — the upstream ERA5 source ends in 2021).
- `alpha = 0.1`, `eta = 0.01`, `tau = 5` init steps.
- Burn-in: inits before 1 March 2020 are excluded from reported coverage but
  **are** fed to the ACI update.
- `p_t` bins: `[0, 0.1, 0.5, 0.9, 1.0]`, fixed a priori.

---

## Session start

```bash
source /workspace/env.sh
cd $PROJECT_ROOT
python scripts/00_inspect_data.py     # what is actually on disk right now
pytest -q
```

`00_inspect_data.py` is the answer to every "is that file real / finished /
complete?" question. Several TODOs in the code point at it by name.

Pipeline order:

```bash
python scripts/02_benchmark.py                         # can the run fit? decide first
python scripts/01_generate.py --limit 2                 # smoke test, 2 inits
python scripts/01_generate.py                           # full archive; resumable
python scripts/03_verify.py --space variable
python scripts/04_figure.py
```

`01_generate.py` is safe to kill at any moment (Ctrl-C, preemption, OOM) and
safe to restart: it writes `NAME.nc.tmp-<pid>` and renames on completion, so a
`.nc` file is either absent or whole. Restart with `--clean-temps` to sweep
debris from a killed run.

---

## Gotchas

*(Seeded from prior pain. Add to this list rather than rediscovering.)*

- **Empty 2022 ERA5 files.** Some `era5_240_2022_*h.nc` files exist with a
  plausible size but `time: 0`. An empty time dimension **disappears silently in
  `xr.concat`** — it does not error, it just shortens your record and shifts
  every downstream index. Always check `ds.sizes["time"] > 0` before
  concatenating, and treat an empty time dimension as *missing data*, never as
  an empty-but-valid file. `00_inspect_data.py` flags these explicitly.

- **`wget` saves HTML as a checkpoint.** A download that hits a redirect, a
  login wall, or an expired URL writes the HTML error page to the `.ckpt`
  filename and exits 0. The file looks fine in `ls`. Real torch checkpoints are
  zip archives beginning with `PK\x03\x04`; an HTML page begins with `<`.
  `00_inspect_data.py` prints the magic bytes of every checkpoint. Prefer
  `python -m geoarches.download.dl_aw_models --output-directory $MODEL_ROOT`,
  which goes through `huggingface_hub` and validates.

- **`df` lies on MooseFS.** `df -h /workspace` reports the whole cluster
  (~1 PB), not our quota or our usage, so it can never tell you whether we are
  near a limit. Use `du -sh <dir>` to learn what we actually occupy.
  `00_inspect_data.py` prints `du`-style totals for each root. Conversely `df -h /`
  *is* meaningful — that is the 20 GB ephemeral container disk, and it filling
  up is a real failure mode (usually a cache that escaped `/workspace`).

- **`tmux` must be reinstalled after every pod restart.** It is present now
  (`/usr/bin/tmux`) but lives on the ephemeral container disk, not on
  `/workspace`, so it disappears with the pod. `apt-get install -y tmux` is a
  per-pod chore. Because `01_generate.py` is resumable, losing a session costs
  at most one init.

- **The GPU is an RTX 4090 (24 GB), not an A100.** Sizing assumptions written
  for an 80 GB A100 do not transfer: VRAM caps the member batch size, and
  per-member timings must be re-measured rather than scaled. Measured
  2026-08-27: 2.35 s per 24 h denoising step, 14.7 s per 5-day member, serial.

- **`load_module` resolves the deterministic backbones relative to CWD, and
  fails SILENTLY.** The archesweathergen config lists its four deterministic
  models as relative paths (`modelstore/archesweather-m-seed0`, ...). If CWD is
  wrong they are not restored, no exception is raised, and sampling proceeds on
  uninitialised backbones producing plausible-looking garbage. `generate.py`
  chdirs to the repo root (which carries a `modelstore` symlink to
  `$MODEL_ROOT`) and then asserts the right NUMBER of backbones is live. Never
  remove that assertion.

- **`det_model` vs `det_models`.** This config instantiates `DiffusionModule`,
  whose `.det_model` is a single `AvgModule` wrapping the four backbones in
  `.core`. `EnsembleDiffusionModule` instead exposes `.det_models`, a
  ModuleList. Checking the wrong attribute reports "0 models loaded" on a
  perfectly healthy module.

- **`DiffusionModule.sample()` takes no `member` kwarg.** It has
  `**kwargs` that are forwarded to the scheduler, so passing `member=` there
  does not error at the call site -- it fails deep inside the scheduler step.
  `member` belongs to `sample_rollout`, which composes
  `seed = member + 1000*step + batch_nb*1e6` itself.

- **geoarches' collate leaves TensorDict `batch_size` empty.**
  `_custom_collate_fn` builds each TensorDict without a batch size, so it comes
  back as `torch.Size([])` even though the tensors inside have a leading
  dimension. The backbone reads `state.shape[0]` and raises
  `IndexError: tuple index out of range`. Stamp `td.batch_size = [n]` after
  collating.

- **`dl_era` downloads every variable and every pressure level** -- about
  4.6 GB per (year, hour) file. Using it to obtain one surface variable is
  enormously wasteful: 2011-2019 at four synoptic hours would have been 165 GB
  to produce a 42 MB threshold field. Stream the slice you need straight from
  the WeatherBench2 zarr instead and persist only the derived product. Size a
  download by the answer, not by which script is convenient.

- **`era5-quantiles-2016_2022.nc` is pooled in time.** Its dims are
  `(quantile, lat, lon)` with no day-of-year, and its levels are
  `[1e-4, 1e-3, 0.01, 0.99, 0.999, 0.9999]` -- there is no 0.95. It cannot be
  used as an extreme threshold. See Decisions.

- **The WeatherBench2 zarr stores `(longitude, latitude)`**, the model output
  and everything in `xconformal` use `(latitude, longitude)`. The loaders in
  `thresholds.py` transpose to the canonical order on the way out.

- **`pytest` is not installed by default.** `pip install pytest` — it is a
  dev-only dependency and deliberately not in `pyproject.toml`'s runtime deps.

- **`geoarches.lightning_modules.load_module` resolves a bare model name
  against a literal `modelstore/` directory relative to the current working
  directory.** From `$PROJECT_ROOT` that is the wrong place. Always pass the
  absolute `config.GEN_MODEL_DIR`.

- **ERA5's `land_sea_mask` is stored `(longitude, latitude)`.** Same
  convention mismatch as the WeatherBench2 zarr, but inside the ERA5 netcdfs we
  read for truth -- and unlike the 3-D fields, `t2m` is explicitly transposed by
  `load_truth` while a naively-read mask is not. `03_verify.load_land_mask`
  transposes to the canonical order. A silently transposed mask would swap land
  for ocean over most of the grid and still produce two plausible-looking
  columns.

- **A converged warm start does NOT mean a converged evaluation year.**
  `aci.warm_start` cycles the calibration year to *the calibration year's*
  equilibrium. Measured 2026-08-29: the per-gridpoint equilibrium padding is
  0.1307 on 2020 against 0.1397 on 2021 (area-weighted), and the two fields
  correlate only 0.75 across the grid. The controller therefore opens the
  evaluation year ~0.009 low in c and spends ~83 inits (about 2.8 months)
  climbing, which costs January 0.008 of coverage and February 0.006. This is
  structural -- no amount of calibration-year data can remove it, because the
  calibration year does not know what the verification year needs. Do not read
  a low January as a warm-start bug; check it against a frozen-field control
  first (controller off, c pinned at the handed-over field), which isolates the
  data from the dynamics.

- **`warm_start`'s `tol` is a relative change in the pass-mean of c, and the
  approach is monotone and geometric**, so stopping at `tol = 0.01` stops about
  1% short of the true fixed point (+0.12330 against +0.12544 at 60 passes).
  Immaterial to the reported coverages, but do not describe the shipped default
  as "the fixed point" in the paper.

- **`Era5Forecast` silently overrides your time bounds** when constructed with
  `domain="val"` or `"test"` — it re-selects a single hard-coded year. Use an
  explicit `filename_filter` plus `set_timestamp_bounds` instead.

---

## Decisions

Record every scientific choice here with its date and its reason, so the paper's
methods section can be written from this file rather than from memory.

| Date | Decision | Reason |
|------|----------|--------|
| 2026-08-27 | **Year plan locked.** `INIT_START = 2020-01-02`, `INIT_END = 2021-12-26`. 2020 = pure calibration (ACI warm-start, never reported); 2021 = verification. | Start is 2020-01-02 because the model needs `prev_state` at init−24 h and ERA5 begins 2020-01-01T00. End is 2021-12-26 because that is the last init whose +5 day valid time has truth: the WB2 1.5° archive ends 2021-12-31T18 and there is no 2022/2023. |
| 2026-08-27 | **Target variable: 2m temperature.** T850 fallback not needed. | Confirmed as surface index **2** (`T2m`) in `geoarches.dataloaders.era5.surface_variables`, and present in ERA5 and in the model output. |
| 2026-08-27, **ratified 08-28** | **ACI adaptation space: standardized variable space.** `lower = q05 − c·s`, `upper = q95 + c·s`, with `s` = per-gridpoint std of the **2020** 0Z t2m series (`t2m_std_2020_0z.nc`). Quantile space stays behind `--space quantile` as the appendix ablation; `AciResult.saturated_frac` is its required diagnostic. | **The decisive reason is the guarantee, not eta transfer.** With M = 20 members, quantile-space adaptation cannot widen an interval past the ensemble min/max: once `alpha − c` reaches 0 the interval is `[min, max]` and further correction does nothing. On underdispersed extremes that breaks convergence outright, unless one emits degenerate infinite intervals. Variable space is unbounded and is therefore the guarantee-preserving choice at small M. (Secondarily, `s` spans 0.46–23.44 K, so an unstandardized `eta` would also be ~50× too slow in the tropics; `s` uses the calibration year only, so the verification year never informs the scaling.) |
| 2026-08-27 | **Threshold: per-gridpoint, per-day-of-year 95th percentile of t2m, 1990–2019, 0Z only, ±15 day window**, streamed from the WB2 zarr to `$DATA_ROOT/thresholds/t2m_p95_clim_1990-2019_0z.nc` (42.5 MB, 907–930 samples/group). | Valid times are all 0Z, so the threshold must condition on valid hour: t2m has a large diurnal cycle, and pooling synoptic hours samples a *mixture*, not a bigger sample of the same distribution. Sample size comes from the day window × years, never from mixing hours. |
| 2026-08-27 | **Never use a pooled-in-time quantile as the extreme threshold.** This rules out `era5-quantiles-2016_2022.nc` outright. | A quantile taken across all seasons at once is effectively a summer threshold that no winter day can reach, so every "extreme" it flags is a July extreme and the high-`p_t` bins would be pure seasonal aliasing rather than extreme weather. |
| 2026-08-27 | Store per-member t2m at leads 1–5, float32, one file per init | 11.6 MB/init. Keeps `p_t`, other quantile levels, quantile-space ACI and lead-dependence recomputable without regenerating. |

| 2026-08-27, **ratified 08-28** | **Init schedule: 2020 every 4th day, 2021 daily.** | Sizing rule (≤ 40 h): daily 62.4 h, every 2d 46.7 h, every 3d 41.5 h, every 4d 38.9 h. Thinning the calibration year costs warm-up samples, not evaluation samples. |
| 2026-08-28 | **ACI delay bookkeeping is by DATETIME, never by array index.** The controller holds a queue of `(verification_time, err_field)` and, at each init, applies every entry whose verification time has passed. `tau` is 5 **days**. | The archive is thinned in 2020 and daily in 2021, so the cadence changes mid-stream. A step-counting implementation would silently apply 20 days of delay in the thinned segment and 5 in the dense one, and nothing in the output would look wrong. With the queue, a gap in the init sequence is **just a longer wait** — the 2020→2021 boundary needs no special case, and correctness falls out of the queue rather than being asserted. |
| 2026-08-28 | **`err = 1` for an empty, crossed, or non-finite interval** (Asch's convention). | `covered()` is `lower <= y <= upper`, which is already False whenever `lower > upper` or anything is NaN, so the convention falls out with no special case — and a corrupt field counts as a miss rather than silently passing. |

| 2026-08-28 | **`c_0` for the evaluation year is obtained by cycling the calibration year (2020) to equilibrium; the evaluation year never informs `c`.** Passes carry `c` but flush-and-clear the in-flight queue at each boundary, run over whole years, and stop when the pass-mean of `c` moves < 1%. **DISCLOSE IN §3 OF THE PAPER.** | ACI relaxes with a time constant of `1/(eta · dcoverage/dc)` ≈ **80 inits** at `eta = 0.01`. A single 92-init calibration pass reaches only ~68% of the equilibrium padding `c*`, so the evaluation year would open with the controller still climbing and its coverage biased low for months. Cycling removes the transient without touching `eta` and without leaking evaluation data into `c`. Whole-year passes matter: the final pass ends on late-December conditions, the correct seasonal phase for entering January. The queue must be cleared between passes or December verification times would all come due at the next pass's first January init, in the wrong order. |
| 2026-08-29 | **Report all 12 months of the evaluation year; do not window out the Jan-Feb residual.** `eta` stays 0.01. The residual is reframed as a measurement: the controller's adaptation lag under a real year-to-year shift in required padding. | Four controls (`scripts/09_diagnose_transient.py`) rule out warm-start convergence, the data, and the handed-over field, and locate the cause in the calibration year's equilibrium padding being +0.00905 below the verification year's, with the two fields correlating only 0.747 across the grid. Windowing to Mar-Dec would be a post-hoc window chosen after seeing the result. Disclosing costs one paragraph; windowing costs credibility. |
| 2026-08-29 | **Quantile space is reported as a failed ablation, not as an alternative.** | On the full archive it saturates on **95.61%** of (init, gridpoint) pairs, stalls at 0.8444 marginal coverage, never converges, and lets c run away to +16.28. Predicted on synthetic data 2026-08-27; now measured. |
| 2026-08-28 | **The model grid is not the ERA5 grid.** `model = roll(flip(era5, lat), 120, lon)` — latitude descending, longitude on the −180…180 convention. `xconformal.grid` converts, and `grid.check_alignment` runs on every archive load. | Found by round-tripping the model's own input state, which is ERA5 at the init time, through denormalize+extract: it came back 12.7 K RMSE with min and max matching exactly, so a permutation. Every symptom was a *plausible* number — day-5 RMSE 12.9 K, mean `p_t` 0.35, raw coverage 0.11 — which is why the check is now automatic rather than trusted. |

**Open — must be closed before the paper:**

- [x] CHECK 2 resolved 2026-08-29: report all 12 months, disclose the residual,
      reframe it as a measurement of the adaptation lag. See Decisions.
- [x] Figure-form sign-off, 2026-08-29. Paired dots + target rule, per-bin counts
      on the axes, `DEFAULT_YLIM = (0.63, 0.95)` in `04_figure.py`.
- [ ] Nothing blocking. **The repo is frozen for the deadline** — no code changes
      unless a bug is found; improvements go to "post-deadline" below.

---

## Post-deadline

Improvements noticed while the deadline was live and deliberately NOT made.

- `01_generate.py` writes integer index coordinates rather than
  `grid.model_latitudes()` / `model_longitudes()`, so the grid convention lives
  in a docstring instead of on disk. The archive is correct and the read side
  fixes it; make the next generation run write real coordinates.
- `warm_start`'s convergence test should be on the fixed-point residual rather
  than on the relative change in pass-mean c between consecutive passes. The
  current rule is monotone-approach-blind and stops ~1% short.
- `03_verify.py` recomputes the raw quantiles in several places (`raw_interval`,
  the `c*` bisection, the width table). One cached pair would be cheaper and
  would remove any chance of the three drifting apart.
- The `--space quantile` appendix ablation has not been rerun on the full
  archive; only the partial-archive number from 2026-08-28 exists.
