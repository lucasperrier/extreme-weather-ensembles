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

- **`tmux` is not in the image.** Long runs die with the SSH session. Either
  `apt-get install -y tmux` at the start of the session (it goes on the
  ephemeral disk and is gone after a restart, so this is a per-pod chore), or
  use `nohup python scripts/01_generate.py > gen.log 2>&1 &` and tail the log.
  Because `01_generate.py` is resumable, losing the session costs at most one
  init.

- **`pytest` is not installed by default.** `pip install pytest` — it is a
  dev-only dependency and deliberately not in `pyproject.toml`'s runtime deps.

- **`geoarches.lightning_modules.load_module` resolves a bare model name
  against a literal `modelstore/` directory relative to the current working
  directory.** From `$PROJECT_ROOT` that is the wrong place. Always pass the
  absolute `config.GEN_MODEL_DIR`.

- **`Era5Forecast` silently overrides your time bounds** when constructed with
  `domain="val"` or `"test"` — it re-selects a single hard-coded year. Use an
  explicit `filename_filter` plus `set_timestamp_bounds` instead.

---

## Decisions

Record every scientific choice here with its date and its reason, so the paper's
methods section can be written from this file rather than from memory.

| Date | Decision | Reason |
|------|----------|--------|
| — | Inits 2020–2021, not 2023 | upstream WeatherBench2 ERA5 source ends 2021 |
| — | Store per-member values, not pre-reduced quantiles | 1.7 GB total; keeps `p_t`, other quantile levels and quantile-space ACI recomputable without regenerating |

**Open — must be closed before the paper:**

- [ ] **ACI adaptation space** (`config.ACI_SPACE`, due Fri). Variable-space vs
      quantile-space. They are not equivalent: quantile-space saturates once the
      interval reaches the ensemble min/max and cannot widen further, which is
      likely to bite exactly in the high-`p_t` bins. Decide on the **burn-in
      window only**, never on the reported evaluation window, and say in the
      paper that the choice was made pre-evaluation.
- [ ] **Target variable** (`config.TARGET_VAR`, due Thu). 2m temperature if the
      model output carries it, else T850. Run `00_inspect_data.py`.
- [ ] **Threshold source** (`thresholds.build_thresholds`, due Thu). Read the
      95th percentile from `era5_240_clim.nc` if it has one; otherwise compute
      it from the ERA5 history. The climatology may hold means only.
- [ ] **Day-of-year smoothing for the threshold** (due Fri). None, or a ±7-day
      window so each day-of-year group has enough samples.
- [ ] **Missing inits in the ACI stream** (due Fri). A gap is not the same as a
      shorter sequence — decide whether a missing init skips the update or
      carries the delay forward.
