# extreme-weather-ensembles

How much conditional calibration does marginal conformal prediction buy an AI
weather ensemble? We generate a 20-member ArchesWeatherGen ensemble of 2 m
temperature at 5-day lead for every day of 2021, calibrate it with adaptive
conformal inference (ACI) in standardized variable space, and measure coverage
within groups defined by the ensemble's own exceedance probability `p_t` (the
fraction of members above the climatological 95th percentile). Raw intervals
cover 69.7% in the highest-warning group against a 90% target; marginal ACI
lifts it to 89.6%, at the largest width cost of any group. Independent
per-group corrections would need roughly seventeen years of daily forecasts to
converge for the rarest group.

Preprint: arXiv ID to follow. Data (ensemble archive, thresholds, verification
outputs): Zenodo DOI to follow.

## Layout

- `src/xconformal/` — ACI controller with datetime-keyed delayed updates
  (`aci.py`), thresholds, binning, coverage, grid conversion. All paths and
  constants in `config.py`.
- `scripts/` — numbered pipeline, run in order (below).
- `tests/` — `test_aci_synthetic.py` is the contract `aci.py` was built against.
- `paper/` — the two figures and `RESULTS.md`, the paper's source of truth.
- `RUNBOOK.md` — every scientific decision with its date and reason, and the
  operational gotchas. `LOG.md` — dated notebook of the runs.

## Reproduction

Python 3.11; exact versions in `requirements.lock`.

    pip install -r requirements.lock
    cp env.example.sh env.sh   # set DATA_ROOT, MODEL_ROOT, EVAL_ROOT, PROJECT_ROOT
    source env.sh
    python -m geoarches.download.dl_aw_models --output-directory $MODEL_ROOT   # ArchesWeather / ArchesWeatherGen weights
    python scripts/00_inspect_data.py     # what is on disk
    python scripts/05_build_thresholds.py # 1990–2019 day-of-year p95, streamed from WeatherBench 2
    python scripts/01_generate.py         # 452 inits; resumable; ~40 h on one RTX 4090
    python scripts/03_verify.py --space variable
    python scripts/04_figure.py

ERA5 truth at 1.5° comes from WeatherBench 2. Member seeds derive from the
initialization date; a regenerated forecast reproduces the archived one
bit for bit (`scripts/08_audit_archive.py`).

## Cite

Perrier, L. (2026). How much conditional calibration does marginal conformal
prediction buy an AI weather ensemble? Preprint.

MIT licence.
