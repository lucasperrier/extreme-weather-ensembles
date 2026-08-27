# extreme-weather-ensembles

Adaptive conformal inference gives AI weather ensembles the *marginal* coverage
they lack. This repo asks what it leaves behind: we bin verification days by the
ensemble's own forecast exceedance probability `p_t` — the fraction of members
above the climatological 95th percentile — and measure coverage within each bin.
Marginal ACI hits its target on average while missing badly on the days that
matter.

- `src/xconformal/` — the reusable core. `config.py` holds every path and
  constant; nothing else hardcodes either.
- `scripts/` — numbered pipeline, run in order. See `RUNBOOK.md`.
- `tests/` — `test_aci_synthetic.py` is the contract `aci.py` is built against.

```bash
source /workspace/env.sh
cd $PROJECT_ROOT
python scripts/00_inspect_data.py
pytest -q
```

Model weights come from the installed `geoarches` package (ArchesWeatherGen);
we consume it, we do not vendor or modify it.
