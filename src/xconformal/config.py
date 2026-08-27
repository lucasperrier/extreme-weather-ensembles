"""Single source of truth for paths and experiment constants.

Nothing else in this repo may hardcode a path or a scientific constant. If you
need a new one, add it here.

Paths come from the environment variables exported by ``/workspace/env.sh``.
Each has a fallback default; :func:`validate_paths` raises if a required
directory is actually missing on disk, so scripts fail loudly at startup
instead of silently writing to a wrong place.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def env_path(name: str, default: str | None = None) -> Path:
    """Return ``$name`` as a Path. Raise KeyError if unset and no default."""
    value = os.environ.get(name)
    if value is None:
        if default is None:
            raise KeyError(
                f"Required environment variable {name} is not set. "
                f"Run `source /workspace/env.sh` first."
            )
        value = default
    return Path(value)


DATA_ROOT = env_path("DATA_ROOT", "/workspace/data")
MODEL_ROOT = env_path("MODEL_ROOT", "/workspace/modelstore")
EVAL_ROOT = env_path("EVAL_ROOT", "/workspace/evalstore")
PROJECT_ROOT = env_path("PROJECT_ROOT", "/workspace/extreme-weather-ensembles")

# ERA5 truth: 6-hourly, 240x121, 13 pressure levels, one file per (year, hour).
ERA5_DIR = DATA_ROOT / "era5_240"
ERA5_FULL_DIR = ERA5_DIR / "full"
ERA5_CLIM_PATH = ERA5_DIR / "era5_240_clim.nc"


def era5_file(year: int, hour: int) -> Path:
    """Path to the ERA5 truth file for one (year, hour) slice."""
    return ERA5_FULL_DIR / f"era5_240_{year}_{hour}h.nc"


# Generative ensemble checkpoint (ArchesWeatherGen). The deterministic
# archesweather-m-* checkpoints are pulled in by its own hydra config; we only
# ever name the generative one.
GEN_MODEL_NAME = "archesweathergen"
GEN_MODEL_DIR = MODEL_ROOT / GEN_MODEL_NAME

# Our own derived products.
FORECAST_DIR = DATA_ROOT / "forecasts" / GEN_MODEL_NAME
THRESHOLD_PATH = DATA_ROOT / "derived" / "clim_p95.nc"
COVERAGE_TABLE_PATH = EVAL_ROOT / "coverage_by_bin.csv"
MARGINAL_TABLE_PATH = EVAL_ROOT / "coverage_marginal.csv"
FIGURE_PATH = EVAL_ROOT / "fig_coverage_by_bin.pdf"
BENCHMARK_PATH = EVAL_ROOT / "benchmark.csv"


def forecast_file(init: np.datetime64) -> Path:
    """One output file per init date: ``<FORECAST_DIR>/YYYYMMDDTHH.nc``."""
    stamp = np.datetime64(init, "h").item().strftime("%Y%m%dT%H")
    return FORECAST_DIR / f"{stamp}.nc"


def validate_paths(*, require_model: bool = False, require_clim: bool = False) -> None:
    """Raise FileNotFoundError for any input this run depends on but lacks.

    Output roots are created; input roots are only checked.
    """
    for label, path in [
        ("DATA_ROOT", DATA_ROOT),
        ("ERA5_FULL_DIR", ERA5_FULL_DIR),
    ]:
        if not path.is_dir():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if require_model and not GEN_MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Generative checkpoint missing: {GEN_MODEL_DIR}\n"
            f"Fetch with: python -m geoarches.download.dl_aw_models "
            f"--output-directory {MODEL_ROOT}"
        )
    if require_clim and not ERA5_CLIM_PATH.is_file():
        raise FileNotFoundError(
            f"Climatology missing: {ERA5_CLIM_PATH}\n"
            f"Fetch with: python -m geoarches.download.dl_era "
            f"--folder {ERA5_FULL_DIR} --clim --years"
        )
    for path in (EVAL_ROOT, FORECAST_DIR, THRESHOLD_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Variables
# --------------------------------------------------------------------------

# Primary target variable. geoarches names surface variables in CF style in the
# ERA5 netcdfs ("2m_temperature") and with short names inside the model's
# TensorDict output ("T2m"); see geoarches.dataloaders.era5.surface_variables.
TARGET_VAR = "2m_temperature"
TARGET_VAR_SHORT = "T2m"

# Fallback if 2m temperature is absent from the model output.
FALLBACK_VAR = "temperature"
FALLBACK_VAR_SHORT = "T850"
FALLBACK_LEVEL = 850

# TODO(lucas): [Thu] Resolve the variable name for real. Run
# `python scripts/00_inspect_data.py` and check (a) which surface variables the
# archesweathergen output actually carries and (b) what the climatology file
# calls them. If 2m_temperature is absent from model output, switch to
# FALLBACK_VAR/T850 here and nowhere else. Do not spread the fallback logic
# into thresholds.py or 01_generate.py.

# --------------------------------------------------------------------------
# Experiment constants (fixed a priori -- do not tune these on results)
# --------------------------------------------------------------------------

ALPHA = 0.1  # miscoverage target; nominal interval is 1 - ALPHA = 90%
TARGET_COVERAGE = 1.0 - ALPHA
ETA = 0.01  # ACI learning rate

# The ensemble's own prediction interval, before any conformal correction.
LOWER_QUANTILE = 0.05
UPPER_QUANTILE = 0.95

# Forecast configuration.
LEAD_DAYS = 5
LEAD_TIME_HOURS = 24  # model step size; 5 autoregressive iterations to reach day 5
ROLLOUT_ITERATIONS = LEAD_DAYS * 24 // LEAD_TIME_HOURS
N_MEMBERS = 20

# Initialisations: daily 0Z over 2020-2021. NOT 2023 -- the upstream WeatherBench2
# ERA5 source used here ends in 2021.
INIT_HOUR = 0
INIT_START = np.datetime64("2020-01-01T00")
INIT_END = np.datetime64("2021-12-31T00")  # inclusive

# ACI feedback delay, in units of init steps. Inits are daily and the lead is
# LEAD_DAYS, so the outcome of the forecast issued at t is only observable at
# t + LEAD_DAYS.
TAU = LEAD_DAYS

# Burn-in: the first ~2 months of 2020 are excluded from the reported coverage
# but ARE fed to the ACI update, so c_t has settled before evaluation starts.
EVAL_START = np.datetime64("2020-03-01T00")

# --------------------------------------------------------------------------
# Extremes and binning
# --------------------------------------------------------------------------

# Climatological exceedance threshold: per-gridpoint, per-dayofyear 95th
# percentile of the target variable, from the WB2 1990-2019 climatology.
CLIM_PERCENTILE = 95.0

# Bins on p_t = fraction of ensemble members above the climatological threshold.
# Fixed a priori. Half-open [lo, hi) except the last bin, which is closed.
P_BIN_EDGES = (0.0, 0.1, 0.5, 0.9, 1.0)


def p_bin_labels(edges: tuple[float, ...] = P_BIN_EDGES) -> list[str]:
    """Human-readable labels for the p_t bins, e.g. '[0.0, 0.1)'."""
    labels = [f"[{lo:g}, {hi:g})" for lo, hi in zip(edges[:-1], edges[1:])]
    labels[-1] = f"[{edges[-2]:g}, {edges[-1]:g}]"
    return labels


# --------------------------------------------------------------------------
# ACI adaptation space
# --------------------------------------------------------------------------

# UNDECIDED. Both are implemented behind one interface in aci.py.
#   "variable"  -- pad the interval endpoints by +/- c, in Kelvin.
#   "quantile"  -- shift the effective quantile level to alpha - c and re-read
#                  the empirical ensemble quantiles at that level.
#
# TODO(lucas): [Fri] Decide the adaptation space. Run 03_verify.py both ways
# (--space variable / --space quantile) and pick on the burn-in period only,
# never on the reported evaluation window. Then set this default and say in the
# paper which one was used and that the choice was made pre-evaluation.
ACI_SPACE = "variable"
