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

# Upstream WeatherBench2 stores. We stream slices from these; we never mirror
# them. dl_era pulls every variable and level (~4.6 GB per year-hour), so it is
# the wrong tool for deriving a single-variable field -- see build_thresholds.
WB2_ERA5_ZARR = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"
)

# Our own derived products.
ENSEMBLE_DIR = DATA_ROOT / "ensembles"
FORECAST_DIR = ENSEMBLE_DIR  # alias; scripts/03 and /04 use FORECAST_DIR
THRESHOLD_DIR = DATA_ROOT / "thresholds"
THRESHOLD_PATH = THRESHOLD_DIR / "t2m_p95_clim_1990-2019_0z.nc"
ACI_SCALE_PATH = THRESHOLD_DIR / "t2m_std_2020_0z.nc"
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
    for path in (EVAL_ROOT, ENSEMBLE_DIR, THRESHOLD_DIR):
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

# RESOLVED 2026-08-27: 2m_temperature is present in ERA5, in the
# era5-quantiles stats file, and in the model's surface output at index
# TARGET_SURFACE_INDEX below. The T850 fallback is not needed and is kept only
# as dead config in case the variable choice is revisited.
TARGET_SURFACE_INDEX = 2  # geoarches.dataloaders.era5.surface_variables order

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

# Initialisations: daily 0Z. LOCKED 2026-08-27.
#   2020 = pure calibration: fed to the ACI update, never reported.
#   2021 = verification: the evaluation window in the paper.
# INIT_START is 2020-01-02 because the model needs the previous state (init - 24h)
# and ERA5 begins at 2020-01-01T00. INIT_END is 2021-12-26 because that is the
# last init whose +5 day valid time (2021-12-31T00) has ERA5 truth: the
# WeatherBench2 1.5 deg archive ends at 2021-12-31T18 and there is no 2022/2023.
INIT_HOUR = 0
INIT_START = np.datetime64("2020-01-02T00")
INIT_END = np.datetime64("2021-12-26T00")  # inclusive
CALIBRATION_YEAR = 2020
VERIFICATION_YEAR = 2021

# Members sampled per denoising pass. Set from scripts/02_benchmark.py.
# Pin it: geoarches draws a batch's initial noise from one generator in a
# single stream, so a member's noise depends on how many members shared its
# pass. Changing this changes the ensemble, and a resumed run must use the
# same value to stay bit-identical.
SAMPLING_BATCH_SIZE = 1

# Init spacing in days, per year. Set to >1 to thin an under-budget run.
# LOCKED: 2021 is never thinned -- it is the reported window.
INIT_STRIDE_CALIBRATION = 1
INIT_STRIDE_VERIFICATION = 1

# Leads archived per init, in days. We verify at LEAD_DAYS but store all of
# 1..5 so lead-dependence can be shown without regenerating.
ARCHIVED_LEADS = (1, 2, 3, 4, 5)

# ACI feedback delay, in units of init steps. Inits are daily and the lead is
# LEAD_DAYS, so the outcome of the forecast issued at t is only observable at
# t + LEAD_DAYS.
TAU = LEAD_DAYS

# All of 2020 is calibration: fed to the ACI update, excluded from every
# reported number. Evaluation starts with the first 2021 init.
EVAL_START = np.datetime64("2021-01-01T00")

# --------------------------------------------------------------------------
# Extremes and binning
# --------------------------------------------------------------------------

# Climatological exceedance threshold: per-gridpoint, per-dayofyear 95th
# percentile of the target variable, computed from ERA5 1990-2019 at the
# verification hour only.
CLIM_PERCENTILE = 95.0
CLIM_YEARS = (1990, 2019)  # inclusive
CLIM_HOUR = 0  # must equal the valid hour being verified: t2m has a strong
               # diurnal cycle, so pooling synoptic hours would sample a
               # mixture distribution, not a bigger sample of the same one.
CLIM_WINDOW_DAYS = 15  # +/- this many days around each dayofyear.
                       # 30 years x 31 days = 930 samples per gridpoint per
                       # dayofyear. Sample size comes from the day window and
                       # the years, never from mixing hours.

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

# LOCKED 2026-08-27: variable space, standardized.
#
# The correction c is dimensionless and is applied as
#     lower = q05 - c * s,   upper = q95 + c * s
# where s is the per-gridpoint standard deviation of the 2020 0Z t2m series
# (cached at ACI_SCALE_PATH, built from the calibration year only so the
# verification year never informs the scaling). Standardizing means one eta
# works everywhere: without it a single learning rate is far too slow in the
# tropics, where the day-to-day t2m spread is a fraction of the mid-latitude
# spread, and c would still be climbing when the record ends.
#
# Quantile space stays available behind --space quantile for the appendix. It
# saturates once alpha - c leaves [0, 1] -- the interval is then already
# [min, max] and cannot widen -- and AciResult.saturated_frac must report it.
ACI_SPACE = "variable"
ACI_STANDARDIZE = True
