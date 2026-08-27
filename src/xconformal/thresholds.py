"""Climatological exceedance thresholds: per-gridpoint, per-dayofyear 95th percentile.

These define what counts as "extreme" and therefore drive the exceedance
probability p_t that the binning is built on. They are climatological (from the
1990-2019 WeatherBench2 climatology), not forecast-dependent, so they are
computed once and cached to config.THRESHOLD_PATH.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from . import config

__all__ = [
    "resolve_target_variable",
    "load_climatology",
    "climatological_percentile",
    "build_thresholds",
    "load_thresholds",
    "threshold_for_times",
]


def resolve_target_variable(ds: xr.Dataset) -> tuple[str, dict]:
    """Pick the target variable name and any dimension selection it needs.

    Returns ``(name, selector)`` where ``selector`` is passed to ``ds.sel``,
    e.g. ``("2m_temperature", {})`` or ``("temperature", {"level": 850})``.

    Prefers config.TARGET_VAR; falls back to config.FALLBACK_VAR at
    config.FALLBACK_LEVEL if the target is absent. Raises KeyError if neither
    is present, naming what the dataset does carry.

    TODO(lucas): [Thu] This function is the ONLY place the t2m-vs-T850 fallback
    is allowed to live. Confirm which branch fires by running
    `python scripts/00_inspect_data.py`, which prints the data_vars of both the
    climatology and a forecast file.
    """
    raise NotImplementedError(
        "in: xr.Dataset; out: (variable_name, sel_kwargs) resolving TARGET_VAR or FALLBACK_VAR"
    )


def load_climatology(path: Path = config.ERA5_CLIM_PATH) -> xr.Dataset:
    """Open the WB2 1990-2019 climatology file.

    in: path to era5_240_clim.nc; out: xr.Dataset (lazy, not loaded into memory).

    TODO(lucas): [Thu] The climatology's variable list is UNVERIFIED and it may
    well contain only means, not percentiles. `scripts/00_inspect_data.py`
    prints data_vars and coords. If there is no 95th-percentile field, use
    build_thresholds(source="era5") instead of source="clim" and say so in the
    paper's methods.
    """
    raise NotImplementedError("in: path to clim netcdf; out: xr.Dataset")


def climatological_percentile(
    da: xr.DataArray, percentile: float = config.CLIM_PERCENTILE
) -> xr.DataArray:
    """Per-gridpoint, per-dayofyear percentile of a (time, lat, lon) field.

    in: DataArray with a 'time' dim; out: DataArray (dayofyear, lat, lon).

    Group by ``time.dt.dayofyear`` and take the percentile within each group.
    Note this gives 366 groups; day 366 is thin (leap years only). Consider a
    +/- 7 day window around each dayofyear so each group has enough samples --
    that is a scientific choice, so it is yours to make.

    TODO(lucas): [Fri] Decide the dayofyear smoothing window (none vs +/-7d) and
    record the decision in RUNBOOK.md under Decisions.
    """
    raise NotImplementedError(
        "in: DataArray (time, lat, lon), percentile; out: DataArray (dayofyear, lat, lon)"
    )


def build_thresholds(
    source: str = "clim", out_path: Path = config.THRESHOLD_PATH
) -> xr.DataArray:
    """Compute the threshold field and cache it to ``out_path``.

    in: source in {"clim", "era5"}; out: DataArray (dayofyear, lat, lon), also
    written to out_path via an atomic temp-file rename.

    "clim" reads the precomputed percentile from the climatology file if it has
    one; "era5" computes it from the ERA5 history in config.ERA5_FULL_DIR.
    """
    raise NotImplementedError(
        'in: source in {"clim","era5"}, out_path; out: DataArray (dayofyear, lat, lon), cached'
    )


def load_thresholds(path: Path = config.THRESHOLD_PATH) -> xr.DataArray:
    """Load the cached threshold field, or raise telling the caller to build it.

    in: path; out: DataArray (dayofyear, lat, lon).
    """
    raise NotImplementedError("in: path; out: DataArray (dayofyear, lat, lon)")


def threshold_for_times(thresholds: xr.DataArray, times: np.ndarray) -> xr.DataArray:
    """Broadcast a (dayofyear, lat, lon) threshold onto a list of valid times.

    in: thresholds (dayofyear, lat, lon), times (n_time,) datetime64;
    out: DataArray (time, lat, lon) selected by each time's dayofyear.
    """
    raise NotImplementedError(
        "in: thresholds (dayofyear, lat, lon), times (n_time,); out: DataArray (time, lat, lon)"
    )
