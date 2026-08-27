"""Climatological exceedance thresholds: per-gridpoint, per-dayofyear 95th percentile.

These define what counts as "extreme" and therefore drive the exceedance
probability p_t that the binning is built on.

Two properties of this threshold are load-bearing and neither is negotiable:

1.  **It conditions on the valid hour.** 2m temperature has a large diurnal
    cycle. Pooling 0Z/6Z/12Z/18Z into one day-of-year group does not give a
    bigger sample of the same distribution, it gives a sample of a mixture, and
    the resulting "95th percentile" is contaminated by the daily maximum.
    Sample size comes from the day-of-year window and the number of years.

2.  **It is never pooled over time.** A quantile taken across all seasons at
    once is effectively a summer threshold that no winter day can reach, so
    every extreme it identifies is a July extreme. The
    `era5-quantiles-2016_2022.nc` file shipped with geoarches is exactly this
    (dims: quantile, lat, lon -- no day-of-year) and must not be used here.

The source slice is streamed from the WeatherBench2 zarr and never persisted:
30 years of one surface variable at one hour is ~1.3 GB through memory, against
~165 GB if the same years were pulled with `geoarches.download.dl_era`, which
fetches every variable and every pressure level.
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

import numpy as np
import xarray as xr

from . import config

__all__ = [
    "resolve_target_variable",
    "build_thresholds",
    "load_thresholds",
    "threshold_for_times",
    "build_aci_scale",
    "load_aci_scale",
]

DAYS_IN_YEAR = 366  # dayofyear is 1..366; day 366 exists in leap years only


def resolve_target_variable(ds: xr.Dataset) -> tuple[str, dict]:
    """Pick the target variable name and any dimension selection it needs.

    Returns ``(name, selector)`` where ``selector`` is passed to ``ds.sel``,
    e.g. ``("2m_temperature", {})`` or ``("temperature", {"level": 850})``.

    RESOLVED 2026-08-27: 2m_temperature is present in every source we touch, so
    the T850 branch has never fired. It is kept because it costs one line and
    the fallback is still the documented plan if the variable choice changes.
    """
    if config.TARGET_VAR in ds.data_vars:
        return config.TARGET_VAR, {}
    if config.FALLBACK_VAR in ds.data_vars:
        return config.FALLBACK_VAR, {"level": config.FALLBACK_LEVEL}
    raise KeyError(
        f"neither {config.TARGET_VAR!r} nor {config.FALLBACK_VAR!r} in dataset; "
        f"it carries {sorted(ds.data_vars)}"
    )


def _write_atomic(obj: xr.DataArray | xr.Dataset, path: Path) -> None:
    """Write via a temp name and rename, so a killed job leaves no half-file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        obj.to_netcdf(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _open_wb2_slice(
    years: tuple[int, int],
    hour: int,
    variable: str = config.TARGET_VAR,
) -> xr.DataArray:
    """Stream one variable at one synoptic hour from the WeatherBench2 zarr.

    in: (first_year, last_year) inclusive, hour, variable;
    out: DataArray (time, latitude, longitude), loaded into memory as float32.

    Only the requested variable and hour cross the network. Nothing is written
    to disk -- the caller persists the derived statistic, not the slice.
    """
    store = xr.open_zarr(config.WB2_ERA5_ZARR, chunks=None)
    if variable not in store.data_vars:
        raise KeyError(f"{variable!r} not in {config.WB2_ERA5_ZARR}; has {sorted(store.data_vars)}")
    da = store[variable]
    first, last = years
    da = da.sel(time=(da.time.dt.year >= first) & (da.time.dt.year <= last))
    da = da.sel(time=da.time.dt.hour == hour)
    if da.sizes.get("time", 0) == 0:
        raise ValueError(f"no {hour:02d}Z timesteps for {first}-{last} in the source zarr")
    return da.astype("float32").load()


def _window_mask(dayofyear: np.ndarray, target: int, window: int) -> np.ndarray:
    """Circular +/- window mask around a day-of-year, wrapping across new year."""
    delta = np.abs(dayofyear - target)
    return np.minimum(delta, DAYS_IN_YEAR - delta) <= window


def _windowed_percentile(da: xr.DataArray, percentile: float, window: int) -> xr.DataArray:
    """Per-gridpoint, per-dayofyear percentile over a +/- window day sample.

    in: DataArray (time, latitude, longitude), percentile in 0..100, window days;
    out: DataArray (dayofyear, latitude, longitude), float32.
    """
    doy = da["time"].dt.dayofyear.values
    values = da.values  # (time, lat, lon)
    out = np.empty((DAYS_IN_YEAR, *values.shape[1:]), dtype="float32")
    counts = np.empty(DAYS_IN_YEAR, dtype="int32")
    for i, target in enumerate(range(1, DAYS_IN_YEAR + 1)):
        mask = _window_mask(doy, target, window)
        counts[i] = int(mask.sum())
        out[i] = np.percentile(values[mask], percentile, axis=0)
    return xr.DataArray(
        out,
        dims=("dayofyear", *da.dims[1:]),
        coords={
            "dayofyear": np.arange(1, DAYS_IN_YEAR + 1, dtype="int16"),
            **{d: da[d] for d in da.dims[1:]},
        },
        name=f"t2m_p{percentile:g}",
        attrs={"n_samples_min": int(counts.min()), "n_samples_max": int(counts.max())},
    )


def build_thresholds(
    out_path: Path = config.THRESHOLD_PATH,
    years: tuple[int, int] = config.CLIM_YEARS,
    hour: int = config.CLIM_HOUR,
    percentile: float = config.CLIM_PERCENTILE,
    window: int = config.CLIM_WINDOW_DAYS,
    overwrite: bool = False,
) -> xr.DataArray:
    """Compute the threshold field from the WB2 zarr and cache it to ``out_path``.

    in: out_path and the climatology parameters (all defaulted from config);
    out: DataArray (dayofyear, latitude, longitude), also written to out_path
    with a provenance attrs block.

    Idempotent: returns the cached field if it already exists unless
    ``overwrite``.
    """
    if out_path.is_file() and not overwrite:
        print(f"threshold already present, reusing: {out_path}")
        return load_thresholds(out_path)

    print(f"streaming {config.TARGET_VAR} at {hour:02d}Z, {years[0]}-{years[1]} "
          f"from {config.WB2_ERA5_ZARR}")
    da = _open_wb2_slice(years, hour)
    print(f"  loaded {da.sizes['time']} timesteps, {da.nbytes / 1e9:.2f} GB in memory")

    print(f"computing p{percentile:g} per gridpoint per dayofyear, +/-{window} day window")
    thresholds = _windowed_percentile(da, percentile, window)
    del da  # the raw slice is never persisted and is not needed past this point

    thresholds.attrs.update({
        "long_name": f"{config.TARGET_VAR} climatological {percentile:g}th percentile",
        "units": "K",
        "source": config.WB2_ERA5_ZARR,
        "source_variable": config.TARGET_VAR,
        "years": f"{years[0]}-{years[1]}",
        "valid_hour_utc": int(hour),
        "percentile": float(percentile),
        "window_days": int(window),
        "window_note": (
            "circular +/- window_days around each dayofyear; sample is one "
            "timestep per day at valid_hour_utc only, never pooled across "
            "synoptic hours (t2m has a strong diurnal cycle)"
        ),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "created_by": "xconformal.thresholds.build_thresholds",
    })
    _write_atomic(thresholds.to_dataset(), out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB), "
          f"samples/group {thresholds.attrs['n_samples_min']}-{thresholds.attrs['n_samples_max']}")
    return thresholds


def load_thresholds(path: Path = config.THRESHOLD_PATH) -> xr.DataArray:
    """Load the cached threshold field.

    in: path; out: DataArray (dayofyear, latitude, longitude).
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"No threshold field at {path}. Build it with "
            f"`python scripts/05_build_thresholds.py`."
        )
    ds = xr.open_dataset(path)
    (name,) = list(ds.data_vars)
    return _canonical(ds[name])


def _canonical(da: xr.DataArray) -> xr.DataArray:
    """Put spatial dims in (..., latitude, longitude) order.

    The WeatherBench2 zarr stores (longitude, latitude); the model output and
    every array in this package are (latitude, longitude). The grid is 121x240
    so a stray transpose would raise on broadcast rather than corrupt silently,
    but normalising here means it never comes up.
    """
    spatial = [d for d in ("latitude", "longitude") if d in da.dims]
    other = [d for d in da.dims if d not in spatial]
    return da.transpose(*other, *spatial)


def threshold_for_times(thresholds: xr.DataArray, times: np.ndarray) -> xr.DataArray:
    """Broadcast a (dayofyear, lat, lon) threshold onto a list of valid times.

    in: thresholds (dayofyear, lat, lon), times (n_time,) datetime64;
    out: DataArray (time, lat, lon) selected by each time's dayofyear.
    """
    times = np.asarray(times)
    doy = xr.DataArray(
        np.asarray(times, dtype="datetime64[ns]").astype("datetime64[D]"),
        dims="time",
        coords={"time": times},
    ).dt.dayofyear
    return thresholds.sel(dayofyear=doy).drop_vars("dayofyear")


def build_aci_scale(
    out_path: Path = config.ACI_SCALE_PATH,
    year: int = config.CALIBRATION_YEAR,
    hour: int = config.CLIM_HOUR,
    overwrite: bool = False,
) -> xr.DataArray:
    """Per-gridpoint std of the calibration-year series, used to standardize c.

    in: out_path, year, hour; out: DataArray (latitude, longitude), cached.

    Built from the CALIBRATION year only, so the verification year never
    informs the scaling that the controller runs on.
    """
    if out_path.is_file() and not overwrite:
        print(f"ACI scale already present, reusing: {out_path}")
        return load_aci_scale(out_path)

    da = _open_wb2_slice((year, year), hour)
    scale = da.std("time").astype("float32").rename("t2m_std")
    scale.attrs.update({
        "long_name": f"{config.TARGET_VAR} std over {year} at {hour:02d}Z",
        "units": "K",
        "source": config.WB2_ERA5_ZARR,
        "year": int(year),
        "valid_hour_utc": int(hour),
        "purpose": "denominator for standardized variable-space ACI (config.ACI_SPACE)",
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    })
    _write_atomic(scale.to_dataset(), out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB), "
          f"std range {float(scale.min()):.2f}-{float(scale.max()):.2f} K")
    return scale


def load_aci_scale(path: Path = config.ACI_SCALE_PATH) -> xr.DataArray:
    """Load the cached per-gridpoint ACI scale field."""
    if not path.is_file():
        raise FileNotFoundError(
            f"No ACI scale field at {path}. Build it with "
            f"`python scripts/05_build_thresholds.py`."
        )
    ds = xr.open_dataset(path)
    (name,) = list(ds.data_vars)
    return _canonical(ds[name])
