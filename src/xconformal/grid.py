"""The model grid convention, and the transform to the ERA5 one.

ArchesWeatherGen emits fields on a grid that is NOT the one the ERA5 netcdfs
use, and nothing in the tensor says so -- the arrays are plain (121, 240)
either way, the values are identical as a multiset, and the min/max match to
the last bit. Comparing them directly produces a field that looks like weather,
correlates 0.78 with the truth, and is completely wrong.

Discovered 2026-08-28 by round-tripping a known field: the model's own INPUT
state, which is ERA5 at the init time, came back with 12.7 K RMSE against ERA5.
Since a round trip through normalize/denormalize cannot change the values, the
difference had to be a permutation. Brute-forcing latitude flip x longitude
roll found an exact match at RMSE 0.000000:

    model = roll(flip(era5, axis=lat), 120, axis=lon)

So the model grid is

    latitude   90 .. -90   DESCENDING   (ERA5 files: -90 .. 90 ascending)
    longitude  180 .. 358.5, 0 .. 178.5 (ERA5 files: 0 .. 358.5)

i.e. the model uses the -180..180 longitude convention, and 120 gridpoints at
1.5 degrees is exactly the 180 degree offset.

Symptoms if this is skipped: day-5 t2m RMSE ~12.9 K instead of ~1.5 K, mean
exceedance probability 0.35 instead of 0.07, and raw 5-95 coverage 0.11 instead
of 0.70. Every one of those is a plausible-looking number, which is why
:func:`check_alignment` exists and is called on every archive load.

TODO(lucas): [Sat] The ensemble files written by the currently-running job carry
integer index coordinates, because Era5Forecast has no ``xr_datasets``
attribute and 01_generate.py's coordinate lookup silently fell through to None.
The DATA IS CORRECT -- this module fixes the read side and there is no need to
regenerate. Before any future run, make 01_generate.py write
``model_latitudes()`` / ``model_longitudes()`` so the convention is on disk
rather than in this docstring. Not done now: the job is live and must not be
touched.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "MODEL_LON_ROLL",
    "model_latitudes",
    "model_longitudes",
    "era5_latitudes",
    "era5_longitudes",
    "to_era5_grid",
    "check_alignment",
]

N_LAT, N_LON = 121, 240
RESOLUTION = 1.5
# 180 degrees / 1.5 degrees per gridpoint.
MODEL_LON_ROLL = int(180.0 / RESOLUTION)

# Any misalignment shows up as several Kelvin; a correct day-1 t2m forecast is
# well under 1 K RMSE. 3 K sits far above the signal and far below the failure.
ALIGNMENT_TOLERANCE_K = 3.0


def era5_latitudes() -> np.ndarray:
    """-90 .. 90 ascending, as stored in the ERA5 netcdfs and the WB2 zarr."""
    return np.linspace(-90.0, 90.0, N_LAT)


def era5_longitudes() -> np.ndarray:
    """0 .. 358.5 ascending."""
    return np.arange(N_LON) * RESOLUTION


def model_latitudes() -> np.ndarray:
    """90 .. -90 descending, as the model emits."""
    return era5_latitudes()[::-1]


def model_longitudes() -> np.ndarray:
    """180 .. 358.5 then 0 .. 178.5, as the model emits."""
    return np.roll(era5_longitudes(), MODEL_LON_ROLL)


def to_era5_grid(array: np.ndarray) -> np.ndarray:
    """Reorient a model-grid field onto the ERA5 grid.

    in: array (..., lat, lon) on the model grid; out: same shape, on the ERA5
    grid. Leading axes (member, lead, time) are untouched.

    Inverse of ``model = roll(flip(era5, lat), +120, lon)``.
    """
    array = np.asarray(array)
    if array.shape[-2:] != (N_LAT, N_LON):
        raise ValueError(
            f"expected a (..., {N_LAT}, {N_LON}) field, got {array.shape}"
        )
    return np.flip(np.roll(array, -MODEL_LON_ROLL, axis=-1), axis=-2)


def check_alignment(field: np.ndarray, truth: np.ndarray, label: str = "") -> float:
    """Assert a reoriented forecast actually lines up with the truth.

    in: field (lat, lon) forecast on the ERA5 grid, truth (lat, lon) ERA5;
    out: RMSE in Kelvin. Raises if it exceeds ALIGNMENT_TOLERANCE_K.

    This runs on every archive load. The grid convention is documented above
    rather than stored on disk, so it is exactly the kind of assumption that
    rots silently -- and its failure mode is a believable number, not a crash.
    """
    field, truth = np.asarray(field), np.asarray(truth)
    if field.shape != truth.shape:
        raise ValueError(f"shape mismatch: field {field.shape} vs truth {truth.shape}")
    rmse = float(np.sqrt(np.mean((field - truth) ** 2)))
    if not np.isfinite(rmse) or rmse > ALIGNMENT_TOLERANCE_K:
        raise ValueError(
            f"grid alignment check failed{' for ' + label if label else ''}: "
            f"RMSE {rmse:.2f} K exceeds {ALIGNMENT_TOLERANCE_K} K. The forecast and the "
            f"truth are on different grids. A latitude flip or a 180 degree longitude "
            f"offset gives ~12.7 K here; see xconformal.grid for the convention."
        )
    return rmse
