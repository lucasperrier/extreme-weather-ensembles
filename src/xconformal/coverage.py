"""Coverage diagnostics: the miscoverage indicator, and marginal / per-bin rates.

The paper's claim lives here: marginal coverage hits its target while per-bin
coverage does not. Both come out of the same indicator array, so they cannot
drift apart through separate implementations.
"""

from __future__ import annotations

import numpy as np

from . import config

__all__ = [
    "covered",
    "miscoverage",
    "latitude_weights",
    "marginal_coverage",
    "per_bin_coverage",
    "coverage_table",
]


def covered(y_true, lower, upper) -> np.ndarray:
    """Indicator that the outcome fell inside the interval.

    in: y_true, lower, upper, all broadcastable to a common shape;
    out: bool array, True where ``lower <= y_true <= upper`` (closed interval).

    This is the complement of the ``err`` that ACI updates on; one definition
    means the verification and the controller cannot disagree about a miss.
    """
    y_true = np.asarray(y_true)
    return (y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))


def miscoverage(y_true, lower, upper) -> np.ndarray:
    """``err`` indicator: 1.0 where the outcome fell outside the interval."""
    return 1.0 - covered(y_true, lower, upper).astype(float)


def latitude_weights(latitudes: np.ndarray) -> np.ndarray:
    """cos(latitude) area weights, normalised to mean 1.

    On a regular lat-lon grid a gridbox at 75N covers about a quarter of the
    area of one at the equator. Averaging coverage unweighted therefore reports
    a number dominated by the poles, where there are far more gridpoints than
    there is planet.
    """
    weights = np.cos(np.deg2rad(np.asarray(latitudes, dtype=float)))
    weights = np.clip(weights, 0.0, None)
    return weights / weights.mean()


def marginal_coverage(is_covered: np.ndarray, weights=None) -> tuple[float, int]:
    """Overall empirical coverage, optionally area-weighted.

    in: bool array (any shape), optional weights broadcastable to it;
    out: (rate, count). rate is NaN if count == 0 -- never 0.0, which would
    read as total miscoverage. count is always the raw sample count, so that
    weighting changes the estimate but never inflates how much data it rests on.
    """
    is_covered = np.asarray(is_covered)
    count = int(is_covered.size)
    if count == 0:
        return float("nan"), 0
    if weights is None:
        return float(is_covered.mean()), count
    weights = np.broadcast_to(np.asarray(weights, dtype=float), is_covered.shape)
    total = float(weights.sum())
    if total == 0:
        return float("nan"), count
    return float((is_covered * weights).sum() / total), count


def per_bin_coverage(
    is_covered: np.ndarray, masks: list[np.ndarray], weights=None
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical coverage within each bin, optionally area-weighted.

    in: is_covered (bool, any shape), masks (list of bool arrays of that shape),
        optional weights broadcastable to is_covered;
    out: (rates (n_bins,) float, counts (n_bins,) int).

    An empty bin yields rate NaN and count 0 -- never 0.0 coverage, which would
    plot as catastrophic miscoverage. Counts sum to ``is_covered.size`` when the
    masks partition the data, and are raw sample counts regardless of weighting.
    """
    is_covered = np.asarray(is_covered)
    if weights is not None:
        weights = np.broadcast_to(np.asarray(weights, dtype=float), is_covered.shape)
    rates = np.empty(len(masks), dtype=float)
    counts = np.empty(len(masks), dtype=int)
    for i, mask in enumerate(masks):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != is_covered.shape:
            raise ValueError(
                f"mask {i} has shape {mask.shape}, expected {is_covered.shape}"
            )
        n = int(mask.sum())
        counts[i] = n
        if not n:
            rates[i] = float("nan")
        elif weights is None:
            rates[i] = float(is_covered[mask].mean())
        else:
            w = weights[mask]
            total = float(w.sum())
            rates[i] = float((is_covered[mask] * w).sum() / total) if total else float("nan")
    return rates, counts


def coverage_table(
    results: dict[str, np.ndarray],
    masks: list[np.ndarray],
    labels: list[str] | None = None,
    edges: tuple[float, ...] = config.P_BIN_EDGES,
    weights=None,
):
    """Assemble per-bin coverage for several methods into one tidy table.

    in: results mapping method name ("raw", "aci-variable") -> bool covered
        array; masks from binning.bin_masks; labels from config.p_bin_labels().
    out: pandas DataFrame with columns
        ``method, bin_label, bin_lo, bin_hi, coverage, count, target``,
        ready to write to config.COVERAGE_TABLE_PATH and read straight back by
        scripts/04_figure.py.
    """
    import pandas as pd

    labels = labels if labels is not None else config.p_bin_labels(edges)
    rows = []
    for method, is_covered in results.items():
        rates, counts = per_bin_coverage(is_covered, masks, weights=weights)
        for i, (label, rate, count) in enumerate(zip(labels, rates, counts)):
            rows.append({
                "method": method,
                "bin_label": label,
                "bin_lo": float(edges[i]),
                "bin_hi": float(edges[i + 1]),
                "coverage": rate,
                "count": int(count),
                "target": config.TARGET_COVERAGE,
            })
    return pd.DataFrame(rows)
