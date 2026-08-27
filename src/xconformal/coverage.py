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
    "marginal_coverage",
    "per_bin_coverage",
    "coverage_table",
]


def covered(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Indicator that the outcome fell inside the interval.

    in: y_true, lower, upper, all broadcastable to a common shape;
    out: bool array, True where ``lower <= y_true <= upper`` (closed interval).

    This is the complement of the ``err`` that ACI updates on; keeping one
    definition means the verification and the controller cannot disagree about
    what a miss is.
    """
    raise NotImplementedError(
        "in: y_true, lower, upper (broadcastable); out: bool array, lower <= y <= upper"
    )


def miscoverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """``err`` indicator: 1.0 where the outcome fell outside the interval."""
    raise NotImplementedError(
        "in: y_true, lower, upper (broadcastable); out: float array, 1.0 outside the interval"
    )


def marginal_coverage(is_covered: np.ndarray) -> tuple[float, int]:
    """Overall empirical coverage.

    in: bool array (any shape); out: (rate, count). rate is NaN if count == 0.
    """
    raise NotImplementedError("in: bool array; out: (coverage_rate float, n int)")


def per_bin_coverage(
    is_covered: np.ndarray, masks: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical coverage within each bin.

    in: is_covered (bool, any shape), masks (list of bool arrays of that shape);
    out: (rates (n_bins,) float, counts (n_bins,) int).

    An empty bin yields rate NaN and count 0 -- never 0.0 coverage, which would
    read as catastrophic miscoverage on the figure. Counts must sum to
    ``is_covered.size`` when the masks partition the data.
    """
    raise NotImplementedError(
        "in: is_covered bool array, masks list of bool arrays; "
        "out: (rates (n_bins,) with NaN for empty bins, counts (n_bins,) int)"
    )


def coverage_table(
    results: dict[str, np.ndarray],
    masks: list[np.ndarray],
    labels: list[str] | None = None,
) -> "object":
    """Assemble per-bin coverage for several methods into one tidy table.

    in: results mapping method name ("raw", "aci") -> bool covered array;
        masks from binning.bin_masks; labels from config.p_bin_labels().
    out: a pandas DataFrame with columns
        ``method, bin_label, bin_lo, bin_hi, coverage, count, target``
        ready to write to config.COVERAGE_TABLE_PATH and to be read straight
        back by scripts/04_figure.py.
    """
    raise NotImplementedError(
        "in: results {method: covered bool array}, masks, labels; "
        "out: tidy DataFrame [method, bin_label, bin_lo, bin_hi, coverage, count, target]"
    )
