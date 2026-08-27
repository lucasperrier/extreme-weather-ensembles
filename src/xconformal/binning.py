"""Forecast exceedance probability p_t and the bins built on it.

p_t is the ensemble's own answer to "how likely is an extreme here?": the
fraction of members exceeding the climatological threshold. Binning
verification days by p_t is what turns a marginal coverage statement into a
conditional one, and is the axis of the headline figure.
"""

from __future__ import annotations

import numpy as np

from . import config

__all__ = [
    "exceedance_probability",
    "bin_index",
    "bin_masks",
    "bin_counts",
]


def exceedance_probability(ensemble: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """Fraction of members strictly above the threshold, per gridpoint.

    in: ensemble (n_members, *grid) or (n_time, n_members, *grid),
        threshold broadcastable to (*grid) / (n_time, *grid);
    out: p in [0, 1] with the member axis removed.

    Strict inequality (``ensemble > threshold``), so that a member exactly at
    the threshold does not count as an exceedance. With n_members members p can
    only take n_members + 1 distinct values; p == 0 and p == 1 are both common
    and must land in the first and last bin respectively.
    """
    raise NotImplementedError(
        "in: ensemble (..., n_members, *grid), threshold broadcastable; "
        "out: p (..., *grid) in [0,1], member axis reduced"
    )


def bin_index(p: np.ndarray, edges: tuple[float, ...] = config.P_BIN_EDGES) -> np.ndarray:
    """Assign each p to a bin index in ``0 .. len(edges) - 2``.

    in: p (any shape) in [0, 1], edges ascending; out: int array, same shape.

    Bins are half-open ``[edges[i], edges[i+1])`` except the last, which is
    closed ``[edges[-2], edges[-1]]`` so that p == 1 is included. Every p in
    [0, 1] gets exactly one bin: the masks partition the data, with no gaps and
    no overlap. Values outside [0, 1] are a bug in the caller -- raise.
    """
    raise NotImplementedError(
        "in: p (any shape) in [0,1], edges; out: int array of same shape in [0, len(edges)-2]"
    )


def bin_masks(
    p: np.ndarray, edges: tuple[float, ...] = config.P_BIN_EDGES
) -> list[np.ndarray]:
    """Boolean mask per bin.

    in: p (any shape), edges; out: list of len(edges)-1 bool arrays of p.shape.

    Guaranteed: the masks are mutually exclusive and their union is everything,
    i.e. ``sum(m.sum() for m in masks) == p.size``.
    """
    raise NotImplementedError(
        "in: p (any shape), edges; out: list of len(edges)-1 bool arrays shaped like p, "
        "mutually exclusive and exhaustive"
    )


def bin_counts(
    p: np.ndarray, edges: tuple[float, ...] = config.P_BIN_EDGES
) -> np.ndarray:
    """Number of samples per bin.

    in: p (any shape), edges; out: int array (len(edges)-1,) summing to p.size.

    These counts go ON the headline figure -- the high-p_t bins are rare by
    construction and the reader must be able to see how rare.
    """
    raise NotImplementedError(
        "in: p (any shape), edges; out: int array (len(edges)-1,) summing to p.size"
    )
