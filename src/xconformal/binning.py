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


def exceedance_probability(
    ensemble: np.ndarray, threshold: np.ndarray, member_axis: int | None = None
) -> np.ndarray:
    """Fraction of members strictly above the threshold, per gridpoint.

    in: ensemble with a member axis somewhere, threshold shaped like the
        ensemble with that axis removed (e.g. ensemble (n_time, n_members,
        lat, lon) against threshold (n_time, lat, lon));
    out: p in [0, 1] with the member axis removed.

    ``member_axis`` is inferred when it is unambiguous: the member axis is the
    one whose removal leaves a shape that broadcasts against the threshold.
    Getting this wrong silently is the failure mode worth guarding against --
    reducing over time instead of over members would produce a p_t that looks
    perfectly plausible and means nothing -- so an ambiguous case raises rather
    than guessing.

    Strict inequality, so a member exactly at the threshold is not an
    exceedance. With M members p takes only M+1 distinct values; p == 0 and
    p == 1 are both common and must land in the first and last bin.
    """
    ensemble = np.asarray(ensemble)
    threshold = np.asarray(threshold)

    if member_axis is None:
        member_axis = _infer_member_axis(ensemble.shape, threshold.shape)
    member_axis %= ensemble.ndim

    expanded = np.expand_dims(threshold, member_axis) if threshold.ndim else threshold
    return (ensemble > expanded).mean(axis=member_axis)


def _infer_member_axis(ensemble_shape: tuple[int, ...], threshold_shape: tuple[int, ...]) -> int:
    """The unique axis of the ensemble whose removal matches the threshold."""
    if len(ensemble_shape) != len(threshold_shape) + 1:
        raise ValueError(
            f"ensemble {ensemble_shape} should have exactly one more axis than "
            f"threshold {threshold_shape}; pass member_axis explicitly if not"
        )
    def removed(axis: int) -> tuple[int, ...]:
        return ensemble_shape[:axis] + ensemble_shape[axis + 1:]

    # Prefer an exact shape match. Falling straight to broadcast rules makes a
    # size-1 grid ambiguous -- (4, 1) against (1,) matches on both axes,
    # because a length-4 axis "broadcasts" against a length-1 threshold.
    candidates = [a for a in range(len(ensemble_shape)) if removed(a) == threshold_shape]
    if not candidates:
        candidates = [
            a for a in range(len(ensemble_shape))
            if _broadcastable(removed(a), threshold_shape)
        ]
    if not candidates:
        raise ValueError(
            f"no axis of ensemble {ensemble_shape} can be the member axis for a "
            f"threshold of shape {threshold_shape}"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"member axis is ambiguous for ensemble {ensemble_shape} and threshold "
            f"{threshold_shape} (candidates {candidates}); pass member_axis explicitly"
        )
    return candidates[0]


def _broadcastable(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    return len(a) == len(b) and all(x == y or x == 1 or y == 1 for x, y in zip(a, b))


def bin_index(p: np.ndarray, edges: tuple[float, ...] = config.P_BIN_EDGES) -> np.ndarray:
    """Assign each p to a bin index in ``0 .. len(edges) - 2``.

    in: p (any shape) in [0, 1], edges ascending; out: int array, same shape.

    Bins are half-open ``[edges[i], edges[i+1])`` except the last, which is
    closed so p == 1 is included. Every p in [0, 1] gets exactly one bin.
    Values outside [0, 1], and NaN, are a bug in the caller and raise.
    """
    p = np.asarray(p, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError(f"edges must be strictly ascending with >=2 entries, got {edges}")
    if np.isnan(p).any():
        raise ValueError("p contains NaN")
    if (p < edges[0]).any() or (p > edges[-1]).any():
        raise ValueError(
            f"p outside [{edges[0]}, {edges[-1]}]: "
            f"min {float(np.min(p))}, max {float(np.max(p))}"
        )
    # side="right" gives left-closed, right-open bins; the clip folds the
    # closed right end of the last bin back into it.
    idx = np.searchsorted(edges, p, side="right") - 1
    return np.clip(idx, 0, edges.size - 2).astype(int)


def bin_masks(
    p: np.ndarray, edges: tuple[float, ...] = config.P_BIN_EDGES
) -> list[np.ndarray]:
    """Boolean mask per bin.

    in: p (any shape), edges; out: list of len(edges)-1 bool arrays of p.shape.

    The masks are mutually exclusive and exhaustive by construction: they are
    equality tests on a single bin-index array, so no sample can be counted
    twice or dropped.
    """
    idx = bin_index(p, edges)
    return [idx == i for i in range(len(edges) - 1)]


def bin_counts(
    p: np.ndarray, edges: tuple[float, ...] = config.P_BIN_EDGES
) -> np.ndarray:
    """Number of samples per bin.

    in: p (any shape), edges; out: int array (len(edges)-1,) summing to p.size.

    These counts go ON the headline figure -- the high-p_t bins are rare by
    construction and the reader must be able to see how rare.
    """
    idx = bin_index(p, edges)
    return np.bincount(idx.ravel(), minlength=len(edges) - 1).astype(int)
