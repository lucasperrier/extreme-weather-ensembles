"""Bin masks must partition the data, and p == 0 / p == 1 must land somewhere.

With N members p_t can only take N+1 values, and the two ends are the common
cases: on most days no member exceeds the climatological 95th percentile, and
on the days that matter every member does. If either end falls outside every
mask, the headline figure loses exactly the samples it is about.
"""

from __future__ import annotations

import numpy as np
import pytest

from xconformal import binning, config

pending = pytest.mark.xfail(
    raises=NotImplementedError,
    strict=False,
    reason="TODO(lucas): [Thu] binning.py is a stub",
)

EDGES = config.P_BIN_EDGES


def test_bin_labels_match_edges():
    """Labels are pure config, so this runs green today."""
    labels = config.p_bin_labels(EDGES)
    assert len(labels) == len(EDGES) - 1
    assert labels[0].startswith("[0")
    assert labels[-1].endswith("]"), "the last bin must be closed so p == 1 is included"
    assert labels[-2].endswith(")"), "every other bin is half-open"


@pending
def test_masks_partition_the_data():
    rng = np.random.default_rng(0)
    p = rng.random((37, 11))
    masks = binning.bin_masks(p, EDGES)

    assert len(masks) == len(EDGES) - 1
    stacked = np.stack(masks)
    assert np.all(stacked.sum(axis=0) == 1), "every sample belongs to exactly one bin"
    assert sum(int(m.sum()) for m in masks) == p.size
    for mask in masks:
        assert mask.shape == p.shape
        assert mask.dtype == bool


@pending
def test_edge_cases_p_zero_and_p_one():
    p = np.array([0.0, 1.0])
    idx = binning.bin_index(p, EDGES)
    assert idx[0] == 0, "p == 0 belongs to the first bin"
    assert idx[1] == len(EDGES) - 2, "p == 1 belongs to the LAST bin, not off the end"

    masks = binning.bin_masks(p, EDGES)
    assert masks[0][0] and not masks[-1][0]
    assert masks[-1][1] and not masks[0][1]


@pending
def test_interior_edges_are_left_closed():
    """A value exactly on an interior edge goes to the bin it opens."""
    for i, edge in enumerate(EDGES[1:-1], start=1):
        idx = binning.bin_index(np.array([edge]), EDGES)
        assert idx[0] == i, f"p == {edge} should open bin {i}, got {idx[0]}"


@pending
def test_out_of_range_probability_raises():
    for bad in (-0.01, 1.01, np.nan):
        with pytest.raises((ValueError, AssertionError)):
            binning.bin_index(np.array([bad]), EDGES)


@pending
def test_bin_counts_sum_to_size():
    rng = np.random.default_rng(1)
    p = rng.random(1000)
    counts = binning.bin_counts(p, EDGES)
    assert counts.shape == (len(EDGES) - 1,)
    assert counts.sum() == p.size


@pending
def test_exceedance_probability_is_a_member_fraction():
    # 4 members, one gridpoint: exactly one member above the threshold.
    ensemble = np.array([[0.0], [1.0], [2.0], [3.0]])
    threshold = np.array([2.5])
    p = binning.exceedance_probability(ensemble, threshold)
    assert p.shape == (1,)
    assert p[0] == pytest.approx(0.25)

    # Strict inequality: a member exactly at the threshold is not an exceedance.
    assert binning.exceedance_probability(ensemble, np.array([3.0]))[0] == pytest.approx(0.0)
    assert binning.exceedance_probability(ensemble, np.array([-1.0]))[0] == pytest.approx(1.0)


@pending
def test_exceedance_probability_over_a_time_axis():
    rng = np.random.default_rng(2)
    ensembles = rng.normal(size=(7, 20, 3, 4))       # (time, member, lat, lon)
    threshold = rng.normal(size=(7, 3, 4))
    p = binning.exceedance_probability(ensembles, threshold)
    assert p.shape == (7, 3, 4), "the member axis is the one that gets reduced"
    assert np.all((p >= 0.0) & (p <= 1.0))
    # 20 members means p is a multiple of 1/20 and nothing else.
    assert np.allclose(p * 20, np.round(p * 20))
