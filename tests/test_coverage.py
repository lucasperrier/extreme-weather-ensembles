"""Coverage counts must add up, and an empty bin must not read as zero coverage.

An empty high-p_t bin plotted as 0.0 coverage would be the most alarming and
most wrong point on the headline figure. It has to be NaN with a count of 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from xconformal import binning, config, coverage



def test_covered_is_a_closed_interval():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    lower = np.array([1.0, 1.0, 1.0, 1.0])
    upper = np.array([2.0, 2.0, 2.0, 2.0])
    assert list(coverage.covered(y, lower, upper)) == [False, True, True, False], (
        "endpoints are inside the interval"
    )


def test_miscoverage_is_the_complement_of_covered():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    lower, upper = -1.0, 1.0
    is_covered = coverage.covered(y, lower, upper)
    err = coverage.miscoverage(y, lower, upper)
    assert np.array_equal(err, 1.0 - is_covered.astype(float))
    # This is the same err ACI updates on; if these two ever disagree, the
    # controller is optimising a different quantity than the one we report.


def test_marginal_coverage_rate_and_count():
    is_covered = np.array([True, True, True, False])
    rate, count = coverage.marginal_coverage(is_covered)
    assert rate == pytest.approx(0.75)
    assert count == 4


def test_marginal_coverage_of_nothing_is_nan_not_zero():
    rate, count = coverage.marginal_coverage(np.zeros(0, dtype=bool))
    assert count == 0
    assert np.isnan(rate)


def test_per_bin_counts_sum_to_total():
    rng = np.random.default_rng(1)
    p = rng.random(2000)
    is_covered = rng.random(2000) < 0.9
    masks = binning.bin_masks(p, config.P_BIN_EDGES)
    rates, counts = coverage.per_bin_coverage(is_covered, masks)

    assert rates.shape == counts.shape == (len(config.P_BIN_EDGES) - 1,)
    assert counts.sum() == is_covered.size
    finite = np.isfinite(rates)
    assert np.all((rates[finite] >= 0.0) & (rates[finite] <= 1.0))
    # The count-weighted mean of the bin rates is the marginal rate.
    marginal, _ = coverage.marginal_coverage(is_covered)
    assert np.average(rates[finite], weights=counts[finite]) == pytest.approx(marginal)


def test_empty_bin_is_nan_with_zero_count():
    # All mass in the first bin, so every other bin is empty.
    p = np.full(100, 0.05)
    is_covered = np.ones(100, dtype=bool)
    masks = binning.bin_masks(p, config.P_BIN_EDGES)
    rates, counts = coverage.per_bin_coverage(is_covered, masks)

    assert counts[0] == 100
    assert rates[0] == pytest.approx(1.0)
    assert np.all(counts[1:] == 0)
    assert np.all(np.isnan(rates[1:])), (
        "an empty bin must be NaN; 0.0 would plot as total miscoverage"
    )


def test_coverage_table_shape_and_columns():
    rng = np.random.default_rng(2)
    p = rng.random(500)
    masks = binning.bin_masks(p, config.P_BIN_EDGES)
    labels = config.p_bin_labels()
    results = {
        "raw": rng.random(500) < 0.7,
        "aci-variable": rng.random(500) < 0.9,
    }
    table = coverage.coverage_table(results, masks, labels)

    expected = {"method", "bin_label", "bin_lo", "bin_hi", "coverage", "count", "target"}
    assert expected <= set(table.columns)
    assert len(table) == len(results) * len(masks)
    assert set(table["method"]) == set(results)
    assert set(table["bin_label"]) == set(labels)
    assert np.allclose(table["target"], config.TARGET_COVERAGE)
    for method in results:
        assert table.loc[table["method"] == method, "count"].sum() == 500
