# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the output comparison and the test-series helpers."""

import numpy as np
import pytest

from openstef_checkpoints.verify import compare_outputs, inject_nan_gaps, synthetic_series


def test_identical_outputs_are_within_tolerance() -> None:
    """An exact match reports zero deviation and passes."""
    reference = np.array([[1.0, 2.0], [3.0, 4.0]])
    report = compare_outputs(reference, reference.copy(), atol=1e-6, rtol=1e-6)
    assert report.max_abs == 0.0
    assert report.within_tolerance


def test_verdict_tracks_tolerance() -> None:
    """The same drift passes a loose tolerance and fails a tight one."""
    reference = np.array([10.0, 20.0, 30.0])
    candidate = reference + 0.01
    assert compare_outputs(reference, candidate, atol=0.1, rtol=0.0).within_tolerance
    assert not compare_outputs(reference, candidate, atol=1e-4, rtol=0.0).within_tolerance


def test_reports_metrics_on_a_known_difference() -> None:
    """Absolute, mean and RMS figures are computed over all elements."""
    report = compare_outputs(np.zeros(4), np.array([0.0, 0.0, 0.0, 2.0]), atol=0.0, rtol=0.0)
    assert report.max_abs == 2.0
    assert report.mean_abs == 0.5
    assert report.rmse == pytest.approx(1.0)
    assert not report.within_tolerance


def test_shape_mismatch_raises() -> None:
    """Comparing differently-shaped outputs is a programming error, not a deviation."""
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_outputs(np.zeros((2, 3)), np.zeros((2, 4)), atol=0.0, rtol=0.0)


def test_synthetic_series_shape_dtype_and_determinism() -> None:
    """A series has the requested length, float32 dtype, and is seed-reproducible."""
    series = synthetic_series(128, seed=0)
    assert series.shape == (128,)
    assert series.dtype == np.float32
    np.testing.assert_array_equal(series, synthetic_series(128, seed=0))
    assert not np.array_equal(series, synthetic_series(128, seed=1))


def test_inject_nan_gaps_adds_missing_and_preserves_source() -> None:
    """Gaps introduce NaNs for the NaN-aware path, leaving the original untouched."""
    series = synthetic_series(100, seed=2)
    gapped = inject_nan_gaps(series, gaps=3, gap_length=5, seed=0)
    assert not np.isnan(series).any()
    assert 1 <= np.isnan(gapped).sum() <= 15  # up to 3 gaps of 5 (may overlap)
    assert not np.isnan(inject_nan_gaps(series, gaps=0, gap_length=5, seed=0)).any()
