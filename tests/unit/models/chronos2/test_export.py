# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the Chronos-2 exporter's degenerate-model guard.

These need torch and chronos (the `[chronos]` extra), so they skip when it is absent. The
guard's logic is exercised without loading the real model by stubbing the torch reference and
the representative inputs, so the tests stay fast and offline.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("chronos")

from openstef_checkpoints.graph.verify import DeviationReport
from openstef_checkpoints.models.chronos2.config import Variant
from openstef_checkpoints.models.chronos2.export import Chronos2Exporter
from openstef_checkpoints.models.registry import MODELS


def _exporter(tmp_path: Path) -> Chronos2Exporter:
    return Chronos2Exporter(model=MODELS["chronos-2"], out_dir=tmp_path, device="cpu")


def _report(*, rel_mean: float, within_tolerance: bool) -> DeviationReport:
    """A DeviationReport with only the fields the verdict reads set meaningfully."""
    return DeviationReport(max_abs=0.0, mean_abs=0.0, rel_mean=rel_mean, rmse=0.0, within_tolerance=within_tolerance)


def _stub_inputs(**_kwargs: object) -> dict[str, np.ndarray]:
    """A minimal representative-input dict: only the keys `_assert_responsive` scales."""
    return {
        "context": np.linspace(0.0, 1.0, 16, dtype=np.float32),
        "future_covariates": np.zeros(8, dtype=np.float32),
    }


def test_responsive_model_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A model whose forecast scales with the input (instance norm) passes the guard."""
    monkeypatch.setattr(Chronos2Exporter, "_representative_inputs", staticmethod(_stub_inputs))
    # Output spread tracks the input's: scaling the input 10x scales the output spread 10x.
    monkeypatch.setattr(Chronos2Exporter, "_reference", lambda _self, inputs: inputs["context"])

    _exporter(tmp_path)._assert_responsive()  # does not raise


def test_degenerate_model_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that emits a near-constant forecast regardless of input is rejected."""
    monkeypatch.setattr(Chronos2Exporter, "_representative_inputs", staticmethod(_stub_inputs))
    # Output ignores the input: same near-constant spread however the input is scaled.
    monkeypatch.setattr(Chronos2Exporter, "_reference", lambda _self, _inputs: np.zeros(4, dtype=np.float32))

    with pytest.raises(RuntimeError, match="degenerate"):
        _exporter(tmp_path)._assert_responsive()


def test_int8_judged_by_relative_mean_not_elementwise() -> None:
    """int8 passes on small aggregate error even when the elementwise gate failed on an outlier."""
    # Elementwise verdict is False (a single large pointwise drift), but mean relative error is
    # well under budget — chronos-2's realistic int8 case (~2.5% synthetic, ~1% real MAE).
    report = _report(rel_mean=0.025, within_tolerance=False)
    assert Chronos2Exporter._passes(report, Variant(precision="int8", static=False))


def test_int8_passes_for_the_smaller_model_higher_but_acceptable_drift() -> None:
    """chronos-2-small quantises worse (~11% synthetic) yet is faithful in practice (~2% real MAE)."""
    report = _report(rel_mean=0.11, within_tolerance=False)
    assert Chronos2Exporter._passes(report, Variant(precision="int8", static=False))


def test_int8_fails_when_aggregate_error_exceeds_budget() -> None:
    """A catastrophically broken int8 (large mean relative error) still fails."""
    report = _report(rel_mean=0.5, within_tolerance=False)
    assert not Chronos2Exporter._passes(report, Variant(precision="int8", static=False))


def test_fp32_keeps_the_strict_elementwise_verdict() -> None:
    """fp32 ignores the int8 budget and uses the tight elementwise verdict, pass or fail."""
    assert Chronos2Exporter._passes(
        _report(rel_mean=0.5, within_tolerance=True), Variant(precision="fp32", static=False)
    )
    assert not Chronos2Exporter._passes(
        _report(rel_mean=0.0, within_tolerance=False), Variant(precision="fp32", static=True)
    )
