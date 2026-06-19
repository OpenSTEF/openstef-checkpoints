# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the publish manifest and card rendering (no network)."""

from pathlib import Path

from openstef_checkpoints.models.chronos2.config import Chronos2Model
from openstef_checkpoints.publish import ExportProvenance, ExportWindow, Manifest, VariantRecord


def _manifest() -> Manifest:
    return Manifest(
        slug="chronos-2",
        repo_id="OpenSTEF/chronos-2-onnx",
        provenance=ExportProvenance(
            source_model_id="amazon/chronos-2",
            source_revision="abc123",
            exporter_revision="def456",
            tooling="onnx=1.17 onnxruntime=1.20 torch=2.4",
            exported_at="2026-06-17T00:00:00+00:00",
        ),
        window=ExportWindow(
            context_length=5760,
            horizon_length=672,
            resolution_minutes=15,
            static_batch=4,
        ),
        variants=[
            VariantRecord(
                filename="chronos-2_static.onnx",
                precision="fp32",
                static_shapes=True,
                max_abs=1e-5,
                within_tolerance=True,
                publish=True,
            ),
            VariantRecord(
                filename="chronos-2.onnx",
                precision="fp32",
                static_shapes=False,
                max_abs=2e-5,
                within_tolerance=True,
                publish=True,
            ),
            VariantRecord(
                filename="chronos-2_fp16.onnx",
                precision="fp16",
                static_shapes=False,
                max_abs=0.3,
                within_tolerance=False,
                publish=False,
            ),
        ],
    )


def test_manifest_round_trips_through_directory(tmp_path: Path) -> None:
    """A manifest writes to and reads back from a directory unchanged."""
    manifest = _manifest()
    manifest.write(tmp_path)
    assert Manifest.read(tmp_path) == manifest


def test_card_advertises_only_published_variants_with_license_and_provenance() -> None:
    """The card lists published variants, the upstream license, and provenance, never build-only ones."""
    card = _manifest().render_card(Chronos2Model.CARD_TEMPLATE, source_license="apache-2.0")
    assert "chronos-2_static.onnx" in card
    assert "chronos-2_fp16.onnx" not in card  # build-only (publish=False) is withheld
    assert "amazon/chronos-2" in card
    assert "apache-2.0" in card  # upstream weights license
    assert "def456" in card  # exporter revision stamped


def test_card_describes_the_static_shape() -> None:
    """The card spells out the fixed batch, context, and horizon a static graph bakes in."""
    card = _manifest().render_card(Chronos2Model.CARD_TEMPLATE, source_license="apache-2.0")
    assert "`context` | (4, 5760)" in card  # batch from static_batch, context from the window
    assert "`future_covariates` | (4, 672)" in card  # horizon from the window
    assert "60 days at 15-minute resolution" in card  # 5760 steps * 15 min


def test_selection_splits_held_back_failing_and_uploadable() -> None:
    """The fixture's passing/failing/build-only mix sorts into the right buckets."""
    manifest = _manifest()
    assert [record.filename for record in manifest.held_back] == ["chronos-2_fp16.onnx"]
    assert manifest.failing == []  # the only failing variant is build-only, so not publishable
    assert {record.filename for record in manifest.selected_for_upload(force=False)} == {
        "chronos-2_static.onnx",
        "chronos-2.onnx",
    }
