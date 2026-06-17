# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the publish manifest and card rendering (no network)."""

from pathlib import Path

from fmckpt.models.chronos2.config import CARD_TEMPLATE
from fmckpt.publish import ExportProvenance, Manifest, VariantRecord, render_card


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


def test_card_advertises_only_published_variants_and_provenance() -> None:
    """The card lists published variants and the provenance, but never build-only ones."""
    card = render_card(CARD_TEMPLATE, _manifest())
    assert "chronos-2_static.onnx" in card
    assert "chronos-2_fp16.onnx" not in card  # build-only (publish=False) is withheld
    assert "amazon/chronos-2" in card
    assert "def456" in card  # exporter revision stamped
