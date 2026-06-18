# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the checkpoint contract: the golden schema and the artifact sidecar.

`CheckpointMetadata` is a governed duplicate of the schema in
`openstef-foundation-models`; the golden test fails CI on any drift instead of
silently shipping a changed sidecar. Regenerate with::

    python tests/test_checkpoint.py
"""

import json
from pathlib import Path

from openstef_checkpoints.checkpoint import CheckpointMetadata, ExportedCheckpoint

GOLDEN_PATH = Path(__file__).parent / "checkpoint.golden.json"


def _metadata() -> CheckpointMetadata:
    return CheckpointMetadata(
        model_family="chronos2",
        input_names=["context"],
        output_name="quantile_preds",
        native_quantiles=[0.5],
        context_length=64,
        output_patch_size=16,
        horizon_patches=2,
        resolution_minutes=15,
    )


def test_metadata_schema_matches_golden() -> None:
    """The metadata JSON Schema is byte-stable against the committed golden."""
    assert CheckpointMetadata.model_json_schema() == json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_exported_checkpoint_round_trips_its_sidecar(tmp_path: Path) -> None:
    """The artifact writes a sidecar next to its weights that re-validates to the same metadata."""
    checkpoint = ExportedCheckpoint(weights_path=tmp_path / "model.onnx", metadata=_metadata())
    path = checkpoint.write_sidecar()
    assert path == tmp_path / "model.metadata.json"
    assert CheckpointMetadata.model_validate_json(path.read_text(encoding="utf-8")) == _metadata()


if __name__ == "__main__":
    GOLDEN_PATH.write_text(json.dumps(CheckpointMetadata.model_json_schema(), indent=2) + "\n", encoding="utf-8")
