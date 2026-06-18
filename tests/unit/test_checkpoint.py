# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for `CheckpointMetadata` and the metadata file it writes.

`metadata_schema.json` is a committed snapshot of the metadata's JSON Schema. The
schema-matches-snapshot test fails on any change, which is what keeps the hand-synced
copy in the OpenSTEF library (see `checkpoint.py`) from drifting. After an intended
schema change, regenerate the snapshot with::

    python tests/unit/test_checkpoint.py
"""

import json
from pathlib import Path

from openstef_checkpoints.checkpoint import CheckpointMetadata, ExportedCheckpoint

SCHEMA_PATH = Path(__file__).parent / "metadata_schema.json"


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


def test_metadata_schema_matches_snapshot() -> None:
    """The metadata schema is unchanged from the committed snapshot."""
    assert CheckpointMetadata.model_json_schema() == json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_written_metadata_reloads_unchanged(tmp_path: Path) -> None:
    """The metadata file written beside the weights reloads to the same metadata."""
    checkpoint = ExportedCheckpoint(weights_path=tmp_path / "model.onnx", metadata=_metadata())
    path = checkpoint.write_metadata()
    assert path == tmp_path / "model.metadata.json"
    assert CheckpointMetadata.model_validate_json(path.read_text(encoding="utf-8")) == _metadata()


if __name__ == "__main__":
    SCHEMA_PATH.write_text(json.dumps(CheckpointMetadata.model_json_schema(), indent=2) + "\n", encoding="utf-8")
