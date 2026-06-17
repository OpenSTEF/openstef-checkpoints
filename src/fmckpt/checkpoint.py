# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""The checkpoint contract and the exported artifact.

`CheckpointMetadata` is a *governed duplicate* of the schema in
`openstef_foundation_models.models.checkpoint`: the two repos share no code (that
would form a release cycle at every schema bump), so the copies are kept compatible
by an append-only rule and the golden conformance test (``tests/test_checkpoint.py``),
which fails CI on any drift rather than silently shipping a changed sidecar.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: Schema version of the sidecar metadata; mirrors the consumer's constant.
METADATA_SCHEMA_VERSION = 2


class CheckpointMetadata(BaseModel):
    """Sidecar metadata describing a checkpoint, written next to its weights."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    schema_version: int = Field(default=METADATA_SCHEMA_VERSION, description="Metadata layout version.")
    model_family: str = Field(description="Model family identifier, e.g. 'chronos2'.")
    input_names: list[str] = Field(min_length=1, description="Ordered model input tensor names.")
    output_name: str = Field(description="Quantile-prediction output tensor name.")
    native_quantiles: list[float] = Field(min_length=1, description="Quantile levels the model emits, ascending.")
    context_length: int = Field(gt=0, description="Historical timesteps consumed as context.")
    output_patch_size: int = Field(gt=0, description="Timesteps produced per output patch.")
    horizon_patches: int = Field(gt=0, description="Output patches emitted (frozen horizon).")
    resolution_minutes: int = Field(gt=0, description="Expected sampling interval, in minutes.")
    precision: Literal["fp32", "fp16", "int8"] = Field(
        default="fp32",
        description="Weight precision. int8 (QDQ) is fast on CPU but not CoreML; fp16/fp32 take the CoreML path.",
    )
    static_shapes: bool = Field(
        default=False,
        description="Whether all graph axes are frozen (CoreML-eligible); sizes are given by context/horizon length.",
    )
    max_covariates: int | None = Field(
        default=None,
        gt=0,
        description="Frozen covariate-series count, or None if that axis is dynamic. Independent of static_shapes.",
    )

    @property
    def horizon_length(self) -> int:
        """Total forecast timesteps emitted (``output_patch_size * horizon_patches``)."""
        return self.output_patch_size * self.horizon_patches


class ExportedCheckpoint(BaseModel):
    """An exported ONNX weights file paired with its metadata, owning its sidecar."""

    model_config = ConfigDict(frozen=True)

    weights_path: Path = Field(description="Path to the exported ONNX weights file.")
    metadata: CheckpointMetadata = Field(description="Metadata describing this checkpoint.")

    @property
    def sidecar_path(self) -> Path:
        """Sidecar path (the weights path with a ``.metadata.json`` suffix)."""
        return self.weights_path.with_suffix(".metadata.json")

    def write_sidecar(self) -> Path:
        """Write the metadata sidecar next to the weights file.

        Returns:
            The path written.
        """
        self.sidecar_path.write_text(self.metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return self.sidecar_path
