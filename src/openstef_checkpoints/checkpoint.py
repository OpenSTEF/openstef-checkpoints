# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Checkpoint metadata and the exported ONNX file it describes.

`CheckpointMetadata` records what is needed to run an exported model: the input and
output tensor names, the context length, the quantile grid, and the shapes and
precision the graph was built for. It is written to a JSON file next to the weights
and read back by OpenSTEF when it loads the model.

The same model is defined in the OpenSTEF library. The two copies are kept identical
by hand rather than shared as code (a shared dependency would couple the two release
cycles). A schema snapshot test, `tests/unit/test_checkpoint.py`, fails if either side
changes the schema without the other.
"""

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field


class CheckpointMetadata(BaseModel):
    """Description of an exported checkpoint, written to a JSON file beside its weights."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    #: Layout version of the metadata. Bumped only on a breaking change, and matched by
    #: the same constant in the OpenSTEF library.
    SCHEMA_VERSION: ClassVar[int] = 2

    schema_version: int = Field(default=SCHEMA_VERSION, description="Metadata layout version.")
    model_family: str = Field(description="Model family identifier, e.g. 'chronos2'.")
    input_names: list[str] = Field(min_length=1, description="Model input tensor names, in order.")
    output_name: str = Field(description="Name of the quantile-prediction output tensor.")
    native_quantiles: list[float] = Field(min_length=1, description="Quantile levels the model emits, ascending.")
    context_length: int = Field(gt=0, description="Number of historical timesteps consumed as context.")
    output_patch_size: int = Field(gt=0, description="Timesteps produced per output patch.")
    horizon_patches: int = Field(gt=0, description="Number of output patches emitted.")
    resolution_minutes: int = Field(gt=0, description="Expected sampling interval, in minutes.")
    precision: Literal["fp32", "fp16", "int8"] = Field(
        default="fp32",
        description="Weight precision. int8 runs on CPU but not CoreML; fp16 and fp32 take the CoreML path.",
    )
    static_shapes: bool = Field(
        default=False,
        description="Whether every graph axis is a fixed size. Static graphs are eligible for CoreML.",
    )
    max_covariates: int | None = Field(
        default=None,
        gt=0,
        description="Number of covariate series the graph is fixed to, or None if that axis is dynamic.",
    )

    @property
    def horizon_length(self) -> int:
        """Total number of forecast timesteps emitted."""
        return self.output_patch_size * self.horizon_patches


class ExportedCheckpoint(BaseModel):
    """An exported ONNX weights file together with its metadata."""

    model_config = ConfigDict(frozen=True)

    weights_path: Path = Field(description="Path to the exported ONNX weights file.")
    metadata: CheckpointMetadata = Field(description="Metadata describing this checkpoint.")

    @property
    def metadata_path(self) -> Path:
        """Path of the metadata file: the weights path with a `.metadata.json` suffix."""
        return self.weights_path.with_suffix(".metadata.json")

    def write_metadata(self) -> Path:
        """Write the metadata file next to the weights.

        Returns:
            The path of the file written.
        """
        self.metadata_path.write_text(self.metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return self.metadata_path
