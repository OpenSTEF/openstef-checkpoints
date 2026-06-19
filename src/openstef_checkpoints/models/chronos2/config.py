# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Configuration for exporting a Chronos-2 model to ONNX.

`Chronos2Model` describes one published size: where to download it, how large a
context and horizon to build for, and which variants to produce. `Variant` is one
point in the export matrix: a weight precision and whether the graph's shapes are
fixed. The preconfigured sizes live in `openstef_checkpoints.models.registry`.

No torch import here, so the CLI can list and size variants without the source-model
stack.
"""

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field


class Variant(BaseModel):
    """One export variant: a weight precision and whether the graph shapes are fixed."""

    model_config = ConfigDict(frozen=True)

    precision: Literal["fp32", "fp16", "int8"] = Field(description="Weight precision.")
    static: bool = Field(description="Whether every graph axis is a fixed size (eligible for CoreML).")
    publish: bool = Field(
        default=True,
        description="Whether to upload this variant. False still builds and checks it, for a variant kept in "
        "the pipeline while a known problem is worked on.",
    )

    @property
    def suffix(self) -> str:
        """Filename suffix for this variant, e.g. `_static_int8` (fp32 adds nothing)."""
        precision_part = "" if self.precision == "fp32" else f"_{self.precision}"
        return f"{'_static' if self.static else ''}{precision_part}"

    @property
    def name(self) -> str:
        """Short identifier, e.g. `fp32-static` or `int8-dynamic`."""
        return f"{self.precision}-{'static' if self.static else 'dynamic'}"


class Chronos2Model(BaseModel):
    """A published Chronos-2 size and how to export it."""

    model_config = ConfigDict(frozen=True)

    #: Chronos-2's output patch length. The horizon is built as a whole number of these;
    #: the exported metadata copies the real value from the loaded model.
    OUTPUT_PATCH_SIZE: ClassVar[int] = 16

    #: Minutes in a day, for turning day-sized windows into step counts.
    MINUTES_PER_DAY: ClassVar[int] = 24 * 60

    #: Model-card template, shared by all sizes.
    CARD_TEMPLATE: ClassVar[Path] = Path(__file__).parent / "card.md.jinja"

    #: Variants built by default, most-preferred first. int8-static is omitted: int8 never
    #: reaches CoreML, and CPU, CUDA, and TensorRT all take the dynamic graph. fp16 is left
    #: out entirely: it is not shipped (an open accuracy issue) and its conversion currently
    #: produces an invalid graph for some models, so building it only breaks the export.
    DEFAULT_VARIANTS: ClassVar[tuple[Variant, ...]] = (
        Variant(precision="fp32", static=True),
        Variant(precision="fp32", static=False),
        Variant(precision="int8", static=False),
    )

    slug: str = Field(description="Short identifier and filename stem, e.g. 'chronos-2'.")
    source_model_id: str = Field(description="HuggingFace id of the model to download and export.")
    source_license: str = Field(
        default="apache-2.0",
        description="License of the upstream weights. The published checkpoint is a derivative, so the model "
        "card carries this license rather than the MPL-2.0 of the export tooling.",
    )
    context_days: int = Field(gt=0, default=60, description="Context window in days, clamped to the model maximum.")
    horizon_days: int = Field(gt=0, default=7, description="Forecast horizon in days, rounded up to a whole patch.")
    resolution_minutes: int = Field(gt=0, default=15, description="Data resolution in minutes.")
    static_covariates: int = Field(
        gt=0,
        default=3,
        description="Covariate rows frozen into a static export; its batch is one target plus this many.",
    )

    @property
    def steps_per_day(self) -> int:
        """Number of timesteps in a day at this resolution.

        Returns:
            The step count.

        Raises:
            ValueError: If the resolution does not divide a day evenly.
        """
        if self.MINUTES_PER_DAY % self.resolution_minutes != 0:
            msg = f"resolution {self.resolution_minutes} min does not divide a day evenly"
            raise ValueError(msg)
        return self.MINUTES_PER_DAY // self.resolution_minutes

    @property
    def context_length(self) -> int:
        """Requested context length in steps."""
        return self.context_days * self.steps_per_day

    @property
    def num_output_patches(self) -> int:
        """Number of output patches needed to cover the horizon, rounded up."""
        horizon_steps = self.horizon_days * self.steps_per_day
        return -(-horizon_steps // self.OUTPUT_PATCH_SIZE)

    def weights_name(self, variant: Variant) -> str:
        """Weights filename for a variant, e.g. `chronos-2_static_int8.onnx`.

        Args:
            variant: The variant to name.

        Returns:
            The .onnx filename.
        """
        return f"{self.slug}{variant.suffix}.onnx"

    def max_covariates(self, variant: Variant) -> int | None:
        """Frozen covariate count for a variant, or None when that axis is dynamic.

        Args:
            variant: The variant.

        Returns:
            `static_covariates` for a static variant, else None.
        """
        return self.static_covariates if variant.static else None
