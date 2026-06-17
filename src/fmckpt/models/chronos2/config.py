# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Declarative configuration for the Chronos-2 export matrix.

Light (pydantic + constants, no torch), so the CLI can list variants and resolve
sizing without the heavy source-model stack. Holds the published sizes and their
HuggingFace targets, the ``{dynamic, static} x {fp32, fp16, int8}`` variant matrix,
window sizing (expressed in days at a resolution), and the FP16 op-block-list.
"""

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

#: The model card template for this family (shared across sizes).
CARD_TEMPLATE = Path(__file__).parent / "card.md.jinja"

#: ONNX opset used for export.
DEFAULT_OPSET = 17

#: Chronos-2's output patch length; the frozen horizon is a whole number of these.
#: The exported sidecar copies the value from the loaded model, so this is only the
#: sizing default used to derive the patch count from a horizon in days.
DEFAULT_OUTPUT_PATCH_SIZE = 16

#: Minutes per day, for converting day-sized windows to step counts.
_MINUTES_PER_DAY = 24 * 60

#: Ops kept at FP32 when converting to FP16. Chronos-2 is attention-heavy: softmax
#: materialises attention scores that overflow FP16 above ~11, and the NaN-aware
#: instance norm divides by a precision-sensitive per-series std. The matmul-heavy
#: layers stay FP16 — that is where the size/speed win comes from — so the graph is
#: *mixed* precision, not a wholesale downcast.
FP16_KEEP_FP32_OPS: tuple[str, ...] = ("Softmax", "Asinh", "Sinh", "ReduceMean", "ReduceSum", "Div", "Sqrt", "Pow")


class Variant(BaseModel):
    """One point in the export matrix: a precision and whether shapes are frozen."""

    model_config = ConfigDict(frozen=True)

    precision: Literal["fp32", "fp16", "int8"] = Field(description="Numeric precision of the variant's weights.")
    static: bool = Field(description="Whether all graph axes are frozen (CoreML-eligible).")
    publish: bool = Field(
        default=True,
        description="Whether to ship this variant. False = built and gated (for debugging) but never uploaded "
        "regardless of the deviation verdict — used for variants known-broken but kept in the pipeline.",
    )

    @property
    def suffix(self) -> str:
        """Filename suffix for this variant, e.g. ``_static_int8`` (fp32 adds none)."""
        precision_part = "" if self.precision == "fp32" else f"_{self.precision}"
        return f"{'_static' if self.static else ''}{precision_part}"

    @property
    def name(self) -> str:
        """Short identifier, e.g. ``fp32-static`` / ``int8-dynamic``."""
        return f"{self.precision}-{'static' if self.static else 'dynamic'}"


class Chronos2Model(BaseModel):
    """A published Chronos-2 size and how to export and ship it."""

    model_config = ConfigDict(frozen=True)

    #: The default export matrix, preferred-default first. int8-static is dropped (int8
    #: never reaches CoreML, and CPU/CUDA/TRT all take the dynamic graph). fp16 is built
    #: and gated for debugging but publish=False — it is the known-broken variant, and the
    #: deviation gate alone cannot be trusted to withhold it (it once false-passed).
    DEFAULT_VARIANTS: ClassVar[tuple[Variant, ...]] = (
        Variant(precision="fp32", static=True),  # zero-config default: CoreML-eligible + portable
        Variant(precision="fp32", static=False),  # CPU / CUDA / TensorRT; variable shapes
        Variant(precision="int8", static=False),  # size, CPU
        Variant(precision="fp16", static=True, publish=False),  # build-only until the FP16 bug is fixed
        Variant(precision="fp16", static=False, publish=False),
    )

    slug: str = Field(description="Filename/identity slug, e.g. 'chronos-2'.")
    source_model_id: str = Field(description="Upstream HuggingFace model id to export, e.g. 'amazon/chronos-2'.")
    repo_id: str = Field(description="Target HuggingFace repo this model's variants publish to.")
    source_license: str = Field(
        default="apache-2.0",
        description="License of the upstream weights (governs the published checkpoint, a derivative). The export "
        "tooling is MPL-2.0, but the model card's license is the weights' license.",
    )
    context_days: int = Field(gt=0, default=60, description="Context window in days (clamped to the model max).")
    horizon_days: int = Field(gt=0, default=7, description="Forecast horizon in days (frozen, rounded up to a patch).")
    resolution_minutes: int = Field(gt=0, default=15, description="Data resolution in minutes.")
    static_covariates: int = Field(
        gt=0,
        default=3,
        description="Covariate rows baked into a static export; the frozen batch is 1 target + this many.",
    )

    @property
    def steps_per_day(self) -> int:
        """Number of timesteps in a day at this resolution.

        Returns:
            ``_MINUTES_PER_DAY // resolution_minutes``.

        Raises:
            ValueError: If the resolution does not divide a day evenly.
        """
        if _MINUTES_PER_DAY % self.resolution_minutes != 0:
            msg = f"resolution {self.resolution_minutes} min does not divide a day evenly"
            raise ValueError(msg)
        return _MINUTES_PER_DAY // self.resolution_minutes

    @property
    def context_length(self) -> int:
        """Requested context length in steps (``context_days x steps_per_day``)."""
        return self.context_days * self.steps_per_day

    @property
    def num_output_patches(self) -> int:
        """Output patches needed to cover the horizon (rounded up)."""
        horizon_steps = self.horizon_days * self.steps_per_day
        return -(-horizon_steps // DEFAULT_OUTPUT_PATCH_SIZE)

    def weights_name(self, variant: Variant) -> str:
        """Filename for *variant*'s weights, e.g. ``chronos-2_static_int8.onnx``.

        Args:
            variant: The matrix point to name.

        Returns:
            The ``.onnx`` filename for this size and variant.
        """
        return f"{self.slug}{variant.suffix}.onnx"

    def max_covariates(self, variant: Variant) -> int | None:
        """Frozen covariate count for *variant* (the static batch minus the target), else None.

        Args:
            variant: The matrix point.

        Returns:
            ``static_covariates`` for a static variant; ``None`` when the covariate
            axis is dynamic.
        """
        return self.static_covariates if variant.static else None


# The published sizes. repo_ids and the -small source id are placeholders pending the
# team's HuggingFace org/account decision (design doc 0002, open questions).
CHRONOS2 = Chronos2Model(slug="chronos-2", source_model_id="amazon/chronos-2", repo_id="OpenSTEF/chronos-2-onnx")
CHRONOS2_SMALL = Chronos2Model(
    slug="chronos-2-small",
    source_model_id="amazon/chronos-2-small",  # NOTE: confirm the upstream small-model id with the team
    repo_id="OpenSTEF/chronos-2-small-onnx",
)

#: Published Chronos-2 sizes, keyed by slug.
MODELS: dict[str, Chronos2Model] = {model.slug: model for model in (CHRONOS2, CHRONOS2_SMALL)}
