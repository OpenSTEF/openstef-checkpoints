# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""The Chronos-2 side of the export: everything specific to this model.

Wraps Chronos-2 to take and return plain tensors, supplies the ONNX operators the tracer
lacks, builds the inputs and torch reference used to check each variant, and runs the
export (fp32 first, then the fp16 and int8 variants derived from it). Needs torch and
chronos, so it is imported only when an export runs.
"""

import logging
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Protocol, cast, override

import numpy as np
import torch
from chronos import Chronos2Pipeline
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.onnx import register_custom_op_symbolic, symbolic_helper

from openstef_checkpoints.checkpoint import CheckpointMetadata, ExportedCheckpoint
from openstef_checkpoints.graph.transforms import export_module, quantize_int8, to_fp16
from openstef_checkpoints.graph.verify import DeviationReport, inject_nan_gaps, run_onnx, synthetic_series
from openstef_checkpoints.models.chronos2.config import Chronos2Model, Variant
from openstef_checkpoints.publish import VariantRecord

logger = logging.getLogger(__name__)

#: ONNX input/output names, matching `Chronos2OnnxModule.forward` and the metadata file.
INPUT_NAMES = ["context", "group_ids", "attention_mask", "future_covariates", "future_covariates_mask"]
OUTPUT_NAME = "quantile_preds"
_ONNX_FLOAT = 1  # TensorProto.FLOAT


class Chronos2OnnxModule(nn.Module):
    """Wrap Chronos-2 so it takes plain tensors in and returns one tensor out.

    A covariate is an extra series row sharing a `group_id` with its target: it carries
    its history in `context` and its known future in `future_covariates`, with its mask
    set to 1, while a target's future is masked out with 0. The model normalises the
    context itself, with a NaN-aware instance norm, so the graph handles scaling.
    """

    def __init__(self, model: nn.Module, num_output_patches: int) -> None:
        """Wrap model with a fixed output-patch count.

        Args:
            model: The inner Chronos-2 model (`pipeline.model`).
            num_output_patches: Output patches to emit (the frozen horizon).
        """
        super().__init__()
        self.model = model
        self.num_output_patches = num_output_patches

    @override
    def forward(
        self,
        context: torch.Tensor,
        group_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        future_covariates: torch.Tensor,
        future_covariates_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the model and return only its quantile predictions.

        Args:
            context: Per-series history `(batch, context_length)`.
            group_ids: Group id per row (covariates share their target's id).
            attention_mask: Validity mask over the context.
            future_covariates: Known-future values per row `(batch, horizon)`.
            future_covariates_mask: Which future values are known (1) vs ignored (0).

        Returns:
            The quantile-prediction tensor.
        """
        return self.model(
            context=context,
            group_ids=group_ids,
            context_mask=attention_mask,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            num_output_patches=self.num_output_patches,
        ).quantile_preds


def _register_symbolic_ops(opset: int) -> None:
    """Lower aten ops the legacy exporter can't map: arcsinh, NaN-aware reductions, unfold."""

    def asinh(g: object, x: object) -> object:
        return g.op("Asinh", x)  # type: ignore

    def sinh(g: object, x: object) -> object:
        return g.op("Sinh", x)  # type: ignore

    def nan_filled(g: object, x: object) -> tuple[object, object]:
        is_nan = g.op("IsNaN", x)  # type: ignore
        zero = g.op("Constant", value_t=torch.tensor(0.0, dtype=torch.float32))  # type: ignore
        return g.op("Where", is_nan, zero, x), g.op("Not", is_nan)  # type: ignore

    @symbolic_helper.parse_args("v", "is", "i", "none")
    def nansum(g: object, x: object, dim: list[int], keepdim: int, _dtype: object) -> object:
        filled, _ = nan_filled(g, x)
        axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.int64))  # type: ignore
        return g.op("ReduceSum", filled, axes, keepdims_i=keepdim)  # type: ignore

    @symbolic_helper.parse_args("v", "is", "i", "none")
    def nanmean(g: object, x: object, dim: list[int], keepdim: int, _dtype: object) -> object:
        filled, not_nan = nan_filled(g, x)
        axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.int64))  # type: ignore
        total = g.op("ReduceSum", filled, axes, keepdims_i=keepdim)  # type: ignore
        count = g.op("ReduceSum", g.op("Cast", not_nan, to_i=_ONNX_FLOAT), axes, keepdims_i=keepdim)  # type: ignore
        return g.op("Div", total, count)  # type: ignore

    @symbolic_helper.parse_args("v", "i", "i", "i")
    def unfold(g: object, x: object, _dim: int, size: int, step: int) -> object:
        # Chronos patches with size == step on the last dim → reshape (..., L) -> (..., L//size, size).
        if size != step:
            msg = f"unfold supports only size == step (got {size}, {step})"
            raise NotImplementedError(msg)
        zero = g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64))  # type: ignore
        neg1 = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))  # type: ignore
        size1d = g.op("Constant", value_t=torch.tensor([size], dtype=torch.int64))  # type: ignore
        prefix = g.op("Slice", g.op("Shape", x), zero, neg1, zero)  # type: ignore
        return g.op("Reshape", x, g.op("Concat", prefix, neg1, size1d, axis_i=0))  # type: ignore

    for name, fn in (("asinh", asinh), ("sinh", sinh), ("nansum", nansum), ("nanmean", nanmean), ("unfold", unfold)):
        register_custom_op_symbolic(f"aten::{name}", fn, opset)


class _ChronosConfig(Protocol):
    """The fields the exporter reads off a loaded model's native `chronos_config`."""

    context_length: int
    output_patch_size: int
    quantiles: Sequence[float]


class VariantResult(BaseModel):
    """One built variant: its spec, the exported checkpoint, and the deviation found."""

    model_config = ConfigDict(frozen=True)

    variant: Variant = Field(description="The variant that was built.")
    checkpoint: ExportedCheckpoint = Field(description="The exported weights and their metadata.")
    deviation: DeviationReport = Field(description="How far the output drifted from the torch reference.")

    def to_record(self) -> VariantRecord:
        """Summarise this result as a manifest record.

        Returns:
            The serialisable manifest entry for this variant.
        """
        return VariantRecord(
            filename=self.checkpoint.weights_path.name,
            precision=self.checkpoint.metadata.precision,
            static_shapes=self.checkpoint.metadata.static_shapes,
            max_abs=self.deviation.max_abs,
            within_tolerance=self.deviation.within_tolerance,
            publish=self.variant.publish,
        )


class Chronos2Exporter(BaseModel):
    """Export and verify the ONNX variants of one Chronos-2 model.

    Construct it with the model and tolerances, then call `run`. The torch model loads
    lazily on first use, so constructing an exporter is cheap and side-effect-free; the
    load, export, and verification all happen in `run`. The loaded model, the resolved
    plan, and the traced wrapper are shared across the variants of a run.
    """

    model_config = ConfigDict(frozen=True)

    #: ONNX opset used for the export.
    OPSET: ClassVar[int] = 17

    #: Ops kept at fp32 when converting to fp16. Chronos-2 is attention-heavy: softmax
    #: scores overflow fp16, and the NaN-aware instance norm divides by a precision-sensitive
    #: standard deviation. The matmul-heavy layers stay fp16, where the size and speed gains
    #: are, so the graph ends up mixed precision rather than a full downcast.
    FP16_KEEP_FP32_OPS: ClassVar[tuple[str, ...]] = (
        "Softmax",
        "Asinh",
        "Sinh",
        "ReduceMean",
        "ReduceSum",
        "Div",
        "Sqrt",
        "Pow",
    )

    model: Chronos2Model = Field(description="The Chronos-2 size to export.")
    out_dir: Path = Field(description="Directory for the `.onnx` files and their metadata files.")
    device: str | None = Field(
        default=None, description="Torch device, e.g. 'cpu' or 'cuda'. Defaults to CUDA when available."
    )
    atol: float = Field(
        default=5e-2, gt=0, description="Absolute deviation tolerance, kept loose since reduced precision drifts."
    )
    rtol: float = Field(default=1e-3, gt=0, description="Relative deviation tolerance.")

    @cached_property
    def _device(self) -> torch.device:
        """Resolve the requested device, defaulting to CUDA when available."""
        return torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    @cached_property
    def _inner(self) -> nn.Module:
        """Load the Chronos-2 pipeline and return its eval-mode inner model.

        Its `chronos_config` carries the quantiles, patch size, and max context the plan reads.
        """
        logger.info("Loading Chronos-2 %r on %s", self.model.source_model_id, self._device)
        return Chronos2Pipeline.from_pretrained(self.model.source_model_id, device_map=str(self._device)).model.eval()

    @cached_property
    def _chronos_config(self) -> _ChronosConfig:
        """The loaded model's native config: quantile grid, patch size, max context.

        The single seam to the upstream `chronos_config`; the sizing properties read it through here.
        """
        return cast("_ChronosConfig", self._inner.chronos_config)

    @cached_property
    def _context_length(self) -> int:
        """Context length in steps, clamped to the model's maximum."""
        model_max = int(self._chronos_config.context_length)
        if self.model.context_length > model_max:
            logger.warning("Context %d exceeds model max %d; clamping", self.model.context_length, model_max)
        return min(self.model.context_length, model_max)

    @property
    def _output_patch_size(self) -> int:
        """The model's output patch length, in steps."""
        return int(self._chronos_config.output_patch_size)

    @property
    def _native_quantiles(self) -> list[float]:
        """The model's native quantile grid."""
        return [float(q) for q in self._chronos_config.quantiles]

    @property
    def _horizon(self) -> int:
        """Horizon length in steps: a whole number of output patches."""
        return self.model.num_output_patches * self._output_patch_size

    @cached_property
    def _wrapper(self) -> Chronos2OnnxModule:
        """The plain-tensor wrapper traced into ONNX, with the tracer's symbolic ops registered."""
        _register_symbolic_ops(self.OPSET)
        return Chronos2OnnxModule(self._inner, num_output_patches=self.model.num_output_patches).eval()

    def run(self, variants: Sequence[Variant] = Chronos2Model.DEFAULT_VARIANTS) -> list[VariantResult]:
        """Export and check each variant, reusing one fp32 base per static setting.

        Args:
            variants: Variants to build; defaults to the full matrix. Each fp32 base is
                exported once and shared by the variants derived from it.

        Returns:
            One result per variant, in the order given. The caller decides what to publish,
            from `variant.publish` and the deviation verdict.
        """
        bases = {static: self._export_base(static=static) for static in sorted({v.static for v in variants})}
        results: list[VariantResult] = []
        for variant in variants:
            weights = self._materialise(variant, base=bases[variant.static])
            checkpoint = ExportedCheckpoint(weights_path=weights, metadata=self._metadata(variant))
            checkpoint.write_metadata()
            deviation = self._verify(checkpoint, variant)
            logger.info(
                "%s: max_abs=%.4g within_tolerance=%s", weights.name, deviation.max_abs, deviation.within_tolerance
            )
            results.append(VariantResult(variant=variant, checkpoint=checkpoint, deviation=deviation))
        return results

    def _representative_inputs(self, *, covariate_rows: int, seed: int) -> dict[str, NDArray[np.generic]]:
        """Build a batch of one target plus `covariate_rows` covariates.

        The target's history carries NaN gaps; each covariate carries a known future
        (mask 1) while the target's future is masked out (mask 0).

        Returns:
            Named arrays matching `INPUT_NAMES`.
        """
        batch = 1 + covariate_rows
        rows = [synthetic_series(self._context_length, seed=seed + r) for r in range(batch)]
        rows[0] = inject_nan_gaps(rows[0], gaps=2, gap_length=max(self._context_length // 20, 1), seed=seed)
        context = np.stack(rows).astype(np.float32)
        # Record the gaps in the attention mask, then zero them out of the context — exactly
        # what the runtime forecaster's zero_fill_with_mask does. Feeding raw NaN instead
        # relies on every masked op suppressing it, which holds on some ONNX Runtime builds
        # but leaks NaN into the output on others (e.g. the Linux CI runner); the model only
        # ever sees finite values plus the mask.
        attention_mask = np.isfinite(context).astype(np.float32)
        context = np.nan_to_num(context, nan=0.0)
        future = np.zeros((batch, self._horizon), dtype=np.float32)
        future_mask = np.zeros((batch, self._horizon), dtype=np.float32)
        for r in range(1, batch):
            future[r] = synthetic_series(self._horizon, seed=seed + 100 + r)
            future_mask[r] = 1.0
        return {
            "context": context,
            "group_ids": np.arange(batch, dtype=np.int64),
            "attention_mask": attention_mask,
            "future_covariates": future,
            "future_covariates_mask": future_mask,
        }

    def _export_base(self, *, static: bool) -> Path:
        """Export the fp32 base graph for static or dynamic shapes.

        Returns:
            The path to the exported base `.onnx`.
        """
        inputs = self._representative_inputs(covariate_rows=self.model.static_covariates if static else 1, seed=0)
        example = tuple(torch.from_numpy(np.asarray(inputs[name])) for name in INPUT_NAMES)
        axes: dict[str, dict[int, str]] = (
            {}
            if static
            else {
                "context": {0: "batch", 1: "context_length"},
                "group_ids": {0: "batch"},
                "attention_mask": {0: "batch", 1: "context_length"},
                "future_covariates": {0: "batch", 1: "future_length"},
                "future_covariates_mask": {0: "batch", 1: "future_length"},
                OUTPUT_NAME: {0: "batch"},
            }
        )
        dst = self.out_dir / f"{self.model.slug}{'_static' if static else ''}.onnx"
        return export_module(
            self._wrapper,
            example,
            input_names=INPUT_NAMES,
            output_names=[OUTPUT_NAME],
            dynamic_axes=axes,
            opset=self.OPSET,
            dst=dst,
        )

    def _materialise(self, variant: Variant, *, base: Path) -> Path:
        """Produce a variant's weights from its fp32 base (the base itself for fp32).

        Returns:
            The path to the variant's weights file.
        """
        if variant.precision == "fp32":
            return base
        dst = self.out_dir / self.model.weights_name(variant)
        if variant.precision == "fp16":
            return to_fp16(base, dst, op_block_list=list(self.FP16_KEEP_FP32_OPS))
        return quantize_int8(base, dst)

    def _metadata(self, variant: Variant) -> CheckpointMetadata:
        """Build the metadata for one variant.

        Returns:
            The metadata describing this variant's checkpoint.
        """
        return CheckpointMetadata(
            model_family="chronos2",
            input_names=INPUT_NAMES,
            output_name=OUTPUT_NAME,
            native_quantiles=self._native_quantiles,
            context_length=self._context_length,
            output_patch_size=self._output_patch_size,
            horizon_patches=self.model.num_output_patches,
            resolution_minutes=self.model.resolution_minutes,
            precision=variant.precision,
            static_shapes=variant.static,
            max_covariates=self.model.max_covariates(variant),
        )

    def _verify(self, checkpoint: ExportedCheckpoint, variant: Variant) -> DeviationReport:
        """Compare the variant's ONNX output to the torch reference on representative inputs.

        Returns:
            The deviation between the two.
        """
        inputs = self._representative_inputs(
            covariate_rows=self.model.static_covariates if variant.static else 1, seed=1
        )
        tensors = {name: torch.from_numpy(np.asarray(inputs[name])) for name in INPUT_NAMES}
        with torch.no_grad():
            reference = (
                self
                ._wrapper(
                    tensors["context"],
                    tensors["group_ids"],
                    tensors["attention_mask"],
                    tensors["future_covariates"],
                    tensors["future_covariates_mask"],
                )
                .cpu()
                .numpy()
            )
        return DeviationReport.compare(
            reference, run_onnx(checkpoint.weights_path, inputs), atol=self.atol, rtol=self.rtol
        )
