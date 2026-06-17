# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Chronos-2 export (needs torch + chronos): the model-specific half of the pipeline.

The torch wrapper flattening Chronos-2's dict/object interface to positional tensors,
the symbolic decompositions the TorchScript exporter needs, the representative inputs
and torch reference that feed the (generic) deviation gate, and the orchestration
that exports the FP32 bases, derives the FP16/INT8 variants, and verifies each.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, override

import numpy as np
import torch
from chronos import Chronos2Pipeline
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.onnx import register_custom_op_symbolic, symbolic_helper

from fmckpt.checkpoint import CheckpointMetadata, ExportedCheckpoint
from fmckpt.export import export_module, quantize_int8, to_fp16
from fmckpt.models.chronos2.config import DEFAULT_OPSET, FP16_KEEP_FP32_OPS, Chronos2Model, Variant
from fmckpt.verify import DeviationReport, compare_outputs, inject_nan_gaps, run_onnx, synthetic_series

logger = logging.getLogger(__name__)

#: ONNX IO names, in lock-step with `Chronos2OnnxModule.forward` and the sidecar.
INPUT_NAMES = ["context", "group_ids", "attention_mask", "future_covariates", "future_covariates_mask"]
OUTPUT_NAME = "quantile_preds"
_ONNX_FLOAT = 1  # TensorProto.FLOAT


class Chronos2OnnxModule(nn.Module):
    """Flatten Chronos-2's dict-in/object-out interface to positional tensors → tensor.

    Covariates are extra series rows sharing a ``group_id`` with their target: a
    covariate carries its history in ``context`` and its known future in
    ``future_covariates`` (mask 1); a target masks its future out (mask 0). The model
    normalises context internally (NaN-aware instance norm), so the graph owns scaling.
    """

    def __init__(self, model: nn.Module, num_output_patches: int) -> None:
        """Wrap *model* with a fixed output-patch count.

        Args:
            model: The inner Chronos-2 model (``pipeline.model``).
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
            context: Per-series history ``(batch, context_length)``.
            group_ids: Group id per row (covariates share their target's id).
            attention_mask: Validity mask over the context.
            future_covariates: Known-future values per row ``(batch, horizon)``.
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


class _Plan(BaseModel):
    """Resolved sizing for an export run (clamped context, patch count, horizon)."""

    model_config = ConfigDict(frozen=True)

    context_length: int = Field(gt=0, description="Context length, clamped to the model maximum.")
    num_patches: int = Field(gt=0, description="Output patches emitted.")
    horizon: int = Field(gt=0, description="Horizon length in steps.")
    output_patch_size: int = Field(gt=0, description="Model's output patch size.")
    native_quantiles: list[float] = Field(min_length=1, description="Model's native quantile grid.")


def export_and_verify(
    model: Chronos2Model,
    *,
    out_dir: Path,
    variants: Sequence[Variant] = Chronos2Model.DEFAULT_VARIANTS,
    device: torch.device | None = None,
    atol: float = 5e-2,
    rtol: float = 1e-3,
) -> list[tuple[Variant, ExportedCheckpoint, DeviationReport]]:
    """Export the selected *variants* of *model* and verify each against the torch reference.

    Args:
        model: The Chronos-2 size to export.
        out_dir: Directory for the ``.onnx`` files and sidecars.
        variants: Variants to build; defaults to the full matrix. Only the FP32 bases
            for the static-nesses actually used are exported.
        device: Torch device; defaults to CUDA when available.
        atol: Absolute tolerance for the verdict (loose; reduced precision drifts).
        rtol: Relative tolerance for the verdict.

    Returns:
        One ``(variant, checkpoint, deviation)`` per variant; the caller decides what
        to publish (``variant.publish`` and the deviation verdict).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inner = _load_model(model.source_model_id, device)
    plan = _plan(model, inner)
    _register_symbolic_ops(DEFAULT_OPSET)
    wrapper = Chronos2OnnxModule(inner, num_output_patches=plan.num_patches).eval()

    bases = {
        static: _export_base(wrapper, model, plan, static=static, out_dir=out_dir)
        for static in sorted({variant.static for variant in variants})
    }
    results: list[tuple[Variant, ExportedCheckpoint, DeviationReport]] = []
    for variant in variants:
        weights = _materialise(variant, base=bases[variant.static], model=model, out_dir=out_dir)
        exported = ExportedCheckpoint(weights_path=weights, metadata=_metadata(model, variant, plan))
        exported.write_sidecar()
        deviation = _verify(inner, exported, variant, plan=plan, model=model, atol=atol, rtol=rtol)
        logger.info("%s: max_abs=%.4g within_tolerance=%s", weights.name, deviation.max_abs, deviation.within_tolerance)
        results.append((variant, exported, deviation))
    return results


def _load_model(source_model_id: str, device: torch.device) -> nn.Module:
    """Load a Chronos-2 pipeline and return its eval-mode inner model.

    Returns:
        The inner model; its ``chronos_config`` carries quantiles, patch size and max context.
    """
    logger.info("Loading Chronos-2 %r on %s", source_model_id, device)
    return Chronos2Pipeline.from_pretrained(source_model_id, device_map=str(device)).model.eval()


def _plan(model: Chronos2Model, inner: nn.Module) -> _Plan:
    """Resolve sizing from the config and the loaded model's chronos_config.

    Returns:
        The clamped context, patch count, horizon and the model's quantile grid.
    """
    cfg: Any = inner.chronos_config  # chronos-specific config object; dynamically typed
    context_length = min(model.context_length, int(cfg.context_length))
    if model.context_length > int(cfg.context_length):
        logger.warning("Context %d exceeds model max %d; clamping", model.context_length, int(cfg.context_length))
    return _Plan(
        context_length=context_length,
        num_patches=model.num_output_patches,
        horizon=model.num_output_patches * int(cfg.output_patch_size),
        output_patch_size=int(cfg.output_patch_size),
        native_quantiles=[float(q) for q in cfg.quantiles],
    )


def _representative_inputs(plan: _Plan, *, covariate_rows: int, seed: int) -> dict[str, NDArray[np.generic]]:
    """Build a batch of ``1 target + covariate_rows`` exercising the NaN and covariate paths.

    The target's history carries NaN gaps; covariate rows carry a known future (mask 1)
    while the target's future is masked out (mask 0).

    Returns:
        Named arrays matching :data:`INPUT_NAMES`.
    """
    batch = 1 + covariate_rows
    rows = [synthetic_series(plan.context_length, seed=seed + r) for r in range(batch)]
    rows[0] = inject_nan_gaps(rows[0], gaps=2, gap_length=max(plan.context_length // 20, 1), seed=seed)
    context = np.stack(rows).astype(np.float32)
    future = np.zeros((batch, plan.horizon), dtype=np.float32)
    future_mask = np.zeros((batch, plan.horizon), dtype=np.float32)
    for r in range(1, batch):
        future[r] = synthetic_series(plan.horizon, seed=seed + 100 + r)
        future_mask[r] = 1.0
    return {
        "context": context,
        "group_ids": np.arange(batch, dtype=np.int64),
        # Missing history is masked out (0 at the NaN gaps): the NaN-aware norm still
        # sees the gaps in its statistics, but attention ignores them, so they do not
        # propagate NaN to the output.
        "attention_mask": np.isfinite(context).astype(np.float32),
        "future_covariates": future,
        "future_covariates_mask": future_mask,
    }


def _export_base(
    wrapper: Chronos2OnnxModule, model: Chronos2Model, plan: _Plan, *, static: bool, out_dir: Path
) -> Path:
    """Export the FP32 base graph for one static-ness.

    Returns:
        The path to the exported base ``.onnx``.
    """
    inputs = _representative_inputs(plan, covariate_rows=model.static_covariates if static else 1, seed=0)
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
    dst = out_dir / f"{model.slug}{'_static' if static else ''}.onnx"
    return export_module(
        wrapper,
        example,
        input_names=INPUT_NAMES,
        output_names=[OUTPUT_NAME],
        dynamic_axes=axes,
        opset=DEFAULT_OPSET,
        dst=dst,
    )


def _materialise(variant: Variant, *, base: Path, model: Chronos2Model, out_dir: Path) -> Path:
    """Produce a variant's weights from its FP32 base (identity for fp32).

    Returns:
        The path to the variant's weights file.
    """
    if variant.precision == "fp32":
        return base
    dst = out_dir / model.weights_name(variant)
    if variant.precision == "fp16":
        return to_fp16(base, dst, op_block_list=list(FP16_KEEP_FP32_OPS))
    return quantize_int8(base, dst)


def _metadata(model: Chronos2Model, variant: Variant, plan: _Plan) -> CheckpointMetadata:
    """Build the sidecar metadata for one variant.

    Returns:
        The metadata describing this variant's checkpoint.
    """
    return CheckpointMetadata(
        model_family="chronos2",
        input_names=INPUT_NAMES,
        output_name=OUTPUT_NAME,
        native_quantiles=plan.native_quantiles,
        context_length=plan.context_length,
        output_patch_size=plan.output_patch_size,
        horizon_patches=plan.num_patches,
        resolution_minutes=model.resolution_minutes,
        precision=variant.precision,
        static_shapes=variant.static,
        max_covariates=model.max_covariates(variant),
    )


def _verify(
    inner: nn.Module,
    exported: ExportedCheckpoint,
    variant: Variant,
    *,
    plan: _Plan,
    model: Chronos2Model,
    atol: float,
    rtol: float,
) -> DeviationReport:
    """Run the deviation gate for one variant on representative inputs (matching its frozen batch).

    Returns:
        The deviation of the variant's ONNX output from the torch reference.
    """
    inputs = _representative_inputs(plan, covariate_rows=model.static_covariates if variant.static else 1, seed=1)
    tensors = {name: torch.from_numpy(np.asarray(inputs[name])) for name in INPUT_NAMES}
    with torch.no_grad():
        reference = (
            Chronos2OnnxModule(inner, num_output_patches=plan.num_patches)
            .eval()(
                tensors["context"],
                tensors["group_ids"],
                tensors["attention_mask"],
                tensors["future_covariates"],
                tensors["future_covariates_mask"],
            )
            .cpu()
            .numpy()
        )
    return compare_outputs(reference, run_onnx(exported.weights_path, inputs), atol=atol, rtol=rtol)
