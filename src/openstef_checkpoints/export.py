# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Model-agnostic export machinery: torch → ONNX, and reduced-precision passes.

The generic steps of turning a traced module into the ONNX variant matrix — the
``torch.onnx.export`` driver, the int64 ``Gather``-index fix, FP16 conversion, and
INT8 quantization — independent of which model is exported. Pulls torch (the
``[chronos]`` extra), so the light verify/publish/CLI paths do not import it.
"""

import logging
from pathlib import Path

import onnx
import torch
from onnx import TensorProto, helper
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_dynamic, quantize_static
from onnxruntime.transformers.float16 import convert_float_to_float16
from torch import nn

logger = logging.getLogger(__name__)

_GATHER_MIN_INPUTS = 2  # Gather is (data, indices)


def export_module(
    module: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    *,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
    dst: Path,
) -> Path:
    """Trace and export *module* to ONNX at *dst*, then fix Gather indices.

    Uses the legacy TorchScript exporter (``dynamo=False``); the caller registers
    any symbolic ops the tracer needs.

    Args:
        module: The eval-mode module to export.
        example_inputs: Positional example tensors matching ``module.forward``.
        input_names: ONNX input names, ordered like *example_inputs*.
        output_names: ONNX output names.
        dynamic_axes: Per-tensor dynamic-axis map; empty for a static export.
        opset: ONNX opset version.
        dst: Path to write to (parent dirs created).

    Returns:
        The path written (``dst``).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting %s (opset %d, %d dynamic tensors)", dst.name, opset, len(dynamic_axes))
    with torch.no_grad():
        torch.onnx.export(
            module,
            example_inputs,
            str(dst),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )
    _fix_gather_indices(dst)
    return dst


def to_fp16(src: Path, dst: Path, *, op_block_list: list[str]) -> Path:
    """Write a mixed-precision FP16 copy of *src* to *dst*.

    FP32 weights/intermediates become FP16 while *op_block_list* ops stay FP32
    (overflow-/precision-sensitive ops). ``keep_io_types`` keeps IO FP32 so the
    consumer is unchanged. Uses ONNX Runtime's converter (the onnxconverter_common
    one crashes on attention graphs).

    Args:
        src: FP32 model path.
        dst: Output path.
        op_block_list: Op types kept at FP32.

    Returns:
        The path written (``dst``).
    """
    model = convert_float_to_float16(
        onnx.load(str(src)),
        keep_io_types=True,
        op_block_list=op_block_list,
        disable_shape_infer=False,
    )
    _drop_duplicate_casts(model)
    onnx.save(model, str(dst))
    _log_size(src, dst)
    return dst


def quantize_int8(src: Path, dst: Path, *, calibration: CalibrationDataReader | None = None) -> Path:
    """Write an INT8 copy of *src* to *dst* (dynamic, or static with *calibration*).

    Args:
        src: FP32 model path.
        dst: Output path.
        calibration: Reader enabling static activation calibration; dynamic if None.

    Returns:
        The path written (``dst``).
    """
    if calibration is not None:
        quantize_static(str(src), str(dst), calibration_data_reader=calibration, weight_type=QuantType.QInt8)
    else:
        quantize_dynamic(model_input=str(src), model_output=str(dst), weight_type=QuantType.QInt8)
    _log_size(src, dst)
    return dst


def _fix_gather_indices(onnx_path: Path) -> int:
    """Cast float ``Gather`` indices to int64 in place; return the count patched.

    The TorchScript exporter can feed float indices into ``Gather``, but ONNX Runtime
    requires int64; a ``Cast`` is inserted before each such ``Gather``, preserving
    topological order.

    Args:
        onnx_path: ONNX file to patch in place.

    Returns:
        The number of ``Cast`` nodes inserted.
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph
    new_nodes = []
    inserts = 0
    for node in graph.node:
        if node.op_type == "Gather" and len(node.input) >= _GATHER_MIN_INPUTS:
            cast_out = f"{node.input[1]}_int64_{inserts}"
            new_nodes.append(
                helper.make_node(
                    "Cast", [node.input[1]], [cast_out], to=TensorProto.INT64, name=f"cast_gather_{inserts}"
                )
            )
            node.input[1] = cast_out
            inserts += 1
        new_nodes.append(node)
    if inserts:
        del graph.node[:]
        graph.node.extend(new_nodes)
        onnx.save(model, str(onnx_path))
        logger.info("Gather fix: inserted %d Cast(float->int64) node(s)", inserts)
    return inserts


def _drop_duplicate_casts(model: onnx.ModelProto) -> int:
    """Drop Cast nodes re-producing an already-produced tensor; return the count.

    ORT's FP16 converter emits one identical ``_cast_to_fp32`` Cast per consumer of a
    blocked op's output; the duplicates (same input and output name) are rejected by
    ORT, so all but the first are dropped.

    Args:
        model: Model to clean in place.

    Returns:
        The number of nodes dropped.
    """
    produced: set[str] = set()
    kept = []
    dropped = 0
    for node in model.graph.node:
        if node.output and node.output[0] in produced:
            dropped += 1
            continue
        kept.append(node)
        produced.update(node.output)
    if dropped:
        del model.graph.node[:]
        model.graph.node.extend(kept)
        logger.info("Dropped %d duplicate Cast node(s) after FP16 conversion", dropped)
    return dropped


def _log_size(src: Path, dst: Path) -> None:
    """Log the file-size reduction from *src* to *dst*."""
    src_mb, dst_mb = src.stat().st_size / 1024**2, dst.stat().st_size / 1024**2
    logger.info("Size: %.1f MB -> %.1f MB (%.0f%% smaller)", src_mb, dst_mb, (1 - dst_mb / src_mb) * 100)
