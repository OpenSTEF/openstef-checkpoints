# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""The deviation gate — the generic side.

Decides whether an exported (possibly reduced-precision) graph still matches its
reference. The pieces here are model-agnostic: the numeric comparison, the raw ORT
run of the candidate, and realistic-series primitives. *Which* inputs a model feeds
and *how* its reference is run are model-specific and live with the model — a clean
dense max-diff once hid the FP16 bug because it skipped the NaN/covariate path, so
representativeness is the model's responsibility, exercised through these primitives.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Mapping

_REL_EPS = 1e-9  # guards the relative-error denominator


class DeviationReport(BaseModel):
    """How far a candidate graph's output deviates from the reference."""

    model_config = ConfigDict(frozen=True)

    max_abs: float = Field(description="Maximum absolute difference over all elements.")
    mean_abs: float = Field(description="Mean absolute difference over all elements.")
    rel_mean: float = Field(description="Mean absolute difference relative to the reference's mean magnitude.")
    rmse: float = Field(description="Root-mean-square difference over all elements.")
    within_tolerance: bool = Field(description="Whether the candidate is within the configured (atol, rtol).")


def compare_outputs(
    reference: NDArray[np.floating],
    candidate: NDArray[np.floating],
    *,
    atol: float,
    rtol: float,
) -> DeviationReport:
    """Compare a candidate's output against the reference, elementwise.

    Args:
        reference: Output of the trusted reference on some inputs.
        candidate: Output of the exported graph on the *same* inputs.
        atol: Absolute tolerance for the verdict.
        rtol: Relative tolerance for the verdict.

    Returns:
        The differences and pass/fail verdict.

    Raises:
        ValueError: If the outputs do not share a shape.
    """
    if reference.shape != candidate.shape:
        msg = f"output shape mismatch: reference={reference.shape} candidate={candidate.shape}"
        raise ValueError(msg)
    ref, cand = reference.astype(np.float64), candidate.astype(np.float64)
    abs_diff = np.abs(ref - cand)
    return DeviationReport(
        max_abs=float(abs_diff.max()),
        mean_abs=float(abs_diff.mean()),
        rel_mean=float(abs_diff.mean() / (np.abs(ref).mean() + _REL_EPS)),
        rmse=float(np.sqrt((abs_diff**2).mean())),
        within_tolerance=bool(np.allclose(ref, cand, atol=atol, rtol=rtol)),
    )


def run_onnx(onnx_path: Path, inputs: "Mapping[str, NDArray[np.generic]]") -> NDArray[np.floating]:
    """Run an ONNX graph on CPU and return its first output.

    CPU-only so the only difference from the reference is the graph's own precision.

    Args:
        onnx_path: Path to the ONNX weights file.
        inputs: Named input tensors matching the graph's input names.

    Returns:
        The graph's first output tensor.
    """
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return np.asarray(session.run(None, dict(inputs))[0])


def synthetic_series(length: int, *, seed: int) -> NDArray[np.float32]:
    """Generate one realistic series (trend + seasonal sine + noise).

    Real load-like structure exercises activation ranges and normalisation far better
    than white noise.

    Args:
        length: Number of timesteps.
        seed: Seed for reproducibility; vary it for distinct series.

    Returns:
        A 1-D ``float32`` array of shape ``(length,)``.
    """
    rng = np.random.default_rng(seed)
    trend = np.linspace(0.0, rng.uniform(-2.0, 2.0), length)
    season = np.sin(np.linspace(0.0, rng.uniform(4.0, 12.0) * np.pi, length))
    return (trend + season + rng.standard_normal(length) * 0.1).astype(np.float32)


def inject_nan_gaps(series: NDArray[np.float32], *, gaps: int, gap_length: int, seed: int) -> NDArray[np.float32]:
    """Return a copy of *series* with *gaps* runs of NaN (the NaN-aware path's input).

    Args:
        series: The 1-D series to copy and punch gaps into.
        gaps: Number of NaN runs.
        gap_length: Length of each run.
        seed: Seed for gap placement.

    Returns:
        A copy with NaN gaps; the original is untouched.
    """
    out = series.copy()
    if gaps <= 0 or gap_length <= 0:
        return out
    rng = np.random.default_rng(seed)
    for start in rng.integers(0, max(len(out) - gap_length, 1), size=gaps):
        out[start : start + gap_length] = np.nan
    return out
