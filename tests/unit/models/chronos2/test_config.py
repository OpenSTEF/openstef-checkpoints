# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the Chronos-2 export configuration and sizing."""

import pytest

from fmckpt.models.chronos2.config import CHRONOS2, MODELS, Chronos2Model, Variant


def test_window_sizing_from_days() -> None:
    """Context and horizon convert from days to steps/patches at the configured resolution."""
    model = Chronos2Model(
        slug="x",
        source_model_id="amazon/x",
        repo_id="OpenSTEF/x",
        context_days=60,
        horizon_days=7,
        resolution_minutes=15,
    )
    assert model.steps_per_day == 96
    assert model.context_length == 60 * 96
    # 7 days x 96 = 672 steps; 672 / 16 = 42 patches.
    assert model.num_output_patches == 42


def test_horizon_patches_round_up() -> None:
    """A horizon that is not a whole number of patches rounds up."""
    model = Chronos2Model(slug="x", source_model_id="a", repo_id="r", horizon_days=1, resolution_minutes=10)
    # 1 day x 144 steps = 144; 144 / 16 = 9 exactly.
    assert model.num_output_patches == 9


def test_resolution_must_divide_a_day() -> None:
    """A resolution that does not divide a day evenly is rejected at use."""
    model = Chronos2Model(slug="x", source_model_id="a", repo_id="r", resolution_minutes=7)
    with pytest.raises(ValueError, match="does not divide a day"):
        _ = model.steps_per_day


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        (Variant(precision="fp32", static=False), "chronos-2.onnx"),
        (Variant(precision="fp32", static=True), "chronos-2_static.onnx"),
        (Variant(precision="int8", static=True), "chronos-2_static_int8.onnx"),
        (Variant(precision="fp16", static=False), "chronos-2_fp16.onnx"),
    ],
)
def test_weights_name_encodes_variant(variant: Variant, expected: str) -> None:
    """The weights filename encodes static-ness and precision."""
    assert CHRONOS2.weights_name(variant) == expected


def test_max_covariates_only_for_static() -> None:
    """A static variant freezes the covariate count; a dynamic one leaves it None."""
    assert CHRONOS2.max_covariates(Variant(precision="fp32", static=True)) == CHRONOS2.static_covariates
    assert CHRONOS2.max_covariates(Variant(precision="fp32", static=False)) is None


def test_published_sizes_registered() -> None:
    """Both shipped sizes are addressable by slug."""
    assert set(MODELS) == {"chronos-2", "chronos-2-small"}
