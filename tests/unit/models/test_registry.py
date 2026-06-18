# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the preconfigured-model registry."""

from openstef_checkpoints.models.registry import CHRONOS2, MODELS


def test_both_sizes_registered_by_slug() -> None:
    """Both Chronos-2 sizes are addressable by slug."""
    assert set(MODELS) == {"chronos-2", "chronos-2-small"}


def test_entry_points_at_its_upstream_model() -> None:
    """A registry entry carries the upstream id it exports from."""
    assert CHRONOS2.source_model_id == "amazon/chronos-2"
