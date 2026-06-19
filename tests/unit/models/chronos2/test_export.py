# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the Chronos-2 exporter's model repair step.

These need torch and chronos (the `[chronos]` extra), so they skip when it is absent.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chronos")

from openstef_checkpoints.models.chronos2.export import Chronos2Exporter  # noqa: E402


class _FakeRoPE(torch.nn.Module):
    """A stand-in for Chronos-2's RoPE: the `dim`/`base`/`inv_freq` trio the repair targets."""

    def __init__(self, dim: int, base: float, garbage: torch.Tensor) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        # Non-persistent, as upstream registers it; seeded with garbage to mimic the
        # uninitialised buffer from_pretrained leaves behind.
        self.register_buffer("inv_freq", garbage, persistent=False)


def _expected_inv_freq(dim: int, base: float) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))


def test_repair_recomputes_garbage_rope_buffer() -> None:
    """A RoPE module whose inv_freq is NaN garbage is restored to the correct frequencies."""
    dim, base = 64, 10000.0
    rope = _FakeRoPE(dim, base, garbage=torch.full((dim // 2,), float("nan")))
    parent = torch.nn.Sequential(rope)

    Chronos2Exporter._repair_rope_buffers(parent)

    assert not torch.isnan(rope.inv_freq).any()
    assert torch.allclose(rope.inv_freq, _expected_inv_freq(dim, base))


def test_repair_leaves_unrelated_modules_untouched() -> None:
    """A module without the dim/base/inv_freq trio is not modified."""
    linear = torch.nn.Linear(4, 4)
    before = linear.weight.detach().clone()

    Chronos2Exporter._repair_rope_buffers(linear)

    assert torch.equal(linear.weight, before)
