# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for publishing settings."""

import pytest

from openstef_checkpoints.settings import Settings


def test_repo_id_uses_default_namespace() -> None:
    """A model's repo id is the slug under the default namespace, suffixed -onnx."""
    assert Settings().repo_id("chronos-2") == "OpenSTEF/chronos-2-onnx"


def test_repo_id_follows_namespace_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """The namespace comes from the environment, so the target is redirectable."""
    monkeypatch.setenv("OPENSTEF_CHECKPOINTS_HF_NAMESPACE", "my-user")
    assert Settings().repo_id("chronos-2") == "my-user/chronos-2-onnx"
