# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the CLI context and dependency injection."""

import pytest
import typer
from typer.testing import CliRunner

from openstef_checkpoints.cli import CliContext, app
from openstef_checkpoints.models.registry import MODELS
from openstef_checkpoints.settings import Settings

runner = CliRunner()


def _context(**overrides: object) -> CliContext:
    return CliContext(settings=Settings(), models=MODELS, **overrides)


def test_model_resolves_a_known_slug() -> None:
    """A known slug resolves to its model config."""
    assert _context().model("chronos-2").source_model_id == "amazon/chronos-2"


def test_model_exits_on_unknown_slug() -> None:
    """An unknown slug exits rather than returning None."""
    with pytest.raises(typer.Exit):
        _context().model("does-not-exist")


def test_commands_use_the_injected_context() -> None:
    """A command resolves models from the injected context, not a module global."""
    empty = CliContext(settings=Settings(), models={})
    result = runner.invoke(app, ["publish", "chronos-2"], obj=empty)
    assert result.exit_code == 1
    assert "Unknown model" in result.output


def test_list_runs_with_an_injected_namespace() -> None:
    """`list` renders using the injected settings without touching the network."""
    ctx = CliContext(settings=Settings(hf_namespace="acme"), models=MODELS)
    result = runner.invoke(app, ["list"], obj=ctx)
    assert result.exit_code == 0
    assert "chronos-2" in result.output
