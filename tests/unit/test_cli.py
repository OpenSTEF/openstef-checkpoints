# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the CLI context and dependency injection."""

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from openstef_checkpoints import cli as cli_module
from openstef_checkpoints.cli import CliContext, app
from openstef_checkpoints.models.registry import MODELS
from openstef_checkpoints.publish import ExportProvenance, ExportWindow, Manifest, VariantRecord
from openstef_checkpoints.settings import Settings

runner = CliRunner()


def _context(**overrides: object) -> CliContext:
    return CliContext(settings=Settings(), models=MODELS, **overrides)


def _write_manifest(model_dir: Path) -> None:
    """Write a manifest with one publishable fp32 variant into <model_dir>."""
    model_dir.mkdir(parents=True, exist_ok=True)
    Manifest(
        slug="chronos-2",
        repo_id="OpenSTEF/chronos-2-onnx",
        provenance=ExportProvenance(
            source_model_id="amazon/chronos-2",
            source_revision="abc",
            exporter_revision="def",
            tooling="onnx=1.17",
            exported_at="2026-06-17T00:00:00+00:00",
        ),
        window=ExportWindow(context_length=5760, horizon_length=672, resolution_minutes=15, static_batch=4),
        variants=[
            VariantRecord(
                filename="chronos-2.onnx",
                precision="fp32",
                static_shapes=False,
                max_abs=2e-5,
                within_tolerance=True,
                publish=True,
            )
        ],
    ).write(model_dir)


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


def test_publish_dry_run_validates_without_uploading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--dry-run runs the gate and reports the intended uploads, but never calls publish_repo."""

    def _fail_if_called(*_args: object, **_kwargs: object) -> str:
        msg = "publish_repo must not be called during a dry run"
        raise AssertionError(msg)

    monkeypatch.setattr(cli_module, "publish_repo", _fail_if_called)
    _write_manifest(tmp_path / "chronos-2")

    result = runner.invoke(app, ["publish", "chronos-2", "--out", str(tmp_path), "--dry-run"], obj=_context())

    assert result.exit_code == 0
    assert "dry-run" in result.output
    assert "chronos-2.onnx" in result.output
