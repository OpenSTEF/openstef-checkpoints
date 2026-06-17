# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""The ``fmckpt`` command line: list / export / publish.

``list`` and ``publish`` are light (no torch); ``export`` lazily imports the Chronos
exporter, which needs the ``[chronos]`` extra — so the CLI is usable, and ``--help``
works, without that heavy stack installed.
"""

import logging
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from fmckpt.models.chronos2.config import CARD_TEMPLATE, MODELS, Chronos2Model
from fmckpt.publish import ExportProvenance, Manifest, VariantRecord, publish_repo, render_card

app = typer.Typer(help="Export, verify, and publish foundation-model ONNX checkpoints.", no_args_is_help=True)
console = Console()


@app.callback()
def _configure() -> None:
    """Configure logging for the CLI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _model(slug: str) -> Chronos2Model:
    """Resolve a model config by slug, or exit with the known slugs.

    Returns:
        The matching model config.

    Raises:
        Exit: If *slug* is not a known model.
    """
    if slug not in MODELS:
        console.print(f"[red]Unknown model {slug!r}[/]. Known: {', '.join(MODELS)}")
        raise typer.Exit(code=1)
    return MODELS[slug]


@app.command("list")
def list_variants() -> None:
    """List the published models and their variant matrix."""
    table = Table(title="Published checkpoints")
    for column in ("Model", "Repo", "Variant file", "Precision", "Static"):
        table.add_column(column)
    for model in MODELS.values():
        for variant in Chronos2Model.DEFAULT_VARIANTS:
            table.add_row(
                model.slug,
                model.repo_id,
                model.weights_name(variant),
                variant.precision,
                "yes" if variant.static else "no",
            )
    console.print(table)


@app.command()
def export(
    model: Annotated[str, typer.Argument(help="Model slug, e.g. 'chronos-2'.")],
    out: Annotated[Path, typer.Option(help="Output directory for weights, sidecars and manifest.")] = Path("artifacts"),
    atol: Annotated[float, typer.Option(help="Absolute deviation tolerance.")] = 5e-2,
    rtol: Annotated[float, typer.Option(help="Relative deviation tolerance.")] = 1e-3,
) -> None:
    """Export a model's variant matrix and verify each against the torch reference."""
    # Lazy import: needs the [chronos] extra; keeps the CLI importable without torch.
    from fmckpt.models.chronos2.export import export_and_verify  # noqa: PLC0415

    config = _model(model)
    out.mkdir(parents=True, exist_ok=True)
    results = export_and_verify(config, out_dir=out, atol=atol, rtol=rtol)

    records = [
        VariantRecord(
            filename=checkpoint.weights_path.name,
            precision=checkpoint.metadata.precision,
            static_shapes=checkpoint.metadata.static_shapes,
            max_abs=deviation.max_abs,
            within_tolerance=deviation.within_tolerance,
        )
        for checkpoint, deviation in results
    ]
    provenance = ExportProvenance.capture(
        source_model_id=config.source_model_id,
        exporter_revision=os.environ.get("GITHUB_SHA", "unknown"),
    )
    Manifest(slug=config.slug, repo_id=config.repo_id, provenance=provenance, variants=records).write(out)
    _print_results(records)


@app.command()
def publish(
    model: Annotated[str, typer.Argument(help="Model slug, e.g. 'chronos-2'.")],
    out: Annotated[Path, typer.Option(help="Directory holding the exported artifacts + manifest.")] = Path("artifacts"),
    repo_id: Annotated[
        str | None, typer.Option(help="Override the target repo (e.g. your personal repo for testing).")
    ] = None,
    private: Annotated[bool, typer.Option(help="Create the repo private.")] = True,
    force: Annotated[bool, typer.Option(help="Publish even if some variants failed the deviation gate.")] = False,
) -> None:
    """Render the model card and upload the exported variants to HuggingFace.

    Raises:
        Exit: If some variants failed the deviation gate and ``--force`` was not given.
    """
    config = _model(model)
    manifest = Manifest.read(out)
    failing = [variant.filename for variant in manifest.variants if not variant.within_tolerance]
    if failing and not force:
        console.print(f"[red]Refusing to publish: {len(failing)} variant(s) failed the gate[/]: {', '.join(failing)}")
        console.print("Re-run with --force to publish anyway.")
        raise typer.Exit(code=1)

    (out / "README.md").write_text(render_card(CARD_TEMPLATE, manifest), encoding="utf-8")
    target = repo_id or config.repo_id
    console.print(f"Publishing {len(manifest.variants)} variant(s) to [bold]{target}[/] (private={private}) ...")
    url = publish_repo(target, out, private=private)
    console.print(f"[green]Published[/] {url}")


def _print_results(records: list[VariantRecord]) -> None:
    """Print the export results, flagging variants that failed the deviation gate."""
    table = Table(title="Export results")
    for column in ("Variant file", "Precision", "Static", "Max abs dev", "Gate"):
        table.add_column(column)
    for record in records:
        table.add_row(
            record.filename,
            record.precision,
            "yes" if record.static_shapes else "no",
            f"{record.max_abs:.4g}",
            "[green]pass[/]" if record.within_tolerance else "[red]FAIL[/]",
        )
    console.print(table)
