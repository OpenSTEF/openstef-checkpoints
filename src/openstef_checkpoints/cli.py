# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""The `openstef-checkpoints` command line: list, export, publish.

`list` and `publish` are light; `export` imports the Chronos exporter only when it
runs, so the CLI and its `--help` work without the torch stack installed.

Commands read their dependencies (settings, the model registry) from a `CliContext`
on the Typer context, built in the callback. Tests inject their own with
`CliRunner().invoke(app, ..., obj=CliContext(...))`.
"""

import logging
import os
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.table import Table

from openstef_checkpoints.models.chronos2.config import Chronos2Model, Variant
from openstef_checkpoints.models.registry import MODELS
from openstef_checkpoints.publish import CARD_NAME, ExportProvenance, Manifest, VariantRecord, publish_repo, render_card
from openstef_checkpoints.settings import Settings

app = typer.Typer(help="Export, verify, and publish foundation-model ONNX checkpoints.", no_args_is_help=True)
console = Console()


class CliContext(BaseModel):
    """The dependencies a command needs: publishing settings and the model registry."""

    model_config = ConfigDict(frozen=True)

    settings: Settings = Field(description="Publishing settings.")
    models: dict[str, Chronos2Model] = Field(description="Exportable models, keyed by slug.")

    def model(self, slug: str) -> Chronos2Model:
        """Resolve a model by slug, or exit listing the known slugs.

        Returns:
            The matching model.

        Raises:
            Exit: If the slug is not a known model.
        """
        if slug not in self.models:
            console.print(f"[red]Unknown model {slug!r}[/]. Known: {', '.join(self.models)}")
            raise typer.Exit(code=1)
        return self.models[slug]


@app.callback()
def main(ctx: typer.Context) -> None:
    """Configure logging and provide the command dependencies."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if ctx.obj is None:
        ctx.obj = CliContext(settings=Settings(), models=MODELS)


def _select_variants(names: list[str] | None) -> list[Variant]:
    """Resolve variant names (e.g. `fp32-static`) to the matrix, or all if none given.

    Returns:
        The selected variants.

    Raises:
        Exit: If any name is not in the matrix.
    """
    by_name = {variant.name: variant for variant in Chronos2Model.DEFAULT_VARIANTS}
    if not names:
        return list(Chronos2Model.DEFAULT_VARIANTS)
    unknown = [name for name in names if name not in by_name]
    if unknown:
        console.print(f"[red]Unknown variant(s)[/]: {', '.join(unknown)}. Known: {', '.join(by_name)}")
        raise typer.Exit(code=1)
    return [by_name[name] for name in names]


@app.command("list")
def list_variants(ctx: typer.Context) -> None:
    """List the published models and their variant matrix."""
    cli: CliContext = ctx.obj
    table = Table(title="Published checkpoints")
    for column in ("Model", "Repo", "Variant file", "Precision", "Static"):
        table.add_column(column)
    for model in cli.models.values():
        for variant in Chronos2Model.DEFAULT_VARIANTS:
            table.add_row(
                model.slug,
                cli.settings.repo_id(model.slug),
                model.weights_name(variant),
                variant.precision,
                "yes" if variant.static else "no",
            )
    console.print(table)


@app.command()
def export(
    ctx: typer.Context,
    model: Annotated[str, typer.Argument(help="Model slug, e.g. 'chronos-2'.")],
    out: Annotated[Path, typer.Option(help="Base output directory; each model writes to <out>/<slug>.")] = Path(
        "checkpoints"
    ),
    variant: Annotated[
        list[str] | None, typer.Option(help="Variant(s) to build, e.g. fp32-static. Default: all.")
    ] = None,
    atol: Annotated[float, typer.Option(help="Absolute deviation tolerance.")] = 5e-2,
    rtol: Annotated[float, typer.Option(help="Relative deviation tolerance.")] = 1e-3,
) -> None:
    """Export the selected variants and verify each against the torch reference."""
    # Lazy import: needs the [chronos] extra; keeps the CLI importable without torch.
    from openstef_checkpoints.models.chronos2.export import export_and_verify  # noqa: PLC0415

    cli: CliContext = ctx.obj
    config = cli.model(model)
    # Per-model subdirectory so exporting several models never clashes on filenames.
    model_dir = out / config.slug
    model_dir.mkdir(parents=True, exist_ok=True)
    results = export_and_verify(config, out_dir=model_dir, variants=_select_variants(variant), atol=atol, rtol=rtol)

    records = [
        VariantRecord(
            filename=result.checkpoint.weights_path.name,
            precision=result.checkpoint.metadata.precision,
            static_shapes=result.checkpoint.metadata.static_shapes,
            max_abs=result.deviation.max_abs,
            within_tolerance=result.deviation.within_tolerance,
            publish=result.variant.publish,
        )
        for result in results
    ]
    provenance = ExportProvenance.capture(
        source_model_id=config.source_model_id,
        exporter_revision=os.environ.get("GITHUB_SHA", "unknown"),
    )
    repo_id = cli.settings.repo_id(config.slug)
    Manifest(slug=config.slug, repo_id=repo_id, provenance=provenance, variants=records).write(model_dir)
    _print_results(records)


@app.command()
def publish(
    ctx: typer.Context,
    model: Annotated[str, typer.Argument(help="Model slug, e.g. 'chronos-2'.")],
    out: Annotated[Path, typer.Option(help="Base directory holding the exports; reads from <out>/<slug>.")] = Path(
        "checkpoints"
    ),
    repo_id: Annotated[
        str | None, typer.Option(help="Override the target repo (e.g. your personal repo for testing).")
    ] = None,
    private: Annotated[bool, typer.Option(help="Create the repo private.")] = True,
    create_repo: Annotated[
        bool,
        typer.Option(
            help="Create the repo if missing (token auth). Use --no-create-repo for OIDC "
            "trusted publishing, where the repo must already exist.",
        ),
    ] = True,
    force: Annotated[bool, typer.Option(help="Publish even if some variants failed the deviation check.")] = False,
) -> None:
    """Render the model card and upload the exported variants to HuggingFace.

    Raises:
        Exit: If some variants failed the deviation check and `--force` was not given.
    """
    cli: CliContext = ctx.obj
    config = cli.model(model)
    model_dir = out / config.slug
    manifest = Manifest.read(model_dir)

    held = [record.filename for record in manifest.variants if not record.publish]
    if held:
        console.print(f"[yellow]Holding back build-only variant(s)[/]: {', '.join(held)}")
    publishable = [record for record in manifest.variants if record.publish]
    failing = [record.filename for record in publishable if not record.within_tolerance]
    if failing and not force:
        console.print(f"[red]Refusing to publish: {len(failing)} variant(s) failed the check[/]: {', '.join(failing)}")
        console.print("Re-run with --force to publish anyway.")
        raise typer.Exit(code=1)
    selected = [record for record in publishable if record.within_tolerance or force]
    if not selected:
        console.print("[red]Nothing to publish.[/]")
        raise typer.Exit(code=1)

    card = render_card(config.CARD_TEMPLATE, manifest, source_license=config.source_license)
    (model_dir / CARD_NAME).write_text(card, encoding="utf-8")
    allow_patterns = [name for record in selected for name in (record.filename, record.metadata_filename)] + [CARD_NAME]
    target = repo_id or manifest.repo_id
    # Scope the OIDC trusted-publishing exchange to the repo we upload to (a no-op when an
    # HF token is present, e.g. local `hf auth`). The resource is always the target repo,
    # so derive it here and keep the manifest the single source of truth, overridable via env.
    os.environ.setdefault("HF_OIDC_RESOURCE", target)
    console.print(f"Publishing {len(selected)} variant(s) to [bold]{target}[/] (private={private}) ...")
    url = publish_repo(target, model_dir, allow_patterns=allow_patterns, private=private, create=create_repo)
    console.print(f"[green]Published[/] {url}")


def _print_results(records: list[VariantRecord]) -> None:
    """Print the export results, flagging variants that failed the deviation check."""
    table = Table(title="Export results")
    for column in ("Variant file", "Precision", "Static", "Max abs dev", "Check"):
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
