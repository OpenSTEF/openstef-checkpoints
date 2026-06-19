# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""The export-to-publish handoff: the manifest, the model card, and the upload.

`export` writes `manifest.json`, recording where each checkpoint came from and how far
it deviated from the reference. `publish` reads it to render the model card and upload
the chosen files. Neither step imports torch, so publishing can run on its own after a
separate export.
"""

from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Self

from huggingface_hub import HfApi
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, ConfigDict, Field

MANIFEST_NAME = "manifest.json"
CARD_NAME = "README.md"


def _tool_version(package: str) -> str:
    """Return package's installed version, or `n/a` if it is absent."""
    try:
        return version(package)
    except PackageNotFoundError:
        return "n/a"


class ExportProvenance(BaseModel):
    """Where a published checkpoint came from, for reproducibility."""

    model_config = ConfigDict(frozen=True)

    source_model_id: str = Field(description="Upstream HuggingFace model id that was exported.")
    source_revision: str = Field(default="unknown", description="Upstream model revision, if known.")
    exporter_revision: str = Field(default="unknown", description="openstef-checkpoints commit that built it.")
    tooling: str = Field(description="Versions of the export toolchain (onnx/onnxruntime/torch).")
    exported_at: str = Field(description="UTC timestamp of the export.")

    @classmethod
    def capture(cls, *, source_model_id: str, source_revision: str = "unknown", exporter_revision: str) -> Self:
        """Capture provenance from the environment at export time.

        Args:
            source_model_id: Upstream model id being exported.
            source_revision: Upstream model revision, if resolvable.
            exporter_revision: This repo's commit (e.g. `$GITHUB_SHA` in CI).

        Returns:
            The captured provenance.
        """
        tooling = " ".join(f"{pkg}={_tool_version(pkg)}" for pkg in ("onnx", "onnxruntime", "torch"))
        return cls(
            source_model_id=source_model_id,
            source_revision=source_revision,
            exporter_revision=exporter_revision,
            tooling=tooling,
            exported_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )


class VariantRecord(BaseModel):
    """One published variant's identity and its deviation from the torch reference."""

    model_config = ConfigDict(frozen=True)

    filename: str = Field(description="Weights filename within the repo.")
    precision: str = Field(description="Variant precision (fp32/fp16/int8).")
    static_shapes: bool = Field(description="Whether the graph's shapes are frozen.")
    max_abs: float = Field(description="Max absolute deviation vs the torch reference.")
    within_tolerance: bool = Field(description="Whether the variant passed the deviation check.")
    publish: bool = Field(description="Whether this variant is intended for upload (False = build-only).")

    @property
    def metadata_filename(self) -> str:
        """The variant's metadata filename."""
        return Path(self.filename).with_suffix(".metadata.json").name


class Manifest(BaseModel):
    """Provenance and one record per exported variant."""

    model_config = ConfigDict(frozen=True)

    slug: str = Field(description="Model slug, e.g. 'chronos-2'.")
    repo_id: str = Field(description="HuggingFace repo this model publishes to.")
    provenance: ExportProvenance = Field(description="Where the checkpoints came from.")
    variants: list[VariantRecord] = Field(description="One record per exported variant.")

    def write(self, directory: Path) -> Path:
        """Write the manifest to `<directory>/manifest.json`.

        Returns:
            The path written.
        """
        path = directory / MANIFEST_NAME
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def read(cls, directory: Path) -> Self:
        """Read the manifest from `<directory>/manifest.json`.

        Returns:
            The parsed manifest.
        """
        return cls.model_validate_json((directory / MANIFEST_NAME).read_text(encoding="utf-8"))

    @property
    def held_back(self) -> list[VariantRecord]:
        """Variants that were built and checked but are not intended for upload."""
        return [record for record in self.variants if not record.publish]

    @property
    def publishable(self) -> list[VariantRecord]:
        """Variants intended for upload, regardless of their deviation verdict."""
        return [record for record in self.variants if record.publish]

    @property
    def failing(self) -> list[VariantRecord]:
        """Publishable variants that failed the deviation check."""
        return [record for record in self.publishable if not record.within_tolerance]

    def selected_for_upload(self, *, force: bool) -> list[VariantRecord]:
        """The publishable variants to actually upload.

        Args:
            force: Upload even the variants that failed the deviation check.

        Returns:
            Publishable variants that passed the check, plus the failing ones when `force`.
        """
        return [record for record in self.publishable if record.within_tolerance or force]

    def render_card(self, template_path: Path, *, source_license: str) -> str:
        """Render this manifest's model card from a Jinja template.

        Args:
            template_path: Path to the model's Jinja card template.
            source_license: License of the upstream weights (governs the published checkpoint).

        Returns:
            The rendered card markdown, advertising only the published variants.
        """
        env = Environment(loader=FileSystemLoader(str(template_path.parent)), autoescape=select_autoescape())
        template = env.get_template(template_path.name)
        return template.render(
            slug=self.slug,
            source_model_id=self.provenance.source_model_id,
            source_license=source_license,
            provenance=self.provenance,
            variants=self.publishable,
        )


def publish_repo(
    repo_id: str,
    source_dir: Path,
    *,
    allow_patterns: list[str],
    private: bool = True,
    token: str | None = None,
    create: bool = True,
) -> str:
    """Upload a fixed list of files to a HuggingFace repo, creating it if asked.

    Only the files named in `allow_patterns` are uploaded, never a wildcard, so a
    build-only variant cannot leak out of the directory.

    Args:
        repo_id: The HuggingFace repo to publish to.
        source_dir: Directory holding the files.
        allow_patterns: Exact filenames to upload.
        private: Visibility when the repo is created. Can be flipped public later.
        token: HuggingFace token. Falls back to the cached login, `HF_TOKEN`, or the
            OIDC trusted-publishing exchange scoped by `HF_OIDC_RESOURCE`.
        create: Whether to create the repo first. An OIDC token is scoped to an
            existing repo and cannot create one, so disable this when publishing with it.

    Returns:
        The repo URL.
    """
    api = HfApi(token=token)
    if create:
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(source_dir), allow_patterns=allow_patterns)
    return f"https://huggingface.co/{repo_id}"
