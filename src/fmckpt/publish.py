# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Publishing: the export→publish manifest, the model card, and HuggingFace upload.

Light (no torch): `export` writes a `manifest.json` recording provenance and each
variant's deviation; `publish` reads it to render the card and upload — so publishing
runs anywhere, decoupled from the heavy export. The HF repo is created private; the
caller decides when to flip it public.
"""

from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from huggingface_hub import HfApi
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, ConfigDict, Field

MANIFEST_NAME = "manifest.json"
_CARD_NAME = "README.md"
_UPLOAD_PATTERNS = ["*.onnx", "*.metadata.json", _CARD_NAME]


def _tool_version(package: str) -> str:
    """Return *package*'s installed version, or ``n/a`` if it is absent."""
    try:
        return version(package)
    except PackageNotFoundError:
        return "n/a"


class ExportProvenance(BaseModel):
    """Where a published checkpoint came from, for reproducibility."""

    model_config = ConfigDict(frozen=True)

    source_model_id: str = Field(description="Upstream HuggingFace model id that was exported.")
    source_revision: str = Field(default="unknown", description="Upstream model revision, if known.")
    exporter_revision: str = Field(default="unknown", description="foundation-model-checkpoints commit that built it.")
    tooling: str = Field(description="Versions of the export toolchain (onnx/onnxruntime/torch).")
    exported_at: str = Field(description="UTC timestamp of the export.")

    @classmethod
    def capture(
        cls, *, source_model_id: str, source_revision: str = "unknown", exporter_revision: str
    ) -> "ExportProvenance":
        """Capture provenance from the environment at export time.

        Args:
            source_model_id: Upstream model id being exported.
            source_revision: Upstream model revision, if resolvable.
            exporter_revision: This repo's commit (e.g. ``$GITHUB_SHA`` in CI).

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
    within_tolerance: bool = Field(description="Whether the variant passed the deviation gate.")


class Manifest(BaseModel):
    """The export→publish hand-off: provenance plus a record per exported variant."""

    model_config = ConfigDict(frozen=True)

    slug: str = Field(description="Model slug, e.g. 'chronos-2'.")
    repo_id: str = Field(description="Default target HuggingFace repo for this model.")
    provenance: ExportProvenance = Field(description="Where the checkpoints came from.")
    variants: list[VariantRecord] = Field(description="One record per exported variant.")

    def write(self, directory: Path) -> Path:
        """Write the manifest to ``<directory>/manifest.json``.

        Returns:
            The path written.
        """
        path = directory / MANIFEST_NAME
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def read(cls, directory: Path) -> "Manifest":
        """Read the manifest from ``<directory>/manifest.json``.

        Returns:
            The parsed manifest.
        """
        return cls.model_validate_json((directory / MANIFEST_NAME).read_text(encoding="utf-8"))


def render_card(template_path: Path, manifest: Manifest) -> str:
    """Render a model card from *template_path* and *manifest*.

    Args:
        template_path: Path to the model's Jinja card template.
        manifest: The export manifest providing variants and provenance.

    Returns:
        The rendered card markdown.
    """
    env = Environment(loader=FileSystemLoader(str(template_path.parent)), autoescape=select_autoescape())
    template = env.get_template(template_path.name)
    return template.render(
        slug=manifest.slug,
        source_model_id=manifest.provenance.source_model_id,
        provenance=manifest.provenance,
        variants=manifest.variants,
    )


def publish_repo(repo_id: str, source_dir: Path, *, private: bool = True, token: str | None = None) -> str:
    """Create (if needed) and upload a directory's checkpoints + card to HuggingFace.

    Uploads only the weights, sidecars and ``README.md`` — never the manifest or
    other local files.

    Args:
        repo_id: Target repo, e.g. ``your-username/chronos-2-onnx``.
        source_dir: Directory holding the ``.onnx`` files, sidecars and card.
        private: Whether to create the repo private (default; flip public later).
        token: HuggingFace token; falls back to the cached login / ``HF_TOKEN``.

    Returns:
        The repo URL.
    """
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(source_dir), allow_patterns=_UPLOAD_PATTERNS)
    return f"https://huggingface.co/{repo_id}"
