<!--
SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>

SPDX-License-Identifier: MPL-2.0
-->

# OpenSTEF-checkpoints

Tooling to export, verify, and publish the ONNX foundation-model checkpoints used by
[OpenSTEF](https://github.com/OpenSTEF/openstef).

## What is OpenSTEF-checkpoints

OpenSTEF runs pre-trained forecasting foundation models through ONNX Runtime. This
project produces those ONNX checkpoints from their upstream source models and publishes
them to Hugging Face, where OpenSTEF resolves them at runtime. The first supported model
is [Chronos-2](https://huggingface.co/amazon/chronos-2).

For each model it:

- converts the upstream model to ONNX in several variants: dynamic or static shapes,
  in `fp32`, `fp16`, or `int8`;
- writes a metadata file describing each variant (tensor names, context length,
  quantiles, precision, shapes);
- checks every variant against the original model on representative inputs, so one that
  drifts beyond tolerance is never published;
- uploads the selected variants and a generated model card to a per-model Hugging Face
  repository.

The published checkpoint metadata follows the same schema OpenSTEF reads, kept in sync by
a JSON-Schema conformance test on both sides rather than a shared dependency.

## Project layout

| Path | Contents |
| --- | --- |
| `src/openstef_checkpoints/` | Model-agnostic machinery: export, verification, publishing, metadata schema, and the CLI. |
| `src/openstef_checkpoints/models/<model>/` | Model-specific code: the export wrapper and the variant and sizing configuration. |

## How to Install

The project uses [uv](https://docs.astral.sh/uv/). The base install provides the metadata
schema, ONNX-graph verification, publishing, and the CLI. Exporting a checkpoint additionally
requires the source-model stack, provided by the `chronos` extra:

```sh
uv sync                  # base environment
uv sync --extra chronos  # add torch and chronos-forecasting to run exports
```

## How to Use

The `openstef-checkpoints` command exposes three subcommands:

```sh
openstef-checkpoints list                  # show the configured models and variants
openstef-checkpoints export chronos-2      # export and verify all variants
openstef-checkpoints publish chronos-2     # upload the verified variants to Hugging Face
```

Exports are written to `checkpoints/<model>/`. See `openstef-checkpoints <command> --help`
for the available options.

## Quick Development Setup

Tasks are defined with [poethepoet](https://poethepoet.natn.io/) and run through uv:

```sh
uv run poe lint
uv run poe type
uv run poe tests
uv run poe all --check   # full CI sequence
```

## License

Licensed under the [Mozilla Public License 2.0](LICENSE.md). Published checkpoints carry
the license of their upstream source model.

## Contributing

We welcome contributions to OpenSTEF-checkpoints! 

**[Read our Contributing Guide](https://openstef.github.io/openstef/contribute/)** - documentation for contributors including:

- How to report bugs and suggest features
- Documentation improvements and examples
- Code contributions and development setup
- Sharing datasets and real-world use cases

## Citations

If you use OpenSTEF in your research or publications, please cite the project. Refer to the [CITATION.cff](CITATION.cff) file in this repository for the preferred citation format, or use:

> Contributors to the OpenSTEF project. *OpenSTEF — Open Short-Term Energy Forecasting*. LF Energy, 2017–2025. Available at: https://github.com/OpenSTEF/openstef

## Contact

- **Slack:** [LF Energy Slack](https://slack.lfenergy.org/)
- **Email:** openstef@lfenergy.org
- **Community meeting:** [OpenSTEF four-weekly community meeting](https://lf-energy.atlassian.net/wiki/spaces/OS/pages/32278358/OpenSTEF+four-weekly+community+meeting)
- **Issues:** [GitHub Issue Tracker](https://github.com/OpenSTEF/openstef-checkpoints/issues)
- **Support Guide:** [How to get help](https://openstef.github.io/openstef/project/support.html)


