<!--
SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>

SPDX-License-Identifier: MPL-2.0
-->

# foundation-model-checkpoints

Export, verify, and publish the ONNX foundation-model checkpoints that
[OpenSTEF](https://github.com/OpenSTEF/openstef) resolves at runtime.

This is the **producer** side of a producer/consumer split (see OpenSTEF design
doc `0002`): it turns an upstream foundation model (Chronos-2 first) into the ONNX
variant matrix — `{dynamic, static} × {fp32, fp16, int8}` — plus a `CheckpointMetadata`
sidecar, gates each variant through an accuracy-deviation check on representative
inputs, and publishes the result to a per-model HuggingFace repo. The consumer
(`openstef-foundation-models`) only ever resolves the published checkpoints via
`HubCheckpoint`; it never depends on this repo.

It depends on **nothing** from `openstef-foundation-models`. The metadata schema is
a *governed duplicate*: each repo owns its copy, kept compatible by an append-only
rule and a golden JSON-Schema conformance test on both sides — not a build-time
dependency (which would form a release cycle at every schema bump).

## Layout

```
src/fmckpt/            # model-agnostic machinery (export, verify, publish, schema, cli)
src/fmckpt/models/<x>/ # per-model specifics only (the torch wrapper + variant/size config)
```

`fmckpt/` is everything that does not know it is Chronos; `models/<x>/` is the
irreducibly model-specific part. One package today; split into `fmckpt-core` /
`fmckpt-cli` only if a second model forces shared reuse (rule of three).

## Develop

```sh
uv sync                 # light env (schema, verify, publish, cli)
uv sync --extra chronos # add torch + chronos-forecasting to actually export
uv run poe lint
uv run poe type
uv run poe tests
```
