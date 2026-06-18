# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Settings that depend on where checkpoints are published, not on the model.

Each field reads from an environment variable prefixed `OPENSTEF_CHECKPOINTS_`, so a
different target (a personal namespace for testing) is a one-variable change, e.g.
`OPENSTEF_CHECKPOINTS_HF_NAMESPACE=my-user`.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Publishing configuration read from the environment."""

    model_config = SettingsConfigDict(env_prefix="OPENSTEF_CHECKPOINTS_")

    hf_namespace: str = Field(default="OpenSTEF", description="HuggingFace user or org the checkpoints publish under.")

    def repo_id(self, slug: str) -> str:
        """Return the publish target for a model, `<namespace>/<slug>-onnx`.

        Args:
            slug: The model slug, e.g. 'chronos-2'.

        Returns:
            The HuggingFace repo id.
        """
        return f"{self.hf_namespace}/{slug}-onnx"
