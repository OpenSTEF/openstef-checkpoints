# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""The preconfigured models this tool exports, in one place.

Each entry is a ready-to-export configuration, registered in `MODELS` by slug. Adding
a model (a new size, or a different family) means adding it here, not editing the CLI.
"""

from openstef_checkpoints.models.chronos2.config import Chronos2Model

CHRONOS2 = Chronos2Model(slug="chronos-2", source_model_id="amazon/chronos-2")
CHRONOS2_SMALL = Chronos2Model(slug="chronos-2-small", source_model_id="amazon/chronos-2-small")

MODELS: dict[str, Chronos2Model] = {model.slug: model for model in (CHRONOS2, CHRONOS2_SMALL)}
