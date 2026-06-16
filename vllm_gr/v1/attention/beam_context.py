# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lightweight shared context for beam-prefix metadata injection.

This module must stay free of backend-specific imports so that both GPU and
NPU runner paths can exchange beam-prefix group metadata without pulling in
Triton / FlashAttention dependencies.
"""

from __future__ import annotations

import contextvars

BEAM_PREFIX_GROUPS_VAR = contextvars.ContextVar(
    "beam_prefix_groups",
    default=None,
)
