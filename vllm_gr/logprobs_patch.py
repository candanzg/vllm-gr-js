# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patch for FlatLogprobs.append_fast to use extend instead of
per-element append."""

from vllm_gr.compat.runtime_v0_30 import resolve_symbol
from vllm_gr.logprobs import append_fast as _append_fast
from vllm_gr.patch_report import observed_installation


@observed_installation
def patch_flat_logprobs():
    FlatLogprobs = resolve_symbol("FlatLogprobs")
    FlatLogprobs.append_fast = _append_fast
