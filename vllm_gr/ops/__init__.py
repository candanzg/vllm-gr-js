# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Original code from Meta Platforms, Inc. (generative-recommenders)

"""
vllm-GR Operators Package

Provides paged-KV multi-head attention and shared jagged-tensor helpers used
by the recommendation model runtime.
"""

from vllm_gr.ops.common import (
    HammerKernel,
    dense_to_jagged,
    jagged_to_padded_dense,
)
from vllm_gr.ops.paged_mha import delta_paged_mha, paged_mha

__all__ = [
    "HammerKernel",
    "dense_to_jagged",
    "delta_paged_mha",
    "jagged_to_padded_dense",
    "paged_mha",
]
