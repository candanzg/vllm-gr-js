# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.

"""
Common kernel-selection primitive and jagged-tensor helpers shared by the
paged-KV attention operators.

These utilities are implemented on top of native PyTorch and carry no
model-specific semantics; they live here so operator modules can depend on
them without pulling in any model package.
"""

from enum import Enum, unique
from typing import List, Optional

import torch

from vllm_gr.utils import ReliableOps


@unique
class HammerKernel(Enum):
    """Requested kernel backends.

    Values in this enum describe the caller's preferred backend, not a blanket
    guarantee that every operator in this fork has a specialized implementation
    for that backend.
    """

    TRITON = "TRITON"
    TLX = "TLX"
    PYTORCH = "PYTORCH"
    CUDA = "CUDA"
    NPU = "NPU"
    TRITON_CC = "TRITON_CC"


@torch.fx.wrap
def fx_unwrap_optional_tensor(optional: Optional[torch.Tensor]) -> torch.Tensor:
    assert optional is not None, "Expected optional to be non-None Tensor"
    return optional


@torch.fx.wrap
def jagged_to_padded_dense(
    values: torch.Tensor,
    offsets: List[torch.Tensor],
    max_lengths: List[int],
    padding_value: float,
) -> torch.Tensor:
    """
    Convert Jagged Tensor to padded dense tensor (native PyTorch implementation).
    """
    return ReliableOps.jagged_to_padded_dense(
        values=values,
        offsets=offsets,
        max_lengths=max_lengths,
        padding_value=padding_value,
    )


@torch.fx.wrap
def dense_to_jagged(
    dense: torch.Tensor,
    x_offsets: List[torch.Tensor],
) -> torch.Tensor:
    """
    Convert padded dense tensor to Jagged Tensor (native PyTorch implementation).
    """
    return ReliableOps.dense_to_jagged(
        dense=dense,
        x_offsets=x_offsets,
    )


__all__ = [
    "HammerKernel",
    "fx_unwrap_optional_tensor",
    "jagged_to_padded_dense",
    "dense_to_jagged",
]
