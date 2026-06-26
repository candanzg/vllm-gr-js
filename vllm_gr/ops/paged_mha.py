# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.

"""
Paged-KV Multi-Head Attention operator entry points.

This module exposes the PyTorch-backed full and incremental (delta) attention
implementations used by the paged-KV attention backend. The historical NPU
fused (mxrec.hstu_jagged) dispatch path has been removed; both GPU and NPU
runtimes now execute through the PyTorch reference path.
"""

from typing import Optional

import torch

from vllm_gr.ops.pytorch.pt_paged_mha import (
    pytorch_cached_paged_mha,
    pytorch_paged_mha,
)


def paged_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool = True,
    dropout_pr: float = 0.0,
    training: bool = False,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
    sort_by_length: bool = False,
    kernel: Optional[object] = None,
    enable_tma: bool = False,
) -> torch.Tensor:
    """Full-sequence paged-KV multi-head attention (PyTorch reference path)."""
    del sort_by_length, enable_tma, kernel
    return pytorch_paged_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        causal=causal,
        dropout_pr=dropout_pr,
        training=training,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        min_full_attn_seq_len=min_full_attn_seq_len,
    )


def delta_paged_mha(
    max_seq_len: int,
    alpha: float,
    delta_q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    kernel: Optional[object] = None,
    enable_tma: bool = False,
    kv_cache: Optional[torch.Tensor] = None,
    block_table_tensor: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    num_kv_heads: Optional[int] = None,
    head_size: Optional[int] = None,
    head_size_v: Optional[int] = None,
    block_size: Optional[int] = None,
) -> torch.Tensor:
    """Incremental paged-KV multi-head attention (PyTorch reference path)."""
    del enable_tma, kernel, kv_cache, block_table_tensor, seq_lens
    del num_kv_heads, head_size, head_size_v, block_size

    return pytorch_cached_paged_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        delta_q=delta_q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
    )
