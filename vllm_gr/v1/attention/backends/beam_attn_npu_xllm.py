# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""xLLM fused-operator execution primitives for the Beam Attention backend.

NPU-only execution module selected by ``_get_platform_execution_module`` when
``vllm_gr_npu_decode_execution == "xllm"``.  It replaces the cascade
per-operator path only for the incremental beam decode branch; every other
branch (std requests, prefill, suffix recompute) forwards to
``beam_attn_npu`` unchanged, and the CUDA path is never touched.

The incremental branch becomes a single fused ``x_attention`` call that
attends over the paged shared prefix and the contiguous unshared beam pool
in one kernel, replacing "two npu_fused_infer_attention_score_v2 calls +
pool->buffer copies + the merge operator chain".
"""

from __future__ import annotations

import math

import torch

from vllm_gr.ops.npu import beam_ops
from vllm_gr.v1.attention.backends import beam_attn_npu

# Fail fast when the mode selects this module but the operator library is
# missing: without this check the failure would surface at the first beam
# decode step, deep inside model execution.
if not beam_ops.is_available():
    raise RuntimeError(
        "vllm_gr_npu_decode_execution='xllm' requires the xllm-ops library "
        f"({beam_ops.unavailable_reason()}); install it with "
        "tools/install_xllm_ops.sh or select 'cascade' explicitly via "
        "additional_config['vllm_gr_npu_decode_execution']='cascade'"
    )

# Forwarded primitives: std requests, prefill and cache writes behave
# exactly like the cascade module.
is_available = beam_attn_npu.is_available
get_flash_attn_version = beam_attn_npu.get_flash_attn_version
supports_sinks = beam_attn_npu.supports_sinks
reshape_and_cache = beam_attn_npu.reshape_and_cache


def cache_unshared_kv(
    unshared_key: torch.Tensor,
    unshared_value: torch.Tensor,
    curr_key: torch.Tensor,
    curr_value: torch.Tensor,
    slot: int,
    decode_step: int,
) -> None:
    """Fused pool write: slot -> block_table, 0-based step -> 1-based."""
    block_table = torch.tensor([slot], dtype=torch.int32, device=unshared_key.device)
    beam_ops.cache_unshared_kv(
        unshared_key,
        unshared_value,
        curr_key,
        curr_value,
        block_table,
        decode_step + 1,
    )


def _xllm_incremental_decode(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata,
    beam_session,
    beam_pool,
    scale: float,
) -> None:
    """One fused x_attention call for the incremental beam decode branch."""
    step = beam_session.decode_step  # 0-based host int
    slot = beam_session.slot
    beam_width = beam_session.beam_width
    head_dim = key_cache.shape[-1]

    if abs(scale - (1.0 / math.sqrt(head_dim))) > 1e-6:
        # The kernel derives its own scale when scale_value <= 0; only the
        # standard 1/sqrt(head_dim) matches the model contract.
        raise RuntimeError(
            "xllm decode execution only supports softmax_scale = 1/sqrt(head_dim), "
            f"got {scale}"
        )

    # Beam queries: [W, num_heads, D] (already gathered at prefix_indices).
    beam_query = query[attn_metadata.prefix_indices]

    # Contiguous unshared pool view: [W, H_kv, max_steps, D] -- exactly the
    # layout x_attention expects for the contiguous unshared branch
    # (unshared_block_tables=None, implicit row mapping).
    unshared_key = beam_pool.unshared_key[slot, :beam_width]
    unshared_value = beam_pool.unshared_value[slot, :beam_width]

    out = beam_ops.x_attention(
        beam_query,
        key_cache,
        value_cache,
        unshared_key,
        unshared_value,
        attn_metadata.prefix_block_table,
        torch.tensor(
            attn_metadata.prefix_seqlens_kv_cpu,
            dtype=torch.int32,
            device=beam_query.device,
        ),
        step + 1,  # dev is 0-based; the kernel takes the 1-based step count
        0.0,  # auto 1/sqrt(head_dim), validated above
    )
    output[attn_metadata.prefix_indices] = out


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata,
    sliding_window_size: list[int] | tuple[int, int] | None,
    scale: float,
    logits_soft_cap: float,
    vllm_flash_attn_version: int,
    s_aux: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    beam_pool: object = None,
) -> None:
    """Dispatch incremental beam decode to x_attention; forward the rest."""
    beam_session = getattr(attn_metadata, "beam_session", None)
    is_beam_decode = getattr(attn_metadata, "is_beam_decode", False)

    if is_beam_decode and beam_session is not None and beam_session.is_active:
        if attn_metadata.std_indices is not None:
            raise RuntimeError(
                "xllm decode execution does not support mixed std/beam batches; "
                "use the cascade mode for mixed scheduling"
            )
        if attn_metadata.prefix_indices is None:
            raise RuntimeError(
                "xllm decode execution requires prefix_indices for beam queries"
            )
        # (-1, -1) is the "no sliding window" sentinel the impl passes for
        # models without one; only a real window is unsupported here (the
        # cascade path ignores the argument entirely on the beam branch).
        if sliding_window_size is not None and any(
            w is not None and w >= 0 for w in sliding_window_size
        ):
            raise RuntimeError(
                "xllm decode execution does not support sliding_window_size"
            )
        if logits_soft_cap:
            raise RuntimeError("xllm decode execution does not support logits_soft_cap")
        if s_aux is not None:
            raise RuntimeError("xllm decode execution does not support sinks (s_aux)")
        if alibi_slopes is not None:
            raise RuntimeError("xllm decode execution does not support alibi_slopes")
        if beam_pool is None:
            raise RuntimeError(
                "xllm decode execution requires the per-layer beam_pool argument"
            )
        _xllm_incremental_decode(
            output,
            query,
            key_cache,
            value_cache,
            attn_metadata,
            beam_session,
            beam_pool,
            scale,
        )
        return

    # Std requests, prefill, recompute and mixed batches keep the cascade
    # implementation.
    beam_attn_npu.cascade_attention(
        output,
        query,
        key_cache,
        value_cache,
        attn_metadata,
        sliding_window_size,
        scale,
        logits_soft_cap,
        vllm_flash_attn_version,
        s_aux=s_aux,
        alibi_slopes=alibi_slopes,
        beam_pool=beam_pool,
    )


__all__ = [
    "cache_unshared_kv",
    "cascade_attention",
    "get_flash_attn_version",
    "is_available",
    "reshape_and_cache",
    "supports_sinks",
]
