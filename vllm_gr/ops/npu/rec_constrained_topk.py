# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU dispatch wrapper for the fused constrained Top-K operator."""

from __future__ import annotations

from typing import Any

import torch


def _custom_ops() -> Any | None:
    """Load the optional NPU custom-op extension lazily."""
    try:
        import custom_ops_lib as custom_ops  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError):
        return None
    return custom_ops


def is_available() -> bool:
    """Return whether the NPU extension exposes the fused operator."""
    custom_ops = _custom_ops()
    return custom_ops is not None and callable(getattr(custom_ops, "rec_constrained_topk", None))


def rec_constrained_topk_npu(
    logits: torch.Tensor,
    sequence_group: torch.Tensor,
    first_token_ids: torch.Tensor,
    prefix1_offsets: torch.Tensor,
    prefix1_values: torch.Tensor,
    prefix1_pair_keys: torch.Tensor,
    prefix2_value_offsets: torch.Tensor,
    prefix2_values: torch.Tensor,
    temperatures: torch.Tensor,
    current_step: int,
    top_k: int,
    max_prefix1_degree: int = 0,
    max_prefix2_degree: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the fused NPU operator using the shared operator signature."""
    custom_ops = _custom_ops()
    if custom_ops is None or not callable(getattr(custom_ops, "rec_constrained_topk", None)):
        raise RuntimeError(
            "NPU custom_ops_lib.rec_constrained_topk is unavailable; "
            "use the Torch implementation instead"
        )
    return custom_ops.rec_constrained_topk(
        logits,
        sequence_group,
        first_token_ids,
        prefix1_offsets,
        prefix1_values,
        prefix1_pair_keys,
        prefix2_value_offsets,
        prefix2_values,
        temperatures,
        current_step,
        top_k,
        max_prefix1_degree,
        max_prefix2_degree,
    )


__all__ = ["is_available", "rec_constrained_topk_npu"]
