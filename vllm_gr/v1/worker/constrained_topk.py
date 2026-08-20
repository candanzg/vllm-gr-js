# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

from vllm_gr.ops.rec_constrained_topk import rec_constrained_topk
from vllm_gr.v1.beam.constraint_context import BeamSearchContext


@dataclass
class WorkerConstrainedTopKInput:
    logits: torch.Tensor
    sequence_group: Optional[torch.Tensor]
    first_token_ids: torch.Tensor
    prefix1_offsets: torch.Tensor
    prefix1_values: torch.Tensor
    prefix1_pair_keys: torch.Tensor
    prefix2_value_offsets: torch.Tensor
    prefix2_values: torch.Tensor
    temperatures: torch.Tensor
    current_step: int
    top_k: int
    max_prefix1_degree: int = 0
    max_prefix2_degree: int = 0
    trie: Optional[Any] = None
    constraint_tables: Optional[Mapping[str, torch.Tensor]] = None


def worker_constrained_topk(
    ctx: Optional[BeamSearchContext],
    inputs: WorkerConstrainedTopKInput,
) -> tuple[torch.Tensor, torch.Tensor]:
    if ctx is not None:
        current_step = ctx.current_step
        temperatures = ctx.temperatures if ctx.temperatures is not None else inputs.temperatures
        sequence_group = ctx.sequence if ctx.sequence is not None else inputs.sequence_group
        tables = (
            ctx.tables.constraint_tables if ctx.tables is not None else inputs.constraint_tables
        )
        if tables is not None:
            _ = tables
    else:
        current_step = inputs.current_step
        temperatures = inputs.temperatures
        sequence_group = inputs.sequence_group

    return rec_constrained_topk(
        inputs.logits,
        sequence_group,
        inputs.first_token_ids,
        inputs.prefix1_offsets,
        inputs.prefix1_values,
        inputs.prefix1_pair_keys,
        inputs.prefix2_value_offsets,
        inputs.prefix2_values,
        temperatures,
        current_step,
        inputs.top_k,
        inputs.max_prefix1_degree,
        inputs.max_prefix2_degree,
    )


def get_worker_beam_context(runner: Any) -> Optional[BeamSearchContext]:
    beam_sessions = getattr(runner, "_beam_sessions", None)
    if beam_sessions:
        for session in beam_sessions.values():
            if session is not None and hasattr(session, "context"):
                return session.context
    return getattr(runner, "_beam_constraint_context", None)
