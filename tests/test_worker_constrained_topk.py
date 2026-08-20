# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm_gr.v1.beam.constraint_context import BeamConstraintTables, BeamSearchContext
from vllm_gr.v1.worker.constrained_topk import (
    WorkerConstrainedTopKInput,
    worker_constrained_topk,
)


def test_worker_constrained_topk_dispatches_with_context() -> None:
    logits = torch.tensor([[0.1, 0.9, 0.2]], dtype=torch.float32)
    first_token_ids = torch.tensor([1, 2], dtype=torch.int64)
    prefix1_offsets = torch.tensor([0, 0, 2], dtype=torch.int64)
    prefix1_values = torch.tensor([1, 2], dtype=torch.int64)
    prefix1_pair_keys = torch.tensor([0], dtype=torch.int64)
    prefix2_value_offsets = torch.tensor([0, 0], dtype=torch.int64)
    prefix2_values = torch.tensor([], dtype=torch.int64)
    temperatures = torch.tensor([1.0], dtype=torch.float32)
    sequence = torch.tensor([[1]], dtype=torch.int32)

    ctx = BeamSearchContext(
        current_step=0,
        sequence=sequence,
        temperatures=temperatures,
        tables=BeamConstraintTables(
            trie={"root": True}, constraint_tables={"demo": torch.tensor([1])}
        ),
    )

    out_tokens, out_logprobs = worker_constrained_topk(
        ctx,
        WorkerConstrainedTopKInput(
            logits=logits,
            sequence_group=sequence,
            first_token_ids=first_token_ids,
            prefix1_offsets=prefix1_offsets,
            prefix1_values=prefix1_values,
            prefix1_pair_keys=prefix1_pair_keys,
            prefix2_value_offsets=prefix2_value_offsets,
            prefix2_values=prefix2_values,
            temperatures=temperatures,
            current_step=0,
            top_k=1,
        ),
    )

    assert out_tokens.shape == (1, 1)
    assert out_logprobs.shape == (1, 1)
    assert out_tokens.item() == 1


def test_worker_beam_context_proxy_fields() -> None:
    from vllm_gr.v1.beam.constraint_context import (
        BeamConstraintTables,
        BeamSearchContext,
    )
    from vllm_gr.v1.worker.constrained_topk import get_worker_beam_context

    ctx = BeamSearchContext(
        current_step=3,
        sequence=torch.tensor([[1, 2, 3]], dtype=torch.int32),
        temperatures=torch.tensor([0.5], dtype=torch.float32),
        tables=BeamConstraintTables(trie={"x": 1}, constraint_tables={"demo": torch.tensor([2])}),
    )

    class Runner:
        _beam_sessions = {"req": type("S", (), {"context": ctx})()}

    got = get_worker_beam_context(Runner())
    assert got is ctx
    assert got.current_step == 3
    assert got.temperatures.item() == 0.5
    assert got.tables.constraint_tables["demo"].item() == 2
