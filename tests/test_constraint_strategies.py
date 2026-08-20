# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import numpy as np
import pytest

from vllm_gr.config import ConstraintBackend
from vllm_gr.constraints import (
    BeamConstraintCoordinator,
    CandidateLayout,
    ParentCandidateBuilder,
    create_constraint_strategy,
)


def test_worker_backend_routes_to_worker_strategy() -> None:
    strategy = create_constraint_strategy(
        backend=ConstraintBackend.CONSTRAINT_TABLE,
        tokenizer=object(),
        catalog_path="catalog.json",
    )
    assert strategy.execution_site == "worker"


def test_parent_candidate_builder_keeps_ragged_parent_mapping() -> None:
    builder = ParentCandidateBuilder(CandidateLayout.STANDARD_SAMPLER)
    builder.add_parent(
        parent_index=0,
        parent_cumulative_logprob=-0.5,
        token_ids=[],
        local_logprobs=[],
        ranks=None,
        decoded_tokens=None,
    )
    builder.add_parent(
        parent_index=1,
        parent_cumulative_logprob=-0.2,
        token_ids=[7, 8],
        local_logprobs=[-0.1, -0.3],
        ranks=[1, 2],
        decoded_tokens=["7", "8"],
    )

    batch = builder.build()

    assert batch.token_ids.tolist() == [7, 8]
    assert batch.cumulative_logprobs == pytest.approx(np.asarray([-0.3, -0.5]))
    assert batch.parent_indices.tolist() == [1, 1]
    assert batch.parent_offsets.tolist() == [0, 0, 2]
    assert len(batch.parent_cache) == 2


def test_coordinator_skips_frontend_filtering_for_worker_strategy() -> None:
    strategy = create_constraint_strategy(
        backend=ConstraintBackend.CONSTRAINT_TABLE,
        tokenizer=object(),
        catalog_path="catalog.json",
    )
    coordinator = BeamConstraintCoordinator(strategy)
    assert coordinator.frontend_enabled is False
    assert coordinator.valid_next_token_ids([1]) is None
    tokens, logprobs, _, _ = coordinator.filter_candidates(
        token_ids=[9, 2],
        logprobs=[-0.1, -0.2],
        ranks=None,
        decoded_tokens=None,
        valid_token_ids=None,
        candidate_strategy="fill",
    )
    assert tokens == [9, 2]
    assert logprobs == [-0.1, -0.2]
