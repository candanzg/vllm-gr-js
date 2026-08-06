# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm_gr.config import ConstraintBackend
from vllm_gr.constraints import (
    BeamConstraintCoordinator,
    CandidateLayout,
    ParentCandidateBuilder,
    TrieConstraintStrategy,
    create_constraint_strategy,
)


@pytest.fixture  # type: ignore[misc]
def tokenizer() -> MagicMock:
    result = MagicMock()
    result.convert_tokens_to_ids.side_effect = lambda values: [int(value) for value in values]
    return result


def test_trie_factory_loads_catalog_and_preserves_compatibility(
    tokenizer: MagicMock,
) -> None:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as file:
        json.dump([["1", "2", "3"], ["1", "4", "5"]], file)
        catalog_path = file.name
    try:
        strategy = create_constraint_strategy(
            backend="TRIE",
            tokenizer=tokenizer,
            catalog_path=catalog_path,
        )

        assert isinstance(strategy, TrieConstraintStrategy)
        assert strategy.valid_next_token_ids([]) == {1}
        assert strategy.valid_next_token_ids([1]) == {2, 4}
    finally:
        os.remove(catalog_path)


def test_constraint_tables_factory_fails_at_phase_boundary(tokenizer: MagicMock) -> None:
    with pytest.raises(NotImplementedError, match="rec_constrained_topk"):
        create_constraint_strategy(
            backend=ConstraintBackend.CONSTRAINT_TABLES,
            tokenizer=tokenizer,
            catalog_path="catalog.json",
        )


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


def test_coordinator_applies_fill_and_null_with_the_same_contract(
    tokenizer: MagicMock,
) -> None:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as file:
        json.dump([["1", "2"], ["1", "4"]], file)
        catalog_path = file.name
    try:
        strategy = TrieConstraintStrategy.load(tokenizer, catalog_path)
        coordinator = BeamConstraintCoordinator(strategy)
        valid = coordinator.valid_next_token_ids([1])

        fill_tokens, fill_logprobs, _, _ = coordinator.filter_candidates(
            token_ids=[9, 2],
            logprobs=[-0.1, -0.2],
            ranks=None,
            decoded_tokens=None,
            valid_token_ids=valid,
            candidate_strategy="fill",
        )
        null_tokens, null_logprobs, _, _ = coordinator.filter_candidates(
            token_ids=[9, 2],
            logprobs=[-0.1, -0.2],
            ranks=None,
            decoded_tokens=None,
            valid_token_ids=valid,
            candidate_strategy="null",
        )

        assert fill_tokens == [2, 2]
        assert fill_logprobs == pytest.approx([-10000.0, -0.2])
        assert null_tokens == [9, 2]
        assert null_logprobs[0] == -float("inf")
        assert null_logprobs[1] == pytest.approx(-0.2)
    finally:
        os.remove(catalog_path)
