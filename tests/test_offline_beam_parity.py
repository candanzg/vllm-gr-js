# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.inputs import tokens_input
from vllm.logprobs import FlatLogprobs

from vllm_gr.constraints import BeamConstraintCoordinator
from vllm_gr.constraints.strategies import TrieConstraintStrategy
from vllm_gr.entrypoints.gr import _parse_step_logprobs
from vllm_gr.utils.catalog import Catalog


def _beam(tokens: list[int], score: float = 0.0) -> BeamSearchSequence:
    beam = BeamSearchSequence(
        orig_prompt=tokens_input(tokens),
        tokens=tokens,
        logprobs=FlatLogprobs(),
        cum_logprob=score,
    )
    beam._lp_parent = None
    beam._lp_step_data = None
    return beam


def _result_with_flat(flat: object) -> SimpleNamespace:
    return SimpleNamespace(
        outputs=[SimpleNamespace(finish_reason=None, logprobs=flat)],
    )


def test_offline_grouped_candidates_keep_explicit_parent_indices() -> None:
    instance = SimpleNamespace(
        beams=[_beam([10], -0.5), _beam([20], -0.2)],
        prompt_len=1,
    )
    flat = SimpleNamespace(
        token_ids=[7, 3, 7, 8, 4, 8, 999],
        logprobs=[-0.01, -0.1, -0.2, -0.02, -0.3, -0.4, -9.0],
        ranks=None,
        decoded_tokens=None,
    )

    is_error, token_ids, scores, beam_idx, cache = _parse_step_logprobs(
        result=_result_with_flat(flat),
        instance=instance,
        token=1,
        logprobs_num=2,
        fork_info_j=[(0, 7), (1, 8)],
        catalog=None,
        catalog_strategy="fill",
    )

    assert not is_error
    assert token_ids == [3, 7, 4, 8]
    assert scores == pytest.approx([-0.6, -0.7, -0.5, -0.6])
    assert beam_idx == [0, 0, 1, 1]
    assert [entry[0] for entry in cache] == [[3, 7], [4, 8]]
    assert 999 not in token_ids


@pytest.mark.parametrize(  # type: ignore[misc]
    ("catalog_strategy", "expected_token_ids", "expected_logprobs"),
    [
        ("fill", [3, 3], [-0.1, -10000.0]),
        ("null", [3, 7], [-0.1, -float("inf")]),
    ],
)
def test_offline_candidates_use_shared_trie_filter_contract(
    catalog_strategy: str,
    expected_token_ids: list[int],
    expected_logprobs: list[float],
) -> None:
    tokenizer = MagicMock()
    catalog = Catalog(tokenizer)
    catalog.trie = {3: {Catalog._TERMINAL_MARKER: True}}
    coordinator = BeamConstraintCoordinator(TrieConstraintStrategy(catalog))
    instance = SimpleNamespace(beams=[_beam([10], -0.5)], prompt_len=1)
    flat = SimpleNamespace(
        token_ids=[7, 3, 7],
        logprobs=[-0.01, -0.1, -0.2],
        ranks=None,
        decoded_tokens=None,
        start_indices=[0],
        end_indices=[3],
    )

    is_error, token_ids, scores, beam_idx, cache = _parse_step_logprobs(
        result=_result_with_flat(flat),
        instance=instance,
        token=0,
        logprobs_num=2,
        fork_info_j=None,
        catalog_strategy=catalog_strategy,
        constraint_coordinator=coordinator,
    )

    assert not is_error
    assert token_ids == expected_token_ids
    assert scores == pytest.approx([-0.5 + logprob for logprob in expected_logprobs])
    assert beam_idx == [0, 0]
    assert list(cache[0][0]) == expected_token_ids
