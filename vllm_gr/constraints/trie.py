# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Catalog Trie implementation of the common constraint strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_gr.config import ConstraintBackend
from vllm_gr.constraints.base import ConstraintStrategy
from vllm_gr.utils.catalog import Catalog

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.tokenizers import TokenizerLike


class TrieConstraintStrategy(ConstraintStrategy):
    backend = ConstraintBackend.TRIE
    execution_site = "frontend"

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    @classmethod
    def load(cls, tokenizer: TokenizerLike, catalog_path: str) -> TrieConstraintStrategy:
        catalog = Catalog(tokenizer)
        catalog.load(catalog_path)
        return cls(catalog)

    def valid_next_token_ids(self, token_ids: Sequence[int]) -> set[int]:
        return self.catalog.valid(list(token_ids))

    def filter_frontend_candidates(
        self,
        token_ids: Sequence[int],
        logprobs: Sequence[float],
        ranks: Sequence[int | None] | None,
        decoded_tokens: Sequence[str | None] | None,
        valid_token_ids: set[int],
        candidate_strategy: str,
    ) -> tuple[
        Sequence[int],
        Sequence[float],
        Sequence[int | None] | None,
        Sequence[str | None] | None,
    ]:
        return self.catalog.apply_catalog_filter(
            list(token_ids),
            list(logprobs),
            list(ranks) if ranks is not None else None,
            list(decoded_tokens) if decoded_tokens is not None else None,
            valid_token_ids,
            candidate_strategy,
        )
