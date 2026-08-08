# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Constraint strategy contracts, implementations, and factory helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

from vllm_gr.config import ConstraintBackend
from vllm_gr.utils.catalog import Catalog

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.tokenizers import TokenizerLike


class ConstraintStrategy(ABC):
    """Base class for request-independent generation constraints."""

    backend: ConstraintBackend
    execution_site: Literal["frontend", "worker"]

    @abstractmethod
    def valid_next_token_ids(self, token_ids: Sequence[int]) -> set[int]:
        """Return valid next token IDs for a generated prefix."""

    @abstractmethod
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
        """Apply a frontend constraint to one parent's candidate row."""


class TrieConstraintStrategy(ConstraintStrategy):
    """Catalog Trie implementation of the common constraint strategy."""

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
        # Beam tokens are already represented as lists. Avoid copying the
        # generated prefix for every beam on every decode step.
        return self.catalog.valid(token_ids)

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
            token_ids if isinstance(token_ids, list) else list(token_ids),
            logprobs if isinstance(logprobs, list) else list(logprobs),
            (ranks if isinstance(ranks, list) else list(ranks)) if ranks is not None else None,
            (decoded_tokens if isinstance(decoded_tokens, list) else list(decoded_tokens))
            if decoded_tokens is not None
            else None,
            valid_token_ids,
            candidate_strategy,
        )


class ConstraintTableStrategy(ConstraintStrategy):
    """Worker-side strategy reserved for the Constraint Table migration.

    Phase one establishes the type and factory branch. Resource compilation,
    Worker propagation, and ``rec_constrained_topk`` execution are delivered
    by the following implementation phases.
    """

    backend = ConstraintBackend.CONSTRAINT_TABLE
    execution_site = "worker"

    def __init__(self, constraint_data_path: str) -> None:
        self.constraint_data_path = constraint_data_path

    def valid_next_token_ids(self, token_ids: Sequence[int]) -> set[int]:
        del token_ids
        raise RuntimeError("Constraint Table candidates are generated in the Worker")

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
        del token_ids, logprobs, ranks, decoded_tokens, valid_token_ids, candidate_strategy
        raise RuntimeError("Constraint Table does not apply frontend candidate filtering")


def create_constraint_strategy(
    *,
    backend: ConstraintBackend | str,
    tokenizer: TokenizerLike,
    catalog_path: str | None,
) -> ConstraintStrategy | None:
    """Create the configured strategy without silently changing backends."""

    normalized_backend = (
        backend if isinstance(backend, ConstraintBackend) else ConstraintBackend(backend.lower())
    )
    if catalog_path is None:
        if normalized_backend is ConstraintBackend.CONSTRAINT_TABLE:
            raise ValueError("constraint_backend='constraint_table' requires catalog_path")
        return None
    if normalized_backend is ConstraintBackend.TRIE:
        return TrieConstraintStrategy.load(tokenizer, catalog_path)

    # Keep the concrete strategy available to callers and tests, but fail at
    # the phase boundary instead of accidentally running the Trie path.
    strategy = ConstraintTableStrategy(catalog_path)
    raise NotImplementedError(
        f"constraint_backend={strategy.backend.value!r} is defined but its Worker/NPU "
        "rec_constrained_topk runtime is not implemented in phase one"
    )
