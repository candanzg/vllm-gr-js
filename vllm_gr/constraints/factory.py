# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Factory for process-level constraint strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_gr.config import ConstraintBackend
from vllm_gr.constraints.base import ConstraintStrategy
from vllm_gr.constraints.tables import ConstraintTablesStrategy
from vllm_gr.constraints.trie import TrieConstraintStrategy

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike


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
        if normalized_backend is ConstraintBackend.CONSTRAINT_TABLES:
            raise ValueError("constraint_backend='constraint_tables' requires catalog_path")
        return None
    if normalized_backend is ConstraintBackend.TRIE:
        return TrieConstraintStrategy.load(tokenizer, catalog_path)

    # Keep the concrete strategy available to callers and tests, but fail at
    # the phase boundary instead of accidentally running the Trie path.
    strategy = ConstraintTablesStrategy(catalog_path)
    raise NotImplementedError(
        f"constraint_backend={strategy.backend.value!r} is defined but its Worker/NPU "
        "rec_constrained_topk runtime is not implemented in phase one"
    )
