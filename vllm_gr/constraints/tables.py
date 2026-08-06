# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Constraint Tables strategy boundary for the later Worker/NPU phase."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_gr.config import ConstraintBackend
from vllm_gr.constraints.base import ConstraintStrategy

if TYPE_CHECKING:
    from collections.abc import Sequence


class ConstraintTablesStrategy(ConstraintStrategy):
    """Worker-side strategy reserved for the Constraint Tables migration.

    Phase one establishes the type and factory branch. Resource compilation,
    Worker propagation, and ``rec_constrained_topk`` execution are delivered
    by the following implementation phases.
    """

    backend = ConstraintBackend.CONSTRAINT_TABLES
    execution_site = "worker"

    def __init__(self, constraint_data_path: str) -> None:
        self.constraint_data_path = constraint_data_path

    def valid_next_token_ids(self, token_ids: Sequence[int]) -> set[int]:
        del token_ids
        raise RuntimeError("Constraint Tables candidates are generated in the Worker")

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
        raise RuntimeError("Constraint Tables does not apply frontend candidate filtering")
