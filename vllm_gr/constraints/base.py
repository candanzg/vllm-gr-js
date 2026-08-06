# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Constraint strategy contract independent of serving and hardware backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

from vllm_gr.config import ConstraintBackend

if TYPE_CHECKING:
    from collections.abc import Sequence


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
