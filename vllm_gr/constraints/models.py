# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Hardware-neutral models at the constraint/beam-decision boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import numpy as np


class CandidateLayout(str, Enum):
    """Describes how a worker encoded per-parent candidate rows."""

    STANDARD_SAMPLER = "standard_sampler"
    CONSTRAINED_TOPK = "constrained_topk"


BeamCandidateCache: TypeAlias = tuple[
    Sequence[int],
    Sequence[float],
    Sequence[int | None] | None,
    Sequence[str | None] | None,
]


@dataclass(frozen=True)
class ParentCandidateBatch:
    """Flattened, explicitly parent-mapped candidates for one beam decision."""

    token_ids: np.ndarray
    cumulative_logprobs: np.ndarray
    parent_indices: np.ndarray
    parent_offsets: np.ndarray
    parent_cache: list[BeamCandidateCache]
    layout: CandidateLayout = CandidateLayout.STANDARD_SAMPLER

    def __post_init__(self) -> None:
        candidate_count = len(self.token_ids)
        if len(self.cumulative_logprobs) != candidate_count:
            raise ValueError("candidate token_ids and cumulative_logprobs must have equal length")
        if len(self.parent_indices) != candidate_count:
            raise ValueError("candidate token_ids and parent_indices must have equal length")
        if len(self.parent_offsets) != len(self.parent_cache) + 1:
            raise ValueError("parent_offsets must contain one boundary per parent plus the end")
        if len(self.parent_offsets) == 0 or int(self.parent_offsets[0]) != 0:
            raise ValueError("parent_offsets must start at zero")
        if int(self.parent_offsets[-1]) != candidate_count:
            raise ValueError("parent_offsets must end at the flattened candidate count")


class ParentCandidateBuilder:
    """Build a ragged candidate batch without inferring parents from stride."""

    def __init__(self, layout: CandidateLayout = CandidateLayout.STANDARD_SAMPLER) -> None:
        self.layout = layout
        self._token_ids: list[int] = []
        self._cumulative_logprobs: list[float] = []
        self._parent_indices: list[int] = []
        self._parent_offsets: list[int] = [0]
        self._parent_cache: list[BeamCandidateCache] = []

    def add_parent(
        self,
        *,
        parent_index: int,
        parent_cumulative_logprob: float,
        token_ids: Sequence[int],
        local_logprobs: Sequence[float],
        ranks: Sequence[int | None] | None,
        decoded_tokens: Sequence[str | None] | None,
    ) -> None:
        if len(token_ids) != len(local_logprobs):
            raise ValueError("candidate token_ids and local_logprobs must have equal length")
        cache: BeamCandidateCache = (token_ids, local_logprobs, ranks, decoded_tokens)
        self._parent_cache.append(cache)
        self._token_ids.extend(int(token_id) for token_id in token_ids)
        self._cumulative_logprobs.extend(
            parent_cumulative_logprob + float(logprob) for logprob in local_logprobs
        )
        self._parent_indices.extend([parent_index] * len(token_ids))
        self._parent_offsets.append(len(self._token_ids))

    def build(self) -> ParentCandidateBatch:
        return ParentCandidateBatch(
            token_ids=np.asarray(self._token_ids, dtype=np.int64),
            cumulative_logprobs=np.asarray(self._cumulative_logprobs, dtype=np.float64),
            parent_indices=np.asarray(self._parent_indices, dtype=np.intp),
            parent_offsets=np.asarray(self._parent_offsets, dtype=np.intp),
            parent_cache=self._parent_cache,
            layout=self.layout,
        )
