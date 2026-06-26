# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pooling module that exposes recommendation models to vLLM's pooling API.

For recommendation models, the "pooling" operation is essentially returning
the item scores/predictions rather than text embeddings, so this is a thin
identity-style adapter that satisfies vLLM's pooling-model contract.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import torch
import torch.nn as nn


class RecommendationPooler(nn.Module):
    """Pooler adapter that lets vLLM treat recommendation models as pooling models."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.dense = nn.Identity()

    def get_supported_tasks(self) -> set:
        """Valid PoolingTask values: "embed", "classify", "score", etc."""
        return {"embed"}

    def get_pooling_updates(self, task: str) -> Any:
        from vllm.model_executor.layers.pooler.common import PoolingParamsUpdate

        return PoolingParamsUpdate()

    def forward(
        self,
        hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        pooling_metadata: Optional[Any] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(hidden_states, list):
            return [self.dense(h) for h in hidden_states]
        return self.dense(hidden_states)


__all__ = ["RecommendationPooler"]
