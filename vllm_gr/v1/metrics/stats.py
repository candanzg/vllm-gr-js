# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field

from vllm.v1.metrics.stats import RequestStateStats as BaseRequestStateStats


@dataclass
class RequestStateStats(BaseRequestStateStats):
    """Stats that need to be tracked across delta updates."""

    # Beam search overhead
    beam_search_overhead: float = 0.0
    beam_search_decode_time: float = 0.0
    num_generation_tokens: int = 0
    stage_breakdown_ms: dict[str, float] = field(default_factory=dict)
    beam_engine_queue_ms: float = 0.0
    beam_engine_prefill_ms: float = 0.0
    beam_engine_decode_ms: float = 0.0
    beam_engine_inference_ms: float = 0.0
    beam_engine_e2e_ms: float = 0.0
    beam_stats: dict[str, float] = field(default_factory=dict)
