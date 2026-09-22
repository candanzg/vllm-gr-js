# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep an admitted GR session out of the unused native sampler batch."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

from vllm.sampling_params import SamplingType
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
)
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_gr.v1.engine.beam_search_v1 import GRStageKind
from vllm_gr.v1.worker.gpu_prefill_graph_inputs import (
    PreparedGRPrefillInputs,
    try_prepare_gr_prefill_graph_inputs,
)

_BUILTIN_PROCESSORS = (LogitBiasLogitsProcessor, MinPLogitsProcessor, MinTokensLogitsProcessor)
_MARKER = "_gr_isolated_state_patch"


@dataclass(slots=True)
class GRWorkerRequest:
    owner: Any
    binding: Any
    generation: int
    request: CachedRequestState


def try_begin_gr_worker_request(
    runner: Any, output: Any, buf: Any, bucket: int
) -> PreparedGRPrefillInputs | None:
    """Called inside the input fence, after complete Prefill graph admission."""
    if not getattr(runner._update_states, _MARKER, False):
        return None
    batch = runner.input_batch
    owner = getattr(runner, "_gr_gpu_stage_runner", None)
    if (
        type(batch) is not InputBatch
        or batch.num_reqs
        or batch.req_ids
        or batch.batch_update_builder.batch_changed
        or runner.requests
        or runner.num_prompt_logprobs
        or getattr(runner, "_gr_worker_request", None) is not None
        or owner is None
        or owner.closed
        or owner.worker is not runner
        or output.finished_req_ids
        or output.free_encoder_mm_hashes
        or output.new_block_ids_to_zero
        or runner.speculative_config is not None
        or batch.thinking_budget_state_holder is not None
        or any(type(proc) not in _BUILTIN_PROCESSORS for proc in batch.logitsprocs.all)
    ):
        return None
    request = output.scheduled_new_reqs[0]
    params = request.sampling_params
    if (
        owner.binding.native_request_id != request.req_id
        or params is None
        or params.sampling_type != SamplingType.GREEDY
        or params.prompt_logprobs is not None
        or request.prompt_is_token_ids is not None
    ):
        return None
    state = CachedRequestState(
        req_id=request.req_id,
        prompt_token_ids=request.prompt_token_ids,
        prompt_embeds=None,
        prompt_is_token_ids=None,
        mm_features=[],
        sampling_params=params,
        pooling_params=None,
        generator=None,
        block_ids=request.block_ids,
        num_computed_tokens=request.num_computed_tokens,
        output_token_ids=[],
        lora_request=None,
    )
    prepared = try_prepare_gr_prefill_graph_inputs(runner, output, buf, bucket, new_request=request)
    if prepared is None:
        return None
    runner.requests[state.req_id] = state
    runner._gr_worker_request = GRWorkerRequest(
        owner, owner.binding, output.gr_stage_metadata[state.req_id].owner_generation, state
    )
    return prepared


def make_update_states_with_gr(original: Callable[..., Any]) -> Callable[..., Any]:
    """Native methods remain the fallback, including after materialization."""
    if getattr(original, _MARKER, False):
        return original

    @wraps(original)
    def update_states(self: Any, scheduler_output: Any) -> Any:
        record = getattr(self, "_gr_worker_request", None)
        if record is None:
            return original(self, scheduler_output)
        state = record.request
        req_id = state.req_id
        if (
            self.requests.get(req_id) is not state
            or getattr(self, "_gr_gpu_stage_runner", None) is not record.owner
            or record.owner.binding != record.binding
        ):
            raise RuntimeError("GR Worker request no longer belongs to its execution owner")
        if req_id in scheduler_output.finished_req_ids:
            # No native row or processor entry was created. Native cleanup
            # still handles all finished IDs, including resubmission of this ID.
            self._gr_worker_request = None
            return original(self, scheduler_output)

        stages = getattr(scheduler_output, "gr_stage_metadata", None)
        stage = stages.get(req_id) if stages and len(stages) == 1 else None
        if stage is not None and stage.owner_generation != record.generation:
            raise RuntimeError("GR Worker request no longer belongs to its execution owner")
        cached = scheduler_output.scheduled_cached_reqs
        if (
            not stages
            and not self.input_batch.num_reqs
            and not scheduler_output.total_num_scheduled_tokens
            and not scheduler_output.num_scheduled_tokens
            and not scheduler_output.scheduled_new_reqs
            and not cached.req_ids
            and not scheduler_output.finished_req_ids
            and not scheduler_output.free_encoder_mm_hashes
            and not scheduler_output.new_block_ids_to_zero
            and not scheduler_output.scheduled_spec_decode_tokens
            and not scheduler_output.scheduled_encoder_inputs
        ):
            # Native empty ticks can precede terminal receipt delivery. Their
            # normal cleanup preserves cached requests; no row is needed.
            return original(self, scheduler_output)
        if (
            stage is not None
            and stage.kind is GRStageKind.DECODE
            and stage.owner_generation == record.generation
            and not record.owner.closed
            and not self.input_batch.num_reqs
            and not scheduler_output.scheduled_new_reqs
            and len(scheduler_output.num_scheduled_tokens) == 1
            and req_id in scheduler_output.num_scheduled_tokens
            and cached.req_ids == [req_id]
            and not cached.resumed_req_ids
            and not scheduler_output.finished_req_ids
            and not scheduler_output.free_encoder_mm_hashes
            and not scheduler_output.new_block_ids_to_zero
            and not scheduler_output.scheduled_spec_decode_tokens
            and not scheduler_output.scheduled_encoder_inputs
            and not state.prev_num_draft_len
            and (
                cached.new_block_ids[0] is None
                or len(cached.new_block_ids[0]) == len(state.block_ids)
            )
        ):
            state.num_computed_tokens = cached.num_computed_tokens[0]
            count = cached.num_output_tokens[0]
            if count < len(state.output_token_ids):
                del state.output_token_ids[count:]
            if cached.new_block_ids[0] is not None:
                for blocks, added in zip(state.block_ids, cached.new_block_ids[0]):
                    blocks.extend(added)
            return None

        # Unsupported scheduling cannot treat this as an absent-row resume:
        # the scheduler need not send all_token_ids for a resident GR request.
        # Reconstruct the row BEFORE applying any cached-request deltas.
        self.input_batch.add_request(state)
        self._gr_worker_request = None
        return original(self, scheduler_output)

    setattr(update_states, _MARKER, True)
    return update_states
