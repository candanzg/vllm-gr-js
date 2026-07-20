# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import copy
import time
from collections.abc import AsyncGenerator, Mapping
from contextlib import nullcontext

import numpy as np
from vllm.beam_search import BeamSearchSequence
from vllm.entrypoints.openai.protocol import VLLMValidationError
from vllm.inputs.data import PromptType, TokensPrompt
from vllm.inputs.parse import is_explicit_encoder_decoder_prompt
from vllm.logger import init_logger
from vllm.logprobs import FlatLogprobs
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.utils import random_uuid

from vllm_gr.logprobs import extract_and_dedup_flat_logprobs, reconstruct_beam_logprobs
from vllm_gr.utils.perf import RequestPerfTrace, log_perf_event
from vllm_gr.v1.engine.types import BeamRequestStepUpdate
from vllm_gr.v1.metrics.beam_search_metrics import record_beam_search_metrics
from vllm_gr.v1.metrics.stats import RequestStateStats
from vllm.v1.metrics.function_profile import function_profile, profiled_function

logger = init_logger(__name__)

_dp_rank_counter = 0
_dp_rank_lock = asyncio.Lock()


# return rank for next beam search request in a round-robin manner across the data parallel ranks
async def _next_data_parallel_rank(data_parallel_size: int) -> int:
    global _dp_rank_counter

    async with _dp_rank_lock:
        rank = _dp_rank_counter % data_parallel_size
        _dp_rank_counter += 1
        return rank


def _emit_request_phase_us(
    request_trace: RequestPerfTrace,
    phase_name: str,
    phase_us: float | None,
    **fields,
) -> None:
    if phase_us is None or phase_us <= 0.0:
        return
    log_perf_event(
        request_trace.module,
        phase_name,
        float(phase_us) / 1_000_000.0,
        request_id=request_trace.request_id,
        phase=phase_name,
        event="request_phase",
        **fields,
    )


def _emit_collector_timing_phases(
    request_trace: RequestPerfTrace,
    collector_stats: dict[str, float | int | None],
    *,
    phase_prefix: str,
    **fields,
) -> None:
    phase_pairs = (
        (f"{phase_prefix}_engine_queue_wait", "prefill_engine_queue_wait_us"),
        (f"{phase_prefix}_engine_compute", "prefill_engine_compute_us"),
        (f"{phase_prefix}_exec_submit", "prefill_exec_submit_us"),
        (f"{phase_prefix}_sample_submit", "prefill_sample_submit_us"),
        (
            f"{phase_prefix}_build_grammar_output",
            "prefill_build_grammar_output_us",
        ),
        (f"{phase_prefix}_wait_model_output", "prefill_wait_model_output_us"),
        (f"{phase_prefix}_update_from_output", "prefill_update_from_output_us"),
        (f"{phase_prefix}_worker_total", "prefill_worker_total_us"),
        (f"{phase_prefix}_worker_execute_model", "prefill_worker_execute_model_us"),
        (f"{phase_prefix}_worker_forward", "prefill_worker_forward_us"),
        (
            f"{phase_prefix}_worker_sample_tokens_body",
            "prefill_worker_sample_tokens_body_us",
        ),
        (
            f"{phase_prefix}_worker_profile_event_sync",
            "prefill_worker_profile_event_sync_us",
        ),
        (
            f"{phase_prefix}_worker_sample_tokens_total",
            "prefill_worker_sample_tokens_total_us",
        ),
        (f"{phase_prefix}_worker_sample", "prefill_worker_sample_us"),
        (f"{phase_prefix}_worker_bookkeeping", "prefill_worker_bookkeeping_us"),
        (f"{phase_prefix}_worker_build_output", "prefill_worker_build_output_us"),
        (
            f"{phase_prefix}_worker_async_output_wrap",
            "prefill_worker_async_output_wrap_us",
        ),
        (f"{phase_prefix}_process_aborts", "prefill_process_aborts_us"),
        (
            f"{phase_prefix}_perf_metrics_propagate",
            "prefill_perf_metrics_propagate_us",
        ),
        (f"{phase_prefix}_step_total_wall", "prefill_step_total_wall_us"),
        (
            f"{phase_prefix}_async_output_queue_wait",
            "prefill_async_output_queue_wait_us",
        ),
        (
            f"{phase_prefix}_async_output_get_output",
            "prefill_async_output_get_output_us",
        ),
        (
            f"{phase_prefix}_async_gpu_output_sync",
            "prefill_async_gpu_output_sync_us",
        ),
        (
            f"{phase_prefix}_async_gpu_output_sampled_token_ids",
            "prefill_async_gpu_output_sampled_token_ids_us",
        ),
        (
            f"{phase_prefix}_async_gpu_output_logprobs_tolist",
            "prefill_async_gpu_output_logprobs_tolist_us",
        ),
        (
            f"{phase_prefix}_async_gpu_output_parse_output",
            "prefill_async_gpu_output_parse_output_us",
        ),
        (
            f"{phase_prefix}_async_gpu_output_assign_output",
            "prefill_async_gpu_output_assign_output_us",
        ),
        (
            f"{phase_prefix}_async_gpu_output_total",
            "prefill_async_gpu_output_total_us",
        ),
        (f"{phase_prefix}_core_to_frontend", "prefill_core_to_frontend_us"),
        (f"{phase_prefix}_frontend_handoff", "prefill_frontend_handoff_us"),
        (f"{phase_prefix}_queue_visible", "prefill_queue_visible_us"),
        (f"{phase_prefix}_submit_to_visible", "prefill_submit_to_visible_us"),
    )
    for phase_name, stats_key in phase_pairs:
        phase_us = collector_stats.get(stats_key)
        _emit_request_phase_us(request_trace, phase_name, phase_us, **fields)


@profiled_function("vllm_gr.entrypoints.openai.serving_engine", "_collect_beam_result")
async def _collect_beam_result(
    q,
    request_trace: RequestPerfTrace | None = None,
    *,
    queue_index: int | None = None,
    queue_count: int | None = None,
):
    """Drain output queue until finished and return simple wait stats."""
    wait_start = time.perf_counter()
    message_count = 0
    queue_fields: dict[str, int] = {}
    if queue_index is not None:
        queue_fields["queue_index"] = queue_index
    if queue_count is not None:
        queue_fields["queue_count"] = queue_count
    with function_profile(
        "vllm_gr.entrypoints.openai.serving_engine", "collect_wait_first_message"
    ):
        with (
            request_trace.phase("collect_wait_first_message", **queue_fields)
            if request_trace
            else nullcontext()
        ):
            out = await q.get()
    message_count = 1
    first_message_at = time.perf_counter()
    while not out.finished:
        with function_profile(
            "vllm_gr.entrypoints.openai.serving_engine", "collect_wait_tail_messages"
        ):
            with (
                request_trace.phase("collect_wait_tail_messages", **queue_fields)
                if request_trace
                else nullcontext()
            ):
                out = await q.get()
        message_count += 1
    finished_at = time.perf_counter()
    submit_ts = getattr(q, "perf_submit_ts", 0.0) or 0.0
    queue_put_ts = getattr(q, "perf_queue_put_ts", 0.0) or 0.0
    return out, {
        "message_count": message_count,
        "wait_ms": round((finished_at - wait_start) * 1000.0, 3),
        "first_wait_ms": round((first_message_at - wait_start) * 1000.0, 3),
        "tail_wait_ms": round((finished_at - first_message_at) * 1000.0, 3),
        "prefill_submit_to_visible_us": round(
            max(0.0, (first_message_at - submit_ts) * 1_000_000.0), 3
        )
        if submit_ts
        else None,
        "prefill_queue_visible_us": round(
            max(0.0, (first_message_at - queue_put_ts) * 1_000_000.0), 3
        )
        if queue_put_ts
        else None,
        "prefill_engine_queue_wait_us": round(
            float(getattr(q, "perf_queued_to_scheduled_us", 0.0)), 3
        )
        or None,
        "prefill_engine_compute_us": round(
            float(getattr(q, "perf_scheduled_to_first_token_us", 0.0)), 3
        )
        or None,
        "prefill_core_to_frontend_us": round(
            float(getattr(q, "perf_core_to_frontend_us", 0.0)), 3
        )
        or None,
        "prefill_frontend_handoff_us": round(
            float(getattr(q, "perf_frontend_handoff_us", 0.0)), 3
        )
        or None,
        "prefill_exec_submit_us": round(
            float(getattr(q, "perf_prefill_exec_submit_us", 0.0)), 3
        )
        or None,
        "prefill_sample_submit_us": round(
            float(getattr(q, "perf_prefill_sample_submit_us", 0.0)), 3
        )
        or None,
        "prefill_build_grammar_output_us": round(
            float(getattr(q, "perf_prefill_build_grammar_output_us", 0.0)), 3
        )
        or None,
        "prefill_wait_model_output_us": round(
            float(getattr(q, "perf_prefill_wait_model_output_us", 0.0)), 3
        )
        or None,
        "prefill_update_from_output_us": round(
            float(getattr(q, "perf_prefill_update_from_output_us", 0.0)), 3
        )
        or None,
        "prefill_process_aborts_us": round(
            float(getattr(q, "perf_prefill_process_aborts_us", 0.0)), 3
        )
        or None,
        "prefill_perf_metrics_propagate_us": round(
            float(getattr(q, "perf_prefill_perf_metrics_propagate_us", 0.0)), 3
        )
        or None,
        "prefill_step_total_wall_us": round(
            float(getattr(q, "perf_prefill_step_total_wall_us", 0.0)), 3
        )
        or None,
        "prefill_worker_total_us": round(
            float(getattr(q, "perf_prefill_worker_total_us", 0.0)), 3
        )
        or None,
        "prefill_worker_execute_model_us": round(
            float(getattr(q, "perf_prefill_worker_execute_model_us", 0.0)), 3
        )
        or None,
        "prefill_worker_forward_us": round(
            float(getattr(q, "perf_prefill_worker_forward_us", 0.0)), 3
        )
        or None,
        "prefill_worker_sample_tokens_body_us": round(
            float(getattr(q, "perf_prefill_worker_sample_tokens_body_us", 0.0)), 3
        )
        or None,
        "prefill_worker_profile_event_sync_us": round(
            float(getattr(q, "perf_prefill_worker_profile_event_sync_us", 0.0)), 3
        )
        or None,
        "prefill_worker_sample_tokens_total_us": round(
            float(getattr(q, "perf_prefill_worker_sample_tokens_total_us", 0.0)), 3
        )
        or None,
        "prefill_worker_sample_us": round(
            float(getattr(q, "perf_prefill_worker_sample_us", 0.0)), 3
        )
        or None,
        "prefill_worker_bookkeeping_us": round(
            float(getattr(q, "perf_prefill_worker_bookkeeping_us", 0.0)), 3
        )
        or None,
        "prefill_worker_build_output_us": round(
            float(getattr(q, "perf_prefill_worker_build_output_us", 0.0)), 3
        )
        or None,
        "prefill_worker_async_output_wrap_us": round(
            float(getattr(q, "perf_prefill_worker_async_output_wrap_us", 0.0)), 3
        )
        or None,
        "prefill_async_output_queue_wait_us": round(
            float(getattr(q, "perf_prefill_async_output_queue_wait_us", 0.0)), 3
        )
        or None,
        "prefill_async_output_get_output_us": round(
            float(getattr(q, "perf_prefill_async_output_get_output_us", 0.0)), 3
        )
        or None,
        "prefill_async_gpu_output_sync_us": round(
            float(getattr(q, "perf_prefill_async_gpu_output_sync_us", 0.0)), 3
        )
        or None,
        "prefill_async_gpu_output_sampled_token_ids_us": round(
            float(getattr(q, "perf_prefill_async_gpu_output_sampled_token_ids_us", 0.0)), 3
        )
        or None,
        "prefill_async_gpu_output_logprobs_tolist_us": round(
            float(getattr(q, "perf_prefill_async_gpu_output_logprobs_tolist_us", 0.0)), 3
        )
        or None,
        "prefill_async_gpu_output_parse_output_us": round(
            float(getattr(q, "perf_prefill_async_gpu_output_parse_output_us", 0.0)), 3
        )
        or None,
        "prefill_async_gpu_output_assign_output_us": round(
            float(getattr(q, "perf_prefill_async_gpu_output_assign_output_us", 0.0)), 3
        )
        or None,
        "prefill_async_gpu_output_total_us": round(
            float(getattr(q, "perf_prefill_async_gpu_output_total_us", 0.0)), 3
        )
        or None,
    }


@profiled_function("vllm_gr.entrypoints.openai.serving_engine", "_gather_beam_results")
async def _gather_beam_results(
    queues,
    request_trace: RequestPerfTrace | None = None,
):
    """Gather results from multiple output queues."""
    tasks = [
        asyncio.create_task(
            _collect_beam_result(
                q,
                request_trace=request_trace,
                queue_index=idx,
                queue_count=len(queues),
            )
        )
        for idx, q in enumerate(queues)
    ]
    with function_profile(
        "vllm_gr.entrypoints.openai.serving_engine", "await_collect_tasks"
    ):
        gathered = list(await asyncio.gather(*tasks))
    outputs = [item[0] for item in gathered]
    stats = [item[1] for item in gathered]
    return outputs, stats


@profiled_function("vllm_gr.entrypoints.openai.serving_engine", "_add_batch_step")
async def _add_batch_step(
    engine_client,
    prompts_batch,
    lora_req_batch,
    request_id_batch,
    beam_search_params,
    use_beam_req,
    trace_headers,
    priority=0,
    data_parallel_rank: int | None = None,
    request_trace: RequestPerfTrace | None = None,
):
    """ADD_BATCH path: prepare and batch-send requests (step 0)."""
    step_fields = {"step_mode": "add_batch"}
    queues = []
    engine_core_requests = []
    with (
        request_trace.phase("prepare_request_batch", **step_fields)
        if request_trace
        else nullcontext()
    ):
        for i, (individual_prompt, lora_req) in enumerate(zip(prompts_batch, lora_req_batch)):
            request_id_item = f"{request_id_batch}-beam-{i}"
            q, ec_req = engine_client.prepare_request(
                request_id_item,
                individual_prompt,
                beam_search_params,
                lora_request=lora_req,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
            )
            queues.append(q)
            engine_core_requests.append(ec_req)

    with (
        request_trace.phase("add_requests_batch", batch_size=len(engine_core_requests), **step_fields)
        if request_trace
        else nullcontext()
    ):
        submit_ts = time.monotonic()
        for q in queues:
            q.perf_submit_ts = submit_ts
        await engine_client._add_requests_batch(
            engine_core_requests, use_batch_message=use_beam_req
        )

    internal_ids = [r.request_id for r in engine_core_requests] if use_beam_req else []
    with (
        request_trace.phase("gather_beam_results", queue_count=len(queues), **step_fields)
        if request_trace
        else nullcontext()
    ):
        output, gather_stats = await _gather_beam_results(
            queues, request_trace=request_trace
        )
    if request_trace and gather_stats:
        def _stats_values(key: str) -> list[float]:
            return [float(stat[key]) for stat in gather_stats if stat.get(key) is not None]

        representative = max(
            gather_stats,
            key=lambda stat: (
                stat.get("prefill_submit_to_visible_us") or 0.0,
                stat.get("first_wait_ms") or 0.0,
            ),
        )
        with function_profile(
            "vllm_gr.entrypoints.openai.serving_engine",
            "prefill_emit_collector_metrics",
        ):
            with request_trace.phase(
                "prefill_emit_collector_metrics",
                queue_count=len(gather_stats),
                step_mode="add_batch",
            ):
                _emit_collector_timing_phases(
                    request_trace,
                    representative,
                    phase_prefix="prefill",
                    queue_count=len(gather_stats),
                    step_mode="add_batch",
                )
        for key, phase_prefix in (
            ("prefill_engine_compute_us", "prefill_engine_compute"),
            ("prefill_submit_to_visible_us", "prefill_submit_to_visible"),
        ):
            samples = _stats_values(key)
            if not samples:
                continue
            aggregates = (
                (f"{phase_prefix}_avg", sum(samples) / len(samples)),
                (f"{phase_prefix}_max", max(samples)),
                (f"{phase_prefix}_min", min(samples)),
                (f"{phase_prefix}_spread", max(samples) - min(samples)),
            )
            for phase_name, phase_us in aggregates:
                if phase_us <= 0.0:
                    continue
                log_perf_event(
                    request_trace.module,
                    phase_name,
                    phase_us / 1_000_000.0,
                    request_id=request_trace.request_id,
                    phase=phase_name,
                    event="request_phase",
                    queue_count=len(gather_stats),
                    step_mode="add_batch",
                )
        queue_count = len(gather_stats)
        if queue_count > 0:
            for key, phase_name in (
                ("prefill_engine_compute_us", "prefill_engine_compute_per_queue"),
                ("prefill_submit_to_visible_us", "prefill_submit_to_visible_per_queue"),
            ):
                samples = _stats_values(key)
                if not samples:
                    continue
                phase_us = (sum(samples) / len(samples)) / queue_count
                if phase_us <= 0.0:
                    continue
                log_perf_event(
                    request_trace.module,
                    phase_name,
                    phase_us / 1_000_000.0,
                    request_id=request_trace.request_id,
                    phase=phase_name,
                    event="request_phase",
                    queue_count=queue_count,
                    step_mode="add_batch",
                )
        with request_trace.phase(
            "gather_result_stats",
            queue_count=len(gather_stats),
            max_wait_ms=max(stat["wait_ms"] for stat in gather_stats),
            total_messages=sum(stat["message_count"] for stat in gather_stats),
            **step_fields,
        ):
            pass
    return output, internal_ids


@profiled_function("vllm_gr.entrypoints.openai.serving_engine", "_beam_request_step")
async def _beam_request_step(
    engine_client,
    session_id: str,
    fork_info,
    prev_beam_internal_ids,
    all_beams,
    request_id_batch,
    beam_search_params,
    eos_token_id,
    lora_request,
    trace_headers,
    prefix_len: int,
    priority=0,
    data_parallel_rank: int | None = None,
    request_trace: RequestPerfTrace | None = None,
):
    """BEAM_REQUEST path: single logical message update for all active branches."""
    step_fields = {"step_mode": "beam_request"}
    parent_beam_ids = []
    child_beam_ids = []
    beam_tokens = []
    beam_width = len(fork_info)

    # Map the current generation step's routing array
    with (
        request_trace.phase("build_fork_info", beam_width=beam_width, **step_fields)
        if request_trace
        else nullcontext()
    ):
        for new_idx, (parent_beam_idx, tok) in enumerate(fork_info):
            parent_id = prev_beam_internal_ids[parent_beam_idx]
            child_id = f"{request_id_batch}-beam-{new_idx}"
            parent_beam_ids.append(parent_id)
            child_beam_ids.append(child_id)

            full_tokens = all_beams[new_idx].tokens
            beam_tokens.append(full_tokens[prefix_len:])

    # -------------------------------------------------------------------------
    # Declare true active beam sizing directly to the core sampler
    # -------------------------------------------------------------------------
    decode_sampling_params = copy.copy(beam_search_params)
    decode_sampling_params.n = beam_width  # Explicitly tell the engine we have beam_width beams

    # Listen exclusively to the parent session_id stream queue
    with (
        request_trace.phase("register_beam_output", beam_width=beam_width, **step_fields)
        if request_trace
        else nullcontext()
    ):
        q = engine_client.register_beam_output(
            session_id,
            all_beams[0].tokens,
            decode_sampling_params,
            eos_token_id=eos_token_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
        )

    used_parents = set(parent_beam_ids)
    pruned_ids = [pid for pid in prev_beam_internal_ids if pid not in used_parents]

    # Dispatch the consolidated persistent update to the client
    with (
        request_trace.phase(
            "beam_request_step_update",
            beam_width=beam_width,
            pruned_count=len(pruned_ids),
            **step_fields,
        )
        if request_trace
        else nullcontext()
    ):
        q.perf_submit_ts = time.monotonic()
        await engine_client.beam_request_step_update(
            BeamRequestStepUpdate(
                session_id=session_id,
                parent_beam_ids=parent_beam_ids,
                child_beam_ids=child_beam_ids,
                beam_tokens=beam_tokens,
                pruned_ids=pruned_ids,
                prefix_len=prefix_len,
                beam_width=beam_width,
                sampling_params=decode_sampling_params,
                eos_token_id=eos_token_id,
                lora_request=lora_request,
                trace_headers=trace_headers,
                data_parallel_rank=data_parallel_rank,
                priority=priority,
            )
        )

    # Await the singular unified engine execution response block
    with (
        request_trace.phase("collect_beam_result", beam_width=beam_width, **step_fields)
        if request_trace
        else nullcontext()
    ):
        single_output, collect_stats = await _collect_beam_result(
            q,
            request_trace=request_trace,
            queue_index=0,
            queue_count=1,
        )
    if request_trace:
        with function_profile(
            "vllm_gr.entrypoints.openai.serving_engine",
            "decode_emit_collector_metrics",
        ):
            with request_trace.phase(
                "decode_emit_collector_metrics",
                beam_width=beam_width,
                **step_fields,
            ):
                _emit_collector_timing_phases(
                    request_trace,
                    collect_stats,
                    phase_prefix="decode",
                    beam_width=beam_width,
                    **step_fields,
                )
        with request_trace.phase(
            "collect_result_stats",
            beam_width=beam_width,
            message_count=collect_stats["message_count"],
            wait_ms=collect_stats["wait_ms"],
            first_wait_ms=collect_stats["first_wait_ms"],
            tail_wait_ms=collect_stats["tail_wait_ms"],
            **step_fields,
        ):
            pass

    # Return the singular raw output instance wrapped as a single-element list
    # to maintain backward-compatible pipeline execution with the loop below
    return [single_output], child_beam_ids


@profiled_function("vllm_gr.entrypoints.openai.serving_engine", "_beam_request_cleanup")
async def _beam_request_cleanup(engine_client, session_id, final_ids, rank):
    """Signals the core engine to fully purge all remaining session cache allocations."""
    if not final_ids:
        return
    try:
        await engine_client.beam_request_step_update(
            BeamRequestStepUpdate(
                session_id=session_id,
                parent_beam_ids=[],
                child_beam_ids=[],
                beam_tokens=[],
                pruned_ids=final_ids,
                prefix_len=0,
                beam_width=0,
                sampling_params=SamplingParams(),
                data_parallel_rank=rank,
            )
        )
    except Exception as e:
        logger.error("Failed to clear beam_request cache for session %s: %s", session_id, e)


def _extract_and_dedup_segment(
    raw_token_ids,
    raw_logprobs,
    raw_ranks,
    raw_decoded,
    start_offset: int,
    end_offset: int,
    logprobs_num: int,
):
    # Extract the precise segment dedicated to this beam's top alternative choices
    token_ids_pos = raw_token_ids[start_offset:end_offset]
    logprobs_pos = raw_logprobs[start_offset:end_offset]
    ranks_pos = raw_ranks[start_offset:end_offset] if raw_ranks is not None else None
    decoded_pos = raw_decoded[start_offset:end_offset] if raw_decoded is not None else None

    # Dedup the sampled token (index 0) if it appears in the top-K
    if len(token_ids_pos) > logprobs_num:
        if any(token_ids_pos[0] == token_ids_pos[j] for j in range(1, logprobs_num + 1)):
            s = slice(1, logprobs_num + 1)
        else:
            s = slice(logprobs_num)
        token_ids_pos = token_ids_pos[s]
        logprobs_pos = logprobs_pos[s]
        ranks_pos = ranks_pos[s] if ranks_pos is not None else None
        decoded_pos = decoded_pos[s] if decoded_pos is not None else None

    return token_ids_pos, logprobs_pos, ranks_pos, decoded_pos


async def beam_search(
    self,
    prompt: PromptType,
    request_id: str,
    params: BeamSearchParams,
    lora_request: LoRARequest | None = None,
    trace_headers: Mapping[str, str] | None = None,
) -> AsyncGenerator[RequestOutput, None]:
    generation_time: float = 0.0
    num_generation_tokens: int = 0
    beam_width = params.beam_width
    max_tokens = params.max_tokens
    ignore_eos = params.ignore_eos
    temperature = params.temperature
    begin_token = params.begin_token
    end_token = params.end_token
    request_trace = RequestPerfTrace(
        request_id=request_id,
        module=__name__,
        fields={
            "beam_width": beam_width,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )

    # All beams from the same logical beam search will share the same priority value
    # (the prefill timestamp), so the scheduler will treat them as a group with equal priority
    MICROSECONDS = 1000000
    priority = int(time.perf_counter() * MICROSECONDS)  # us granularity
    rank = None
    if (
        hasattr(self.engine_client, "vllm_config")
        and self.engine_client.vllm_config.parallel_config is not None
    ):
        data_parallel_size = self.engine_client.vllm_config.parallel_config.data_parallel_size
        if data_parallel_size is not None and data_parallel_size > 1:
            # In DP mode, we assign ranks to beam search requests in a round-robin manner.
            rank = await _next_data_parallel_rank(data_parallel_size)
            logger.debug(
                f"rank for beam search: {rank} out of data_parallel_size: {data_parallel_size}"
            )

    include_stop_str_in_output = params.include_stop_str_in_output
    if beam_width == 0:
        raise VLLMValidationError(
            "Beam width must be greater than 0", parameter="beam_width", value=0
        )
    input_processor = self.input_processor
    tokenizer = input_processor.tokenizer
    if tokenizer is None:
        raise VLLMValidationError(
            "You cannot use beam search when `skip_tokenizer_init=True`",
            parameter="skip_tokenizer_init",
            value=True,
        )

    eos_token_id: int = tokenizer.eos_token_id
    sid_begin_token_id: int | None = None
    if begin_token is not None:
        sid_begin_token_id = tokenizer.convert_tokens_to_ids(begin_token)
        if sid_begin_token_id is None or sid_begin_token_id == -1:
            raise VLLMValidationError(
                "begin_token must be a valid token in the tokenizer vocabulary",
                parameter="begin_token",
                value=begin_token,
            )
    sid_end_token_id: int | None = None
    if end_token is not None:
        sid_end_token_id = tokenizer.convert_tokens_to_ids(end_token)
        if sid_end_token_id is None or sid_end_token_id == -1:
            raise VLLMValidationError(
                "end_token must be a valid token in the tokenizer vocabulary",
                parameter="end_token",
                value=end_token,
            )

    if is_explicit_encoder_decoder_prompt(prompt):
        raise NotImplementedError

    prompt_text = prompt if isinstance(prompt, str) else prompt.get("prompt")
    prompt_token_ids = [] if isinstance(prompt, str) else prompt.get("prompt_token_ids", [])
    multi_modal_data = None if isinstance(prompt, str) else prompt.get("multi_modal_data")
    mm_processor_kwargs = None

    tokenized_length = len(prompt_token_ids)
    logprobs_num = beam_width

    beam_search_params = SamplingParams(
        logprobs=logprobs_num,
        max_tokens=1,
        temperature=temperature,
        detokenize=False,
        flat_logprobs=True,
    )
    initial_tokens = list(prompt_token_ids)
    # NOTE: FlatLogprobs intentionally passed where BeamSearchSequence
    # expects list[dict[int, Logprob]].  Intermediate logprobs are never
    # read as dicts; the final beam.logprobs is replaced during
    # reconstruction at the end.
    # This reference is shared read-only across all beams — do not mutate
    # after the initial append below.
    initial_logprobs = FlatLogprobs()
    if sid_begin_token_id is not None:
        initial_tokens.append(sid_begin_token_id)
        initial_logprobs.append_fast([sid_begin_token_id], [0.0], [None], [None])

    all_beams = [
        BeamSearchSequence(
            tokens=initial_tokens,
            cum_logprob=0,
            logprobs=initial_logprobs,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            lora_request=lora_request,
        )
    ]
    # Deferred logprobs: instead of copying 6 FlatLogprobs lists per beam
    # per step (O(beams * steps^2)), store parent pointers and reconstruct
    # only for the final best beams at the end.
    # These dynamic attrs live on BeamSearchSequence (a plain @dataclass
    # without __slots__).  If upstream adds __slots__, these will break —
    # upstream the fields if that happens.
    all_beams[0]._lp_parent = None
    all_beams[0]._lp_step_data = None
    completed = []

    beam_search_start: int = time.perf_counter()
    prefill_start = beam_search_start  # Track initial start for prefill measurement
    pre_calc = 0 if sid_begin_token_id is None else 1
    if sid_end_token_id is not None:
        pre_calc += 1
    if max_tokens - pre_calc < 0:
        raise VLLMValidationError(
            f"max tokens should be not lower than {pre_calc} to accommodate reserved begin/end tokens.",
            parameter="max_tokens",
            value=max_tokens,
        )

    # Check once if the engine supports batch submission / beam fork.
    use_beam_request = hasattr(self.engine_client, "beam_request_step_update")

    prev_beam_internal_ids: list[str] = []
    fork_info: list[tuple[int, int]] | None = None
    prefill_id: str | None = None

    for token in range(max_tokens - pre_calc):
        if token == 1:
            beam_search_start = time.perf_counter()
        round_start = time.perf_counter() if token > 0 else None

        build_prompts_phase = (
            "prefill_build_prompts_batch" if token == 0 else "decode_build_prompts_batch"
        )
        with function_profile(
            "vllm_gr.entrypoints.openai.serving_engine",
            build_prompts_phase,
        ):
            with request_trace.phase(build_prompts_phase, step=token, beam_count=len(all_beams)):
                prompts_batch, lora_req_batch = zip(
                    *[
                        (
                            TokensPrompt(
                                prompt_token_ids=beam.tokens,
                                multi_modal_data=beam.multi_modal_data,
                                mm_processor_kwargs=beam.mm_processor_kwargs,
                            ),
                            beam.lora_request,
                        )
                        for beam in all_beams
                    ]
                )
        if token > 0:
            num_generation_tokens += len(all_beams)
        build_request_id_phase = (
            "prefill_build_request_id" if token == 0 else "decode_build_request_id"
        )
        with function_profile(
            "vllm_gr.entrypoints.openai.serving_engine",
            build_request_id_phase,
        ):
            with request_trace.phase(build_request_id_phase, step=token):
                request_id_batch = f"{request_id}-{random_uuid()}"

        # Launch catalog filtering in parallel with the engine step.
        catalog_task = None
        if self.models.catalog is not None:

            def get_valid_tokens_set(beam) -> set[int]:
                generated_tokens = beam.tokens[len(prompt_token_ids) :]
                return self.models.catalog.valid(generated_tokens)

            async def _run_catalog(coros):
                return list(await asyncio.gather(*coros))

            catalog_task = asyncio.create_task(
                _run_catalog([asyncio.to_thread(get_valid_tokens_set, beam) for beam in all_beams])
            )

        gen_start = time.perf_counter()

        # Execute Engine Step Pass
        if use_beam_request and fork_info is not None:
            with request_trace.phase("beam_request_step", step=token, mode="beam_request"):
                output, prev_beam_internal_ids = await _beam_request_step(
                    self.engine_client,
                    session_id=prefill_id,
                    fork_info=fork_info,
                    prev_beam_internal_ids=prev_beam_internal_ids,
                    all_beams=all_beams,
                    request_id_batch=request_id_batch,
                    beam_search_params=beam_search_params,
                    eos_token_id=eos_token_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    prefix_len=len(initial_tokens),
                    priority=priority,
                    data_parallel_rank=rank,
                    request_trace=request_trace,
                )
        else:
            with request_trace.phase("add_batch_step", step=token, mode="add_batch"):
                output, new_ids = await _add_batch_step(
                    self.engine_client,
                    prompts_batch,
                    lora_req_batch,
                    request_id_batch,
                    beam_search_params,
                    use_beam_request,
                    trace_headers,
                    priority=priority,
                    data_parallel_rank=rank,
                    request_trace=request_trace,
                )
            if new_ids:
                prev_beam_internal_ids = new_ids
                if prefill_id is None:
                    prefill_id = new_ids[0]

        valid_tokens_sets = None
        if catalog_task is not None:
            with request_trace.phase("catalog_wait", step=token):
                valid_tokens_sets = await catalog_task
        if token > 0:
            generation_time += time.perf_counter() - gen_start
        new_beams = []
        # Store all new tokens generated by beam
        all_beams_token_id = []
        # Store the cumulative probability of all tokens
        # generated by beam search
        all_beams_logprob = []
        # Per-beam flat data cache: (token_ids, logprobs, ranks, decoded_tokens)
        # Used to build logprobs_entry dicts without calling FlatLogprobs.__getitem__
        beam_flat_cache: list[tuple | None] = []

        # -------------------------------------------------------------------------
        # STRIDE-AWARE METADATA EXTRACTION FOR GENERATED BEAMS
        # -------------------------------------------------------------------------
        if use_beam_request and fork_info is not None:
            # Under the optimized mode, 'output' contains exactly 1 combined RequestOutput instance
            beam_result = output[0]
            if beam_result.outputs[0].finish_reason == "error":
                with request_trace.phase("cleanup", kind="beam_error", step=token):
                    await _beam_request_cleanup(
                        self.engine_client, prefill_id, prev_beam_internal_ids, rank
                    )
                request_trace.finish(status="error", completed=False)
                yield RequestOutput(
                    request_id=request_id,
                    prompt=prompt_text,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text="",
                            token_ids=[],
                            cumulative_logprob=None,
                            logprobs=None,
                            finish_reason="error",
                        )
                    ],
                    finished=True,
                    prompt_token_ids=prompt_token_ids,
                    prompt_logprobs=None,
                )
                return

            flat = beam_result.outputs[0].logprobs
            if flat is not None:
                # Extract tracking attributes cleanly
                raw_token_ids = flat.token_ids
                raw_logprobs = flat.logprobs
                raw_ranks = getattr(flat, "ranks", None)
                raw_decoded = getattr(flat, "decoded_tokens", None)
                total_logprobs = len(raw_token_ids)

                stride_k = total_logprobs // logprobs_num
                beam_segments: list[
                    tuple[BeamSearchSequence, list[int], list[float], list[int] | None, list[str] | None]
                ] = []

                with function_profile(
                    "vllm_gr.entrypoints.openai.serving_engine",
                    "decode_extract_beam_segments",
                ):
                    with request_trace.phase("decode_extract_beam_segments", step=token):
                        for b_idx in range(len(fork_info)):
                            current_beam = all_beams[b_idx]

                            # Compute the true exact allocation footprint boundaries
                            # from the flat remuxed vector block for this specific beam.
                            start_offset = b_idx * stride_k
                            end_offset = start_offset + stride_k

                            token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = (
                                _extract_and_dedup_segment(
                                    raw_token_ids,
                                    raw_logprobs,
                                    raw_ranks,
                                    raw_decoded,
                                    start_offset,
                                    end_offset,
                                    logprobs_num,
                                )
                            )
                            beam_segments.append(
                                (
                                    current_beam,
                                    token_ids_pos,
                                    logprobs_pos,
                                    ranks_pos,
                                    decoded_pos,
                                )
                            )

                if valid_tokens_sets is not None:
                    with function_profile(
                        "vllm_gr.entrypoints.openai.serving_engine",
                        "decode_filter_valid_tokens",
                    ):
                        with request_trace.phase("decode_filter_valid_tokens", step=token):
                            filtered_segments = []
                            for b_idx, (
                                current_beam,
                                token_ids_pos,
                                logprobs_pos,
                                ranks_pos,
                                decoded_pos,
                            ) in enumerate(beam_segments):
                                valid_tokens_set = valid_tokens_sets[b_idx]
                                filtered_logprobs_pos = [
                                    lp if tid in valid_tokens_set else -float("inf")
                                    for tid, lp in zip(token_ids_pos, logprobs_pos)
                                ]
                                filtered_segments.append(
                                    (
                                        current_beam,
                                        token_ids_pos,
                                        filtered_logprobs_pos,
                                        ranks_pos,
                                        decoded_pos,
                                    )
                                )
                            beam_segments = filtered_segments

                with function_profile(
                    "vllm_gr.entrypoints.openai.serving_engine",
                    "decode_flatten_candidates",
                ):
                    with request_trace.phase("decode_flatten_candidates", step=token):
                        for (
                            current_beam,
                            token_ids_pos,
                            logprobs_pos,
                            ranks_pos,
                            decoded_pos,
                        ) in beam_segments:
                            # Save the clean segment safely into the historical timeline cache.
                            beam_flat_cache.append(
                                (token_ids_pos, logprobs_pos, ranks_pos, decoded_pos)
                            )
                            all_beams_token_id.extend(token_ids_pos)
                            all_beams_logprob.extend(
                                current_beam.cum_logprob + lp for lp in logprobs_pos
                            )
            else:
                beam_flat_cache = [None] * len(fork_info)
        else:
            # Baseline execution path for prefill / batch loops
            for i, result in enumerate(output):
                current_beam = all_beams[i]
                if result.outputs[0].finish_reason == "error":
                    request_trace.finish(status="error", completed=False)
                    yield RequestOutput(
                        request_id=request_id,
                        prompt=prompt_text,
                        outputs=[
                            CompletionOutput(
                                index=0,
                                text="",
                                token_ids=[],
                                cumulative_logprob=None,
                                logprobs=None,
                                finish_reason="error",
                            )
                        ],
                        finished=True,
                        prompt_token_ids=prompt_token_ids,
                        prompt_logprobs=None,
                    )
                    return

                if result.outputs[0].logprobs is not None:
                    flat = result.outputs[0].logprobs

                    token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = (
                        extract_and_dedup_flat_logprobs(flat, logprobs_num)
                    )
                    if valid_tokens_sets is not None:
                        valid_tokens_set = valid_tokens_sets[i]
                        logprobs_pos = [
                            lp if tid in valid_tokens_set else -float("inf")
                            for tid, lp in zip(token_ids_pos, logprobs_pos)
                        ]
                    beam_flat_cache.append((token_ids_pos, logprobs_pos, ranks_pos, decoded_pos))
                    all_beams_token_id.extend(token_ids_pos)
                    all_beams_logprob.extend(current_beam.cum_logprob + lp for lp in logprobs_pos)
                else:
                    beam_flat_cache.append(None)

        with request_trace.phase("merge_candidate_scores", step=token):
            # Handle the token for the end of sentence (EOS)
            all_beams_token_id = np.array(all_beams_token_id)
            all_beams_logprob = np.array(all_beams_logprob)

            if not ignore_eos:
                # Get the index position of eos token in all generated results
                eos_idx = np.where(all_beams_token_id == eos_token_id)[0]
                for idx in eos_idx:
                    current_beam = all_beams[idx // logprobs_num]
                    cached = beam_flat_cache[idx // logprobs_num]
                    assert cached is not None
                    eos_beam = BeamSearchSequence(
                        tokens=current_beam.tokens + [eos_token_id]
                        if include_stop_str_in_output
                        else current_beam.tokens,
                        logprobs=initial_logprobs,
                        cum_logprob=float(all_beams_logprob[idx]),
                        finish_reason="stop",
                        stop_reason=eos_token_id,
                    )
                    eos_beam._lp_parent = current_beam
                    eos_beam._lp_step_data = cached
                    completed.append(eos_beam)
                # After processing, set the log probability of the eos condition
                # to negative infinity.
                all_beams_logprob[eos_idx] = -np.inf

            # Processing non-EOS tokens
            # Get indices of the top beam_width probabilities
            if all_beams_logprob.size > beam_width:
                topn_idx = np.argpartition(np.negative(all_beams_logprob), beam_width)[:beam_width]
            else:
                topn_idx = np.arange(all_beams_logprob.size)

        with request_trace.phase("materialize_next_beams", step=token):
            for idx in topn_idx:
                current_beam = all_beams[idx // logprobs_num]
                token_id = int(all_beams_token_id[idx])
                cached = beam_flat_cache[idx // logprobs_num]
                assert cached is not None
                new_beam = BeamSearchSequence(
                    tokens=current_beam.tokens + [token_id],
                    logprobs=initial_logprobs,
                    lora_request=current_beam.lora_request,
                    cum_logprob=float(all_beams_logprob[idx]),
                    multi_modal_data=current_beam.multi_modal_data,
                    mm_processor_kwargs=current_beam.mm_processor_kwargs,
                )
                new_beam._lp_parent = current_beam
                new_beam._lp_step_data = cached
                new_beams.append(new_beam)

            # Build fork_info for next iteration's BEAM_FORK.
            if use_beam_request:
                fork_info = [(idx // logprobs_num, int(all_beams_token_id[idx])) for idx in topn_idx]

        all_beams = new_beams

        if round_start is not None:
            round_elapsed = time.perf_counter() - round_start
            request_trace.phases.append(("decode_round_total", round_elapsed))
            log_perf_event(
                request_trace.module,
                "decode_round_total",
                round_elapsed,
                request_id=request_trace.request_id,
                phase="decode_round_total",
                event="request_phase",
                step=token,
                beam_count=len(all_beams),
            )

    if use_beam_request and prev_beam_internal_ids:
        with request_trace.phase("cleanup", kind="beam_finalize"):
            await _beam_request_cleanup(
                self.engine_client, prefill_id, prev_beam_internal_ids, rank
            )

    if sid_end_token_id is not None:
        for beam in all_beams:
            beam.tokens.append(sid_end_token_id)
    completed.extend(all_beams)

    # Measure beam sort time
    with request_trace.phase("sort_completed"):
        beam_search_sort_start = time.perf_counter()
        sorted_completed = sorted(completed, key=lambda x: x.cum_logprob, reverse=True)
        beam_search_sort_time = time.perf_counter() - beam_search_sort_start
        best_beams = sorted_completed[:beam_width]

    # Reconstruct full logprobs only for the final best beams by walking
    # the parent-pointer tree.  This replaces the O(beams * steps^2)
    # per-step copying with a single O(beams * steps) pass at the end.
    with request_trace.phase("reconstruct_logprobs"):
        for beam in best_beams:
            reconstruct_beam_logprobs(beam, initial_logprobs, sid_end_token_id)

    with request_trace.phase("decode_best_beams"):
        for i, beam in enumerate(best_beams):
            tokens = (
                beam.tokens[tokenized_length:-1]
                if beam.tokens[-1] == eos_token_id and not ignore_eos
                else beam.tokens[tokenized_length:]
            )
            beam.text = tokenizer.decode(tokens)

    beam_search_decode_time = time.perf_counter() - beam_search_start
    beam_search_prefill_time = beam_search_start - prefill_start
    beam_search_overhead = beam_search_decode_time - generation_time

    metrics = RequestStateStats(
        beam_search_overhead=beam_search_overhead,
        beam_search_prefill_time=beam_search_prefill_time,
        beam_search_decode_time=beam_search_decode_time,
        beam_search_sort_time=beam_search_sort_time,
        num_generation_tokens=num_generation_tokens,
        num_beams=beam_width,
    )

    out = RequestOutput(
        request_id=request_id,
        prompt=prompt_text,
        outputs=[
            CompletionOutput(
                text=beam.text,
                cumulative_logprob=beam.cum_logprob,
                token_ids=beam.tokens[tokenized_length:],
                index=i,
                logprobs=beam.logprobs,
                finish_reason=beam.finish_reason if beam.finish_reason is not None else "length",
                stop_reason=beam.stop_reason,
            )
            for i, beam in enumerate(best_beams)
        ],
        finished=True,
        prompt_token_ids=prompt_token_ids,
        prompt_logprobs=None,
        metrics=metrics,
    )

    # Record beam search metrics for Prometheus
    # Get the model name - it could be a property or method
    try:
        model_name = self.models.model_name
        if callable(model_name):
            model_name = model_name()
        model_name = str(model_name)
    except Exception:
        model_name = "unknown"
    record_beam_search_metrics(metrics, engine="0", model_name=model_name)

    request_trace.finish(
        completed=True,
        model_name=model_name,
        total_decode_ms=round(beam_search_decode_time * 1000.0, 3),
        prefill_ms=round(beam_search_prefill_time * 1000.0, 3),
        sort_ms=round(beam_search_sort_time * 1000.0, 3),
        generation_ms=round(generation_time * 1000.0, 3),
    )

    yield out
