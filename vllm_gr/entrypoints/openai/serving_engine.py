# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import time
from collections.abc import AsyncGenerator, Mapping
from typing import Any

import numpy as np
from vllm.beam_search import BeamSearchSequence
from vllm.entrypoints.openai.protocol import VLLMValidationError
from vllm.inputs.data import PromptType, TokensPrompt
from vllm.inputs.parse import is_explicit_encoder_decoder_prompt
from vllm.logger import init_logger
from vllm.logprobs import FlatLogprobs
from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalDataDict
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.utils import random_uuid

from vllm_gr.logprobs import extract_and_dedup_flat_logprobs, reconstruct_beam_logprobs
from vllm_gr.v1.engine.types import BeamForkRequest
from vllm_gr.v1.metrics.stats import RequestStateStats

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


async def _collect_beam_result(q):
    """Drain output queue until finished."""
    while True:
        out = await q.get()
        if out.finished:
            return out


async def _gather_beam_results(queues):
    """Gather results from multiple output queues."""
    tasks = [asyncio.create_task(_collect_beam_result(q)) for q in queues]
    return list(await asyncio.gather(*tasks))


def _update_beam_engine_metrics(
    aggregate: dict[str, float],
    outputs: list[RequestOutput],
) -> None:
    for output in outputs:
        metrics = getattr(output, "metrics", None)
        if metrics is None:
            continue

        arrival_time = float(getattr(metrics, "arrival_time", 0.0) or 0.0)
        queued_ts = float(getattr(metrics, "queued_ts", 0.0) or 0.0)
        scheduled_ts = float(getattr(metrics, "scheduled_ts", 0.0) or 0.0)
        first_token_ts = float(getattr(metrics, "first_token_ts", 0.0) or 0.0)
        last_token_ts = float(getattr(metrics, "last_token_ts", 0.0) or 0.0)
        first_token_latency = float(
            getattr(metrics, "first_token_latency", 0.0) or 0.0
        )

        if arrival_time > 0.0:
            current = aggregate.get("arrival_time", 0.0)
            aggregate["arrival_time"] = arrival_time if current == 0.0 else min(current, arrival_time)
        if queued_ts > 0.0:
            current = aggregate.get("queued_ts", 0.0)
            aggregate["queued_ts"] = queued_ts if current == 0.0 else min(current, queued_ts)
        if scheduled_ts > 0.0:
            current = aggregate.get("scheduled_ts", 0.0)
            aggregate["scheduled_ts"] = (
                scheduled_ts if current == 0.0 else min(current, scheduled_ts)
            )
        if first_token_ts > 0.0:
            current = aggregate.get("first_token_ts", 0.0)
            aggregate["first_token_ts"] = (
                first_token_ts if current == 0.0 else min(current, first_token_ts)
            )
        if last_token_ts > 0.0:
            aggregate["last_token_ts"] = max(aggregate.get("last_token_ts", 0.0), last_token_ts)
        if first_token_latency > 0.0:
            current = aggregate.get("first_token_latency", 0.0)
            aggregate["first_token_latency"] = (
                first_token_latency
                if current == 0.0
                else min(current, first_token_latency)
            )


def _finalize_beam_engine_metrics(
    metrics: RequestStateStats,
    aggregate: dict[str, float],
    request_wall_start_time: float,
) -> None:
    metrics.arrival_time = aggregate.get("arrival_time", request_wall_start_time)
    metrics.queued_ts = aggregate.get("queued_ts", 0.0)
    metrics.scheduled_ts = aggregate.get("scheduled_ts", 0.0)
    metrics.first_token_ts = aggregate.get("first_token_ts", 0.0)
    metrics.last_token_ts = aggregate.get("last_token_ts", 0.0)
    metrics.first_token_latency = aggregate.get("first_token_latency", 0.0)

    if metrics.queued_ts > 0.0 and metrics.scheduled_ts >= metrics.queued_ts:
        metrics.beam_engine_queue_ms = (
            metrics.scheduled_ts - metrics.queued_ts
        ) * 1000.0
    if metrics.scheduled_ts > 0.0 and metrics.first_token_ts >= metrics.scheduled_ts:
        metrics.beam_engine_prefill_ms = (
            metrics.first_token_ts - metrics.scheduled_ts
        ) * 1000.0
    if metrics.first_token_ts > 0.0 and metrics.last_token_ts >= metrics.first_token_ts:
        metrics.beam_engine_decode_ms = (
            metrics.last_token_ts - metrics.first_token_ts
        ) * 1000.0
    if metrics.scheduled_ts > 0.0 and metrics.last_token_ts >= metrics.scheduled_ts:
        metrics.beam_engine_inference_ms = (
            metrics.last_token_ts - metrics.scheduled_ts
        ) * 1000.0


def _current_platform_device_type() -> str | None:
    try:
        from vllm.platforms import current_platform
    except Exception:
        return None
    device_type = getattr(current_platform, "device_type", None)
    return None if device_type is None else str(device_type)


def _is_npu_piecewise_runtime(engine_client) -> bool:
    vllm_config = getattr(engine_client, "vllm_config", None)
    compilation_config = getattr(vllm_config, "compilation_config", None)
    cudagraph_mode = str(getattr(compilation_config, "cudagraph_mode", "")).upper()
    if cudagraph_mode != "PIECEWISE":
        return False

    device_config = getattr(vllm_config, "device_config", None)
    device_type = getattr(device_config, "device_type", None)
    if device_type is None:
        device_type = _current_platform_device_type()
    return str(device_type).lower() == "npu"


def _get_beam_step_priority(engine_client, base_priority: int, step_index: int) -> int:
    """Separate long prefill and short decode beam steps on NPU PIECEWISE mode.

    vLLM priority scheduling treats smaller integers as higher priority.  On
    NPU PIECEWISE mode, we want decode-heavy beam waves to outrank newly
    arriving long-prefill waves globally, not just by a tiny delta, so we
    encode the phase in the high-order portion of the integer.
    """
    if step_index <= 0 or not _is_npu_piecewise_runtime(engine_client):
        return base_priority

    phase_window = 1 << 60
    decode_timestamp = min(base_priority, phase_window - 1)
    return decode_timestamp


def _get_initial_beam_priority(engine_client, base_priority: int) -> int:
    if not _is_npu_piecewise_runtime(engine_client):
        return base_priority

    phase_window = 1 << 60
    prefill_timestamp = min(base_priority, phase_window - 1)
    return phase_window + prefill_timestamp


async def _beam_fork_step(
    engine_client,
    fork_info,
    prev_beam_internal_ids,
    all_beams,
    request_id_batch,
    beam_search_params,
    eos_token_id,
    lora_request,
    trace_headers,
    priority=0,
    data_parallel_rank: int | None = None,
):
    """BEAM_FORK path: fork parent beams into children (steps 1+).

    Returns (output_list, new_internal_ids).
    """
    parent_ids = []
    child_ids = []
    child_token_ids = []
    queues = []

    for new_idx, (parent_beam_idx, tok) in enumerate(fork_info):
        parent_id = prev_beam_internal_ids[parent_beam_idx]
        child_id = f"{request_id_batch}-beam-{new_idx}"
        parent_ids.append(parent_id)
        child_ids.append(child_id)
        child_token_ids.append(tok)

        q = engine_client.register_beam_output(
            child_id,
            all_beams[parent_beam_idx].tokens,
            beam_search_params,
            eos_token_id=eos_token_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
        )
        queues.append(q)

    used = set(parent_ids)
    abort_ids = [pid for pid in prev_beam_internal_ids if pid not in used]

    await engine_client.beam_fork(
        BeamForkRequest(
            parent_ids=parent_ids,
            child_ids=child_ids,
            token_ids=child_token_ids,
            abort_ids=abort_ids,
            sampling_params=beam_search_params,
            eos_token_id=eos_token_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            data_parallel_rank=data_parallel_rank,
            priority=priority,
        )
    )

    output = await _gather_beam_results(queues)
    return output, child_ids


async def _add_batch_step(
    engine_client,
    prompts_batch,
    lora_req_batch,
    request_id_batch,
    beam_search_params,
    use_beam_fork,
    trace_headers,
    priority=0,
    data_parallel_rank: int | None = None,
):
    """ADD_BATCH path: prepare and batch-send requests (step 0).

    Returns (output_list, new_internal_ids).
    """
    queues = []
    engine_core_requests = []
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

    await engine_client._add_requests_batch(engine_core_requests, use_batch_message=use_beam_fork)

    internal_ids = [r.request_id for r in engine_core_requests] if use_beam_fork else []
    output = await _gather_beam_results(queues)
    return output, internal_ids


async def beam_search(
    self,
    prompt: PromptType,
    request_id: str,
    params: BeamSearchParams,
    lora_request: LoRARequest | None = None,
    trace_headers: Mapping[str, str] | None = None,
) -> AsyncGenerator[RequestOutput, None]:
    request_wall_start_time = time.time()
    generation_time: float = 0.0
    num_generation_tokens: int = 0
    beam_engine_metrics_aggregate: dict[str, float] = {}
    stage_breakdown_s: dict[str, float] = {
        "beam_first_step_total_s": 0.0,
        "beam_first_step_engine_s": 0.0,
        "beam_first_step_catalog_wait_s": 0.0,
        "beam_first_step_logprob_process_s": 0.0,
        "beam_first_step_topn_select_s": 0.0,
        "beam_later_steps_total_s": 0.0,
        "beam_later_steps_engine_s": 0.0,
        "beam_later_steps_catalog_wait_s": 0.0,
        "beam_later_steps_logprob_process_s": 0.0,
        "beam_later_steps_topn_select_s": 0.0,
        "beam_catalog_launch_s": 0.0,
        "beam_catalog_wait_s": 0.0,
        "beam_result_postprocess_s": 0.0,
        "beam_reconstruct_s": 0.0,
        "beam_decode_text_s": 0.0,
        "beam_cleanup_s": 0.0,
    }
    beam_width = params.beam_width
    max_tokens = params.max_tokens
    ignore_eos = params.ignore_eos
    temperature = params.temperature
    begin_token = params.begin_token
    end_token = params.end_token

    # All beams from the same logical beam search will share the same priority value
    # (the prefill timestamp), so the scheduler will treat them as a group with equal priority
    MICROSECONDS = 1000000
    priority = _get_initial_beam_priority(
        self.engine_client,
        int(time.perf_counter() * MICROSECONDS),
    )
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

    eos_token_id: int = tokenizer.eos_token_id  # type: ignore
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

    prompt_text: str | None
    prompt_token_ids: list[int]
    multi_modal_data: MultiModalDataDict | None
    if isinstance(prompt, str):
        prompt_text = prompt
        prompt_token_ids = []
        multi_modal_data = None
    else:
        prompt_text = prompt.get("prompt")  # type: ignore
        prompt_token_ids = prompt.get("prompt_token_ids", [])  # type: ignore
        multi_modal_data = prompt.get("multi_modal_data")  # type: ignore

    mm_processor_kwargs: dict[str, Any] | None = None

    # This is a workaround to fix multimodal beam search; this is a
    # bandaid fix for 2 small problems:
    # 1. Multi_modal_data on the processed_inputs currently resolves to
    #    `None`.
    # 2. preprocessing above expands the multimodal placeholders. However,
    #    this happens again in generation, so the double expansion causes
    #    a mismatch.
    # TODO - would be ideal to handle this more gracefully.

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
    pre_calc = 0 if sid_begin_token_id is None else 1
    if sid_end_token_id is not None:
        pre_calc += 1
    if max_tokens - pre_calc < 0:
        raise VLLMValidationError(
            f"max tokens should be not lower than {pre_calc} to accomedate reserved begin/end tokens.",
            parameter="max_tokens",
            value=max_tokens,
        )

    # Check once if the engine supports batch submission / beam fork.
    use_batch = hasattr(self.engine_client, "prepare_request")
    use_beam_fork = hasattr(self.engine_client, "beam_fork")

    if not use_batch:
        raise VLLMValidationError(
            "use_batch must be enabled",
            parameter="use_batch",
            value=use_batch,
        )

    # BEAM_FORK tracking: internal IDs from previous step + fork info
    prev_beam_internal_ids: list[str] = []
    # fork_info: list of (parent_beam_idx, token_id) for BEAM_FORK
    fork_info: list[tuple[int, int]] | None = None

    for token in range(max_tokens - pre_calc):
        if token == 1:
            beam_search_start = time.perf_counter()
        step_total_start = time.perf_counter()
        step_priority = _get_beam_step_priority(self.engine_client, priority, token)

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
        request_id_batch = f"{request_id}-{random_uuid()}"

        # Launch catalog filtering in parallel with the engine step.
        catalog_task = None
        if self.models.catalog is not None:
            catalog_launch_start = time.perf_counter()

            def get_valid_tokens_set(beam) -> set[int]:
                generated_tokens = beam.tokens[len(prompt_token_ids) :]
                return self.models.catalog.valid(generated_tokens)

            async def _run_catalog(coros):
                return list(await asyncio.gather(*coros))

            catalog_task = asyncio.create_task(
                _run_catalog([asyncio.to_thread(get_valid_tokens_set, beam) for beam in all_beams])
            )
            stage_breakdown_s["beam_catalog_launch_s"] += (
                time.perf_counter() - catalog_launch_start
            )

        engine_start = time.perf_counter()

        if use_beam_fork and fork_info is not None:
            output, prev_beam_internal_ids = await _beam_fork_step(
                self.engine_client,
                fork_info,
                prev_beam_internal_ids,
                all_beams,
                request_id_batch,
                beam_search_params,
                eos_token_id,
                lora_request,
                trace_headers,
                priority=step_priority,
                data_parallel_rank=rank,
            )
        elif use_batch:
            output, new_ids = await _add_batch_step(
                self.engine_client,
                prompts_batch,
                lora_req_batch,
                request_id_batch,
                beam_search_params,
                use_beam_fork,
                trace_headers,
                priority=step_priority,
                data_parallel_rank=rank,
            )
            if new_ids:
                prev_beam_internal_ids = new_ids
        else:
            raise VLLMValidationError(
                "internal error. use_batch must be enabled",
                parameter="use_batch",
                value=use_batch,
            )
        engine_elapsed = time.perf_counter() - engine_start
        _update_beam_engine_metrics(beam_engine_metrics_aggregate, output)

        valid_tokens_sets = None
        if catalog_task is not None:
            catalog_wait_start = time.perf_counter()
            valid_tokens_sets = await catalog_task
            catalog_wait_elapsed = time.perf_counter() - catalog_wait_start
            stage_breakdown_s["beam_catalog_wait_s"] += catalog_wait_elapsed
        else:
            catalog_wait_elapsed = 0.0
        if token > 0:
            generation_time += engine_elapsed
        new_beams = []
        # Store all new tokens generated by beam
        all_beams_token_id = []
        # Store the cumulative probability of all tokens
        # generated by beam search
        all_beams_logprob = []
        # Per-beam flat data cache: (token_ids, logprobs, ranks, decoded_tokens)
        # Used to build logprobs_entry dicts without calling FlatLogprobs.__getitem__
        beam_flat_cache: list[tuple | None] = []
        logprob_process_start = time.perf_counter()
        # Iterate through all beam inference results
        for i, result in enumerate(output):
            current_beam = all_beams[i]

            # check for error finish reason and abort beam search
            if result.outputs[0].finish_reason == "error":
                # Clean up beam cache before returning to avoid leak.
                if use_beam_fork and prev_beam_internal_ids:
                    await self.engine_client.beam_fork(
                        BeamForkRequest(
                            parent_ids=[],
                            child_ids=[],
                            token_ids=[],
                            abort_ids=prev_beam_internal_ids,
                            sampling_params=beam_search_params,
                            data_parallel_rank=rank,
                        )
                    )
                # yield error output and terminate beam search
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
                if self.models.catalog is not None and catalog_task:
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
        logprob_process_elapsed = time.perf_counter() - logprob_process_start

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
        topn_select_start = time.perf_counter()
        if all_beams_logprob.size > beam_width:
            topn_idx = np.argpartition(np.negative(all_beams_logprob), beam_width)[:beam_width]
        else:
            topn_idx = np.arange(all_beams_logprob.size)
        topn_select_elapsed = time.perf_counter() - topn_select_start

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
        if use_beam_fork:
            fork_info = [(idx // logprobs_num, int(all_beams_token_id[idx])) for idx in topn_idx]

        all_beams = new_beams
        step_total_elapsed = time.perf_counter() - step_total_start
        step_prefix = "beam_first_step" if token == 0 else "beam_later_steps"
        stage_breakdown_s[f"{step_prefix}_total_s"] += step_total_elapsed
        stage_breakdown_s[f"{step_prefix}_engine_s"] += engine_elapsed
        stage_breakdown_s[f"{step_prefix}_catalog_wait_s"] += catalog_wait_elapsed
        stage_breakdown_s[f"{step_prefix}_logprob_process_s"] += logprob_process_elapsed
        stage_breakdown_s[f"{step_prefix}_topn_select_s"] += topn_select_elapsed
        stage_breakdown_s["beam_result_postprocess_s"] += max(
            0.0,
            step_total_elapsed
            - engine_elapsed
            - catalog_wait_elapsed
            - topn_select_elapsed,
        )

    # Cleanup: remove remaining beam cache entries.
    if use_beam_fork and prev_beam_internal_ids:
        cleanup_start = time.perf_counter()
        await self.engine_client.beam_fork(
            BeamForkRequest(
                parent_ids=[],
                child_ids=[],
                token_ids=[],
                abort_ids=prev_beam_internal_ids,
                sampling_params=beam_search_params,
                data_parallel_rank=rank,
            )
        )
        stage_breakdown_s["beam_cleanup_s"] += time.perf_counter() - cleanup_start

    if sid_end_token_id is not None:
        for beam in all_beams:
            beam.tokens.append(sid_end_token_id)
    completed.extend(all_beams)
    sorted_completed = sorted(completed, key=lambda x: x.cum_logprob, reverse=True)
    best_beams = sorted_completed[:beam_width]

    # Reconstruct full logprobs only for the final best beams by walking
    # the parent-pointer tree.  This replaces the O(beams * steps^2)
    # per-step copying with a single O(beams * steps) pass at the end.
    reconstruct_start = time.perf_counter()
    for beam in best_beams:
        reconstruct_beam_logprobs(beam, initial_logprobs, sid_end_token_id)
    stage_breakdown_s["beam_reconstruct_s"] += time.perf_counter() - reconstruct_start

    decode_text_start = time.perf_counter()
    for beam in best_beams:
        if beam.tokens[-1] == eos_token_id and not ignore_eos:
            # Skip the eos token in the text.
            tokens = beam.tokens[tokenized_length:-1]
        else:
            tokens = beam.tokens[tokenized_length:]
        beam.text = tokenizer.decode(tokens)
    stage_breakdown_s["beam_decode_text_s"] += time.perf_counter() - decode_text_start

    beam_search_decode_time = time.perf_counter() - beam_search_start
    beam_search_overhead = beam_search_decode_time - generation_time
    metrics = RequestStateStats(
        beam_search_overhead=beam_search_overhead,
        beam_search_decode_time=beam_search_decode_time,
        num_generation_tokens=num_generation_tokens,
        stage_breakdown_ms={
            key.removesuffix("_s") + "_ms": value * 1000.0
            for key, value in stage_breakdown_s.items()
        },
    )
    _finalize_beam_engine_metrics(
        metrics, beam_engine_metrics_aggregate, request_wall_start_time
    )
    metrics.beam_engine_e2e_ms = (time.time() - request_wall_start_time) * 1000.0

    out = RequestOutput(
        request_id=request_id,
        prompt=prompt_text,
        outputs=[
            CompletionOutput(
                text=beam.text,  # type: ignore
                cumulative_logprob=beam.cum_logprob,
                token_ids=beam.tokens[tokenized_length:],
                index=i,
                logprobs=beam.logprobs,
                finish_reason=beam.finish_reason if beam.finish_reason is not None else "length",
                stop_reason=beam.stop_reason,
            )
            for (i, beam) in enumerate(best_beams)
        ],
        finished=True,
        prompt_token_ids=prompt_token_ids,
        prompt_logprobs=None,
        metrics=metrics,
    )
    yield out
