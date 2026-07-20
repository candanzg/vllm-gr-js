# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Parse structured performance JSON logs into a CSV summary.

The parser expects the JSON lines emitted by ``vllm_gr.utils.perf`` and
produces a CSV similar to ``performance_examle.csv``:

    Evidence group,Segment / event,Start us,End us,p50 us,What happens

It supports two modes:

* aggregate mode: summarize all parsed requests together
* request mode: isolate one request_id and reconstruct a single timeline

Usage examples:

    python -m vllm_gr.tools.parse_perf_logs --input server.log --output perf.csv
    python -m vllm_gr.tools.parse_perf_logs --input server.log --request-id abc123
    python -m vllm_gr.tools.parse_perf_logs --input server.log --stage-summary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DESCRIPTION = "request timing event"
QUEUEING_PHASES = {
    "prepare_request_batch",
    "add_requests_batch",
    "build_fork_info",
    "register_beam_output",
    "beam_request_step_update",
}
COLLECT_MERGE_PHASES = {
    "gather_beam_results",
    "collect_beam_result",
    "gather_result_stats",
    "collect_result_stats",
    "merge_candidate_scores",
    "materialize_next_beams",
    "catalog_wait",
    "cleanup",
    "sort_completed",
    "reconstruct_logprobs",
    "decode_best_beams",
}
COLLECT_MERGE_FUNCTIONS = {
    ("vllm_gr.entrypoints.openai.serving_engine", "_collect_beam_result"),
    ("vllm_gr.entrypoints.openai.serving_engine", "_gather_beam_results"),
    ("vllm_gr.entrypoints.openai.serving_engine", "create_collect_tasks"),
    ("vllm_gr.entrypoints.openai.serving_engine", "await_collect_tasks"),
    ("vllm_gr.entrypoints.openai.serving_engine", "extract_beam_logprobs_segment"),
    ("vllm_gr.entrypoints.openai.serving_engine", "filter_beam_logprobs_by_valid_tokens"),
    ("vllm_gr.entrypoints.openai.serving_engine", "extract_flat_logprobs_segment"),
    ("vllm_gr.entrypoints.openai.serving_engine", "filter_flat_logprobs_by_valid_tokens"),
    ("vllm.v1.worker.gpu.async_utils", "get_output"),
}


def _percentile(samples: list[float], percentile: float) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return samples[0]
    ordered = sorted(samples)
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class PerfEvent:
    event: str
    module: str
    func: str
    duration_ms: float
    request_id: str | None
    phase: str | None
    fields: dict[str, Any]


def _json_candidates(line: str) -> Iterable[str]:
    line = line.strip()
    if not line:
        return []
    if line.startswith("{") and line.endswith("}"):
        return [line]
    # Greedy JSON extraction from log lines such as:
    # INFO ... {"event": "...", ...}
    start = line.find("{")
    if start < 0:
        return []
    candidate = line[start:]
    if candidate.endswith("}"):
        return [candidate]
    return []


def _load_event_from_obj(obj: dict[str, Any]) -> PerfEvent | None:
    if not isinstance(obj, dict):
        return None
    event = obj.get("event")
    module = obj.get("module")
    func = obj.get("func") or obj.get("phase")
    duration_ms = obj.get("duration_ms")
    if event is None or module is None or func is None or duration_ms is None:
        return None
    try:
        duration = float(duration_ms)
    except (TypeError, ValueError):
        return None
    request_id = obj.get("request_id")
    phase = obj.get("phase")
    fields = {
        key: value
        for key, value in obj.items()
        if key not in {"event", "module", "func", "duration_ms", "request_id", "phase"}
    }
    return PerfEvent(
        event=str(event),
        module=str(module),
        func=str(func),
        duration_ms=duration,
        request_id=str(request_id) if request_id is not None else None,
        phase=str(phase) if phase is not None else None,
        fields=fields,
    )


def load_events(path: Path | None) -> list[PerfEvent]:
    raw_lines = sys.stdin if path is None else path.open("r", encoding="utf-8")
    close_after = path is not None
    events: list[PerfEvent] = []
    try:
        for line in raw_lines:
            for candidate in _json_candidates(line):
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                event = _load_event_from_obj(obj)
                if event is not None:
                    events.append(event)
    finally:
        if close_after:
            raw_lines.close()
    return events


def _group_for_event(event: PerfEvent) -> str:
    module = event.module
    if module.startswith("vllm_gr.entrypoints.openai"):
        return "API / client"
    if module.startswith("vllm_gr.entrypoints"):
        return "API / client"
    if module.startswith("vllm.v1.core.sched") or module.startswith("vllm_ascend.core"):
        return "Scheduler / driver"
    if module.startswith("vllm.v1.engine.llm_engine"):
        return "Scheduler / driver"
    if module.startswith("vllm.v1.engine.core"):
        return "Scheduler / driver"
    if module.startswith("vllm.v1.executor"):
        return "Scheduler / driver"
    if module.startswith("vllm_ascend.worker"):
        return "Worker / Ascend"
    if module.startswith("vllm.v1.worker"):
        return "Worker / GPU"
    return "Worker / Other"


def _segment_label(event: PerfEvent) -> str:
    if event.event == "request_phase":
        return event.phase or event.func
    module_tail = event.module.split(".")[-1]
    if event.module.startswith("vllm.v1.worker.gpu.async_utils"):
        module_tail = "async_utils"
    elif event.module.startswith("vllm.v1.worker"):
        module_tail = "worker"
    elif event.module.startswith("vllm.v1.core.sched"):
        module_tail = "scheduler"
    elif event.module.startswith("vllm.v1.engine.llm_engine"):
        module_tail = "llm_engine"
    elif event.module.startswith("vllm_ascend.core"):
        module_tail = event.module.split(".")[-1]
    elif event.module.startswith("vllm_gr.entrypoints.openai"):
        module_tail = "serving_engine"
    if event.module.startswith("vllm.v1.executor.uniproc_executor"):
        method = event.fields.get("method")
        if method:
            return f"{module_tail}.{event.func}.{method}"
    if event.module.startswith("vllm.v1.executor.multiproc_executor"):
        method = event.fields.get("method")
        if method:
            return f"{module_tail}.{event.func}.{method}"
    return f"{module_tail}.{event.func}"


def _description_for_event(event: PerfEvent) -> str:
    module = event.module
    func = event.func
    if event.event == "request_summary":
        return "aggregate request summary"
    if event.event == "request_phase":
        phase = event.phase or func
        descriptions = {
            "add_batch_step": "batch-add request path",
            "prefill_build_prompts_batch": "prefill: build prompt/token batches for the engine request",
            "prefill_build_request_id": "prefill: build the internal batch request id",
            "prepare_request_batch": "build EngineCore requests for the batch",
            "add_requests_batch": "submit the whole batch into the engine core",
            "gather_beam_results": "wait for all queued beam outputs to finish",
            "prefill_emit_collector_metrics": "prefill: emit derived collector timing metrics into the request log",
            "beam_request_step": "beam-update request path",
            "decode_build_prompts_batch": "decode: rebuild per-beam prompt batches before the next engine step",
            "decode_build_request_id": "decode: build the internal request id for the beam-update step",
            "build_fork_info": "build parent/child beam routing metadata",
            "register_beam_output": "register a unified output queue for beam update",
            "beam_request_step_update": "send beam step update into the engine core",
            "collect_beam_result": "wait for the merged beam result to finish",
            "decode_emit_collector_metrics": "decode: emit derived collector timing metrics into the request log",
            "decode_extract_beam_segments": "decode: slice per-beam candidate segments from the merged flat logprobs",
            "decode_filter_valid_tokens": "decode: apply valid-token filtering to each beam candidate list",
            "decode_flatten_candidates": "decode: flatten per-beam candidates into the global ranking arrays",
            "decode_round_total": "decode: full per-round beam-search loop body",
            "merge_candidate_scores": "merge EOS and top-k candidate scores",
            "materialize_next_beams": "build the next beam sequences from ranked candidates",
            "gather_result_stats": "record queue wait statistics for batch beam collection",
            "collect_result_stats": "record queue wait statistics for merged beam collection",
            "prefill_engine_queue_wait": "engine-internal queue wait from queued to first scheduled",
            "prefill_engine_compute": "engine-internal prefill compute from first scheduled to first token ready",
            "prefill_engine_compute_avg": "average engine-internal prefill compute across all beam result queues",
            "prefill_engine_compute_max": "slowest engine-internal prefill compute among beam result queues",
            "prefill_engine_compute_min": "fastest engine-internal prefill compute among beam result queues",
            "prefill_engine_compute_spread": "difference between slowest and fastest prefill engine compute",
            "prefill_exec_submit": "prefill request-scoped: executor execute_model non-block submit cost",
            "prefill_sample_submit": "prefill request-scoped: executor sample_tokens non-block submit cost",
            "prefill_build_grammar_output": "prefill request-scoped: build grammar / structured-output bitmask before sample_tokens",
            "prefill_wait_model_output": "prefill request-scoped: engine core wait on queued future.result()",
            "prefill_update_from_output": "prefill request-scoped: engine core scheduler.update_from_output cost",
            "prefill_worker_total": "prefill request-scoped: worker execute_model plus sample_tokens total",
            "prefill_worker_execute_model": "prefill request-scoped: worker execute_model total",
            "prefill_worker_forward": "prefill request-scoped: worker forward body",
            "prefill_worker_sample_tokens_body": "prefill request-scoped: sample_tokens body before profiling sync",
            "prefill_worker_profile_event_sync": "prefill request-scoped: NPU event synchronize while collecting profiling durations",
            "prefill_worker_sample_tokens_total": "prefill request-scoped: worker sample_tokens total before return",
            "prefill_worker_sample": "prefill request-scoped: worker sampler forward",
            "prefill_worker_bookkeeping": "prefill request-scoped: worker bookkeeping before draft",
            "prefill_worker_build_output": "prefill request-scoped: build ModelRunnerOutput object",
            "prefill_worker_async_output_wrap": "prefill request-scoped: wrap AsyncGPUModelRunnerOutput",
            "prefill_process_aborts": "prefill request-scoped: process abort queue before scheduler update_from_output",
            "prefill_perf_metrics_propagate": "prefill request-scoped: propagate perf metrics into model output and queued outputs",
            "prefill_step_total_wall": "prefill request-scoped: full engine-core batch-queue wall clock",
            "prefill_async_output_queue_wait": "prefill request-scoped: ThreadPoolExecutor queue wait before get_output starts",
            "prefill_async_output_get_output": "prefill request-scoped: async output get_output wall time",
            "prefill_async_gpu_output_sync": "prefill request-scoped: wait for async device-to-host copy event",
            "prefill_async_gpu_output_sampled_token_ids": "prefill request-scoped: materialize sampled token ids on host",
            "prefill_async_gpu_output_logprobs_tolist": "prefill request-scoped: materialize logprobs tensors on host",
            "prefill_async_gpu_output_parse_output": "prefill request-scoped: parse spec decode output on host",
            "prefill_async_gpu_output_assign_output": "prefill request-scoped: assign host results into ModelRunnerOutput",
            "prefill_async_gpu_output_total": "prefill request-scoped: total async GPU output materialization time",
            "prefill_wall_sample_tokens_body": "prefill wall-clock: sample_tokens body before profiling sync",
            "prefill_wall_profile_event_sync": "prefill wall-clock: NPU event synchronize while collecting profiling durations",
            "prefill_wall_async_output_wrap": "prefill wall-clock: wrap AsyncGPUModelRunnerOutput",
            "prefill_wall_perf_log_emit": "prefill wall-clock: emit performance JSON events from sample_tokens",
            "prefill_core_to_frontend": "front-end receive gap from engine-core output timestamp to AsyncLLM receipt",
            "prefill_frontend_handoff": "front-end output handling from socket receipt to request queue put",
            "prefill_queue_visible": "request queue wait from first queue put to first client-visible message",
            "prefill_submit_to_visible": "full prefill first-result path from submit to first visible message",
            "prefill_submit_to_visible_avg": "average full prefill submit-to-visible latency across beam result queues",
            "prefill_submit_to_visible_max": "slowest full prefill submit-to-visible latency among beam result queues",
            "prefill_submit_to_visible_min": "fastest full prefill submit-to-visible latency among beam result queues",
            "prefill_submit_to_visible_spread": "difference between slowest and fastest prefill submit-to-visible latency",
            "prefill_engine_compute_per_queue": "prefill engine compute divided by queue count",
            "prefill_submit_to_visible_per_queue": "prefill submit-to-visible latency divided by queue count",
            "decode_engine_queue_wait": "decode request-scoped: engine-internal queue wait from queued to first scheduled",
            "decode_engine_compute": "decode request-scoped: engine-internal compute from first scheduled to first token ready",
            "decode_exec_submit": "decode request-scoped: executor execute_model non-block submit cost",
            "decode_sample_submit": "decode request-scoped: executor sample_tokens non-block submit cost",
            "decode_build_grammar_output": "decode request-scoped: build grammar / structured-output bitmask before sample_tokens",
            "decode_wait_model_output": "decode request-scoped: engine core wait on queued future.result()",
            "decode_update_from_output": "decode request-scoped: engine core scheduler.update_from_output cost",
            "decode_worker_total": "decode request-scoped: worker execute_model plus sample_tokens total",
            "decode_worker_execute_model": "decode request-scoped: worker execute_model total",
            "decode_worker_forward": "decode request-scoped: worker forward body",
            "decode_worker_sample_tokens_body": "decode request-scoped: sample_tokens body before profiling sync",
            "decode_worker_profile_event_sync": "decode request-scoped: NPU event synchronize while collecting profiling durations",
            "decode_worker_sample_tokens_total": "decode request-scoped: worker sample_tokens total before return",
            "decode_worker_sample": "decode request-scoped: worker sampler forward",
            "decode_worker_bookkeeping": "decode request-scoped: worker bookkeeping before draft",
            "decode_worker_build_output": "decode request-scoped: build ModelRunnerOutput object",
            "decode_worker_async_output_wrap": "decode request-scoped: wrap AsyncGPUModelRunnerOutput",
            "decode_process_aborts": "decode request-scoped: process abort queue before scheduler update_from_output",
            "decode_perf_metrics_propagate": "decode request-scoped: propagate perf metrics into model output and queued outputs",
            "decode_step_total_wall": "decode request-scoped: full engine-core batch-queue wall clock",
            "decode_async_output_queue_wait": "decode request-scoped: ThreadPoolExecutor queue wait before get_output starts",
            "decode_async_output_get_output": "decode request-scoped: async output get_output wall time",
            "decode_async_gpu_output_sync": "decode request-scoped: wait for async device-to-host copy event",
            "decode_async_gpu_output_sampled_token_ids": "decode request-scoped: materialize sampled token ids on host",
            "decode_async_gpu_output_logprobs_tolist": "decode request-scoped: materialize logprobs tensors on host",
            "decode_async_gpu_output_parse_output": "decode request-scoped: parse spec decode output on host",
            "decode_async_gpu_output_assign_output": "decode request-scoped: assign host results into ModelRunnerOutput",
            "decode_async_gpu_output_total": "decode request-scoped: total async GPU output materialization time",
            "decode_core_to_frontend": "decode front-end receive gap from engine-core output timestamp to AsyncLLM receipt",
            "decode_frontend_handoff": "decode front-end output handling from socket receipt to request queue put",
            "decode_queue_visible": "decode request queue wait from first queue put to first client-visible message",
            "decode_submit_to_visible": "decode full first-result path from submit to first visible message",
            "catalog_wait": "wait for catalog filtering",
            "cleanup": "release beam request resources",
            "sort_completed": "sort completed beams by score",
            "reconstruct_logprobs": "rebuild logprobs from parent pointers",
            "decode_best_beams": "decode the final top-ranked beams to text",
        }
        return descriptions.get(phase, DEFAULT_DESCRIPTION)
    if module.startswith("vllm.v1.core.sched") or module.startswith("vllm_ascend.core"):
        if func == "schedule":
            return "scheduler chooses and prepares the next step"
        scheduler_descriptions = {
            "update_from_output_prepare": "scheduler update_from_output: pre-loop prep including connector stats and invalid-block handling",
            "update_from_output_loop_total": "scheduler update_from_output: per-request main loop total",
            "update_from_output_request_update": "scheduler update_from_output: append new tokens and stop-state checks",
            "update_from_output_free_stopped": "scheduler update_from_output: free KV/request state for stopped requests",
            "update_from_output_logprobs": "scheduler update_from_output: slice request logprobs from batched outputs",
            "update_from_output_structured_accept": "scheduler update_from_output: structured-output grammar accept_tokens",
            "update_from_output_build_outputs": "scheduler update_from_output: construct EngineCoreOutput objects",
            "update_from_output_remove_stopped": "scheduler update_from_output: remove stopped requests from running/waiting queues",
            "update_from_output_failed_kv": "scheduler update_from_output: finish requests affected by failed KV loads",
            "update_from_output_kv_xfer_finished": "scheduler update_from_output: finalize finished KV transfer state",
            "update_from_output_kv_events": "scheduler update_from_output: collect and publish KV cache events",
            "update_from_output_build_engine_outputs": "scheduler update_from_output: group per-request outputs into EngineCoreOutputs",
            "update_from_output_attach_finished": "scheduler update_from_output: attach finished-request sets to client outputs",
            "update_from_output_make_stats": "scheduler update_from_output: build scheduler stats payload",
            "update_from_output_total": "scheduler update_from_output: full scheduler-side post-worker processing",
        }
        if func in scheduler_descriptions:
            return scheduler_descriptions[func]
    if module.startswith("vllm.v1.engine.llm_engine") and func == "step":
        return "engine step: get outputs, process them, and record stats"
    if module.startswith("vllm.v1.engine.async_llm"):
        async_llm_descriptions = {
            "output_handler_core_to_frontend": "async frontend wall-clock: engine core output ready to frontend receive gap",
            "output_handler_process_outputs": "async frontend wall-clock: OutputProcessor.process_outputs over one output slice",
        }
        if func in async_llm_descriptions:
            return async_llm_descriptions[func]
    if module.startswith("vllm.v1.engine.core"):
        core_descriptions = {
            "step_wait_model_output": "engine core wall-clock: wait for execute_model future to finish",
            "step_sample_tokens": "engine core wall-clock: sample_tokens call after execute_model returned None",
            "step_update_from_output": "engine core wall-clock: scheduler update_from_output processing",
            "step_total_wall": "engine core wall-clock: full step body after schedule",
            "batch_queue_wait_model_output": "engine core batch-queue wall-clock: wait for queued worker future output",
            "batch_queue_build_grammar_output": "engine core batch-queue wall-clock: build grammar / structured-output bitmask before sample_tokens",
            "batch_queue_wait_execute_model_error": "engine core batch-queue wall-clock: wait for execute_model error propagation",
            "batch_queue_update_from_output": "engine core batch-queue wall-clock: scheduler update_from_output processing",
            "batch_queue_step_total_wall": "engine core batch-queue wall-clock: pop result through scheduler output update",
            "batch_queue_deferred_sample_tokens_submit": "engine core batch-queue wall-clock: submit deferred sample_tokens",
        }
        if func in core_descriptions:
            return core_descriptions[func]
    if module.startswith("vllm.v1.executor.multiproc_executor"):
        executor_descriptions = {
            "future_wrapper_result_total": "executor future wall-clock: resolve requested future after draining older pending responses",
            "future_wrapper_wait_for_response": "executor future wall-clock: aggregate one pending worker response into its future",
            "collective_rpc_mq_dequeue": "executor MQ wall-clock: block on one worker response queue dequeue",
            "collective_rpc_wait_response": "executor RPC wall-clock: wait for all response queues for one collective call",
            "collective_rpc_drain_pending_futures": "executor RPC wall-clock: drain older non-blocking futures before servicing current blocking call",
        }
        if func in executor_descriptions:
            return executor_descriptions[func]
    if module.startswith("vllm.v1.executor.uniproc_executor"):
        executor_descriptions = {
            "executor_init": "executor selection marker for uniprocess backend",
            "collective_rpc_run_method_sync": "executor wall-clock: synchronous worker method execution",
            "collective_rpc_run_method_non_block": "executor wall-clock: non-blocking worker method execution before future handoff",
            "async_output_get_output": "executor wall-clock: background future get_output materialization",
            "async_output_get_output_inline": "executor wall-clock: inline get_output materialization when no async thread is used",
        }
        if func in executor_descriptions:
            method = event.fields.get("method")
            if method:
                return f"{executor_descriptions[func]} ({method})"
            return executor_descriptions[func]
    if module.startswith("vllm.v1.worker.gpu_model_runner"):
        gpu_async_descriptions = {
            "async_gpu_output_sync": "async GPU output: wait for non-blocking device-to-host copy event",
            "async_gpu_output_sampled_token_ids": "async GPU output: materialize sampled token ids on host",
            "async_gpu_output_logprobs_tolist": "async GPU output: materialize logprobs tensors into host lists",
            "async_gpu_output_parse_output": "async GPU output: parse speculative or multi-token sampled output",
            "async_gpu_output_assign_output": "async GPU output: attach host-side sampled outputs to ModelRunnerOutput",
            "async_gpu_output_total": "async GPU output: full get_output materialization path",
        }
        if func in gpu_async_descriptions:
            return gpu_async_descriptions[func]
    if module.startswith("vllm.v1.worker") and func == "execute_model":
        return "worker main execute path"
    if module.startswith("vllm.v1.worker") and func == "sample_tokens":
        return "worker sampling path"
    if module.startswith("vllm.v1.worker.gpu.async_utils") and func == "get_output":
        return "output materialization / handoff"
    worker_stage_descriptions = {
        "prefill_stage_prepare_input": "prefill: build runner inputs and attention metadata",
        "prefill_stage_forward": "prefill: model forward body on worker",
        "prefill_stage_post_process": "prefill: logits build and worker post-processing",
        "prefill_stage_sample": "prefill: sampler and logprob selection",
        "prefill_stage_draft": "prefill: speculative draft handling and cleanup",
        "prefill_wall_prepare_input": "prefill wall-clock: prepare inputs and preprocess",
        "prefill_wall_forward": "prefill wall-clock: forward body and worker-side waits inside forward phase",
        "prefill_wall_post_process": "prefill wall-clock: post-process and logits materialization",
        "prefill_wall_sample": "prefill wall-clock: sampler forward",
        "prefill_wall_bookkeeping": "prefill wall-clock: bookkeeping and output reconstruction before draft",
        "prefill_wall_draft": "prefill wall-clock: speculative draft handling",
        "prefill_wall_build_output": "prefill wall-clock: build model runner output object",
        "prefill_wall_execute_model_total": "prefill wall-clock: execute_model total",
        "prefill_wall_sample_tokens_total": "prefill wall-clock: sample_tokens total",
        "prefill_wall_total": "prefill wall-clock: execute_model plus sample_tokens total",
        "decode_stage_prepare_input": "decode: build runner inputs and attention metadata",
        "decode_stage_forward": "decode: model forward body on worker",
        "decode_stage_post_process": "decode: logits build and worker post-processing",
        "decode_stage_sample": "decode: sampler and logprob selection",
        "decode_stage_draft": "decode: speculative draft handling and cleanup",
    }
    if func in worker_stage_descriptions:
        return worker_stage_descriptions[func]
    qwen3_descriptions = {
        "qwen3_model_input": "prefill/decode: embed or materialize model input",
        "qwen3_model_intermediate_input": "prefill/decode: read pipeline-parallel intermediate input",
        "qwen3_model_layers": "prefill/decode: run decoder layer stack",
        "qwen3_model_norm": "prefill/decode: final model norm",
        "qwen3_decoder_layer": "prefill/decode: one decoder layer end-to-end",
        "qwen3_input_layernorm": "prefill/decode: decoder input RMSNorm",
        "qwen3_self_attn": "prefill/decode: decoder self-attention block",
        "qwen3_attention": "prefill/decode: attention projection and attention core wrapper",
        "qwen3_qk_norm": "prefill/decode: q/k normalization",
        "qwen3_rotary_emb": "prefill/decode: rotary position embedding",
        "qwen3_attn_core": "prefill/decode: attention backend core call",
        "qwen3_o_proj": "prefill/decode: attention output projection",
        "qwen3_post_attention_layernorm": "prefill/decode: post-attention RMSNorm",
        "qwen3_mlp": "prefill/decode: decoder MLP block",
    }
    if func in qwen3_descriptions:
        return qwen3_descriptions[func]
    qwen3_next_descriptions = {
        "qwen3_next_model_input": "prefill/decode: embed or materialize qwen3-next model input",
        "qwen3_next_model_intermediate_input": "prefill/decode: read qwen3-next pipeline-parallel intermediate input",
        "qwen3_next_model_layers": "prefill/decode: run qwen3-next decoder layer stack",
        "qwen3_next_model_norm": "prefill/decode: final qwen3-next model norm",
        "qwen3_next_decoder_layer": "prefill/decode: one qwen3-next decoder layer end-to-end",
        "qwen3_next_input_layernorm": "prefill/decode: qwen3-next decoder input RMSNorm",
        "qwen3_next_linear_attn": "prefill/decode: qwen3-next linear-attention block",
        "qwen3_next_self_attn": "prefill/decode: qwen3-next full-attention block",
        "qwen3_next_attention": "prefill/decode: qwen3-next attention projection and wrapper",
        "qwen3_next_split_q_gate": "prefill/decode: split q and attention output gate",
        "qwen3_next_qk_norm": "prefill/decode: qwen3-next q/k normalization",
        "qwen3_next_rotary_emb": "prefill/decode: qwen3-next rotary position embedding",
        "qwen3_next_attn_core": "prefill/decode: qwen3-next attention backend core call",
        "qwen3_next_attn_output_gate": "prefill/decode: qwen3-next attention output gate",
        "qwen3_next_o_proj": "prefill/decode: qwen3-next attention output projection",
        "qwen3_next_attn_layer_scale": "prefill/decode: qwen3-next attention layer scale",
        "qwen3_next_post_attention_layernorm": "prefill/decode: qwen3-next post-attention RMSNorm",
        "qwen3_next_mlp": "prefill/decode: qwen3-next decoder MLP or sparse MoE block",
        "qwen3_next_ffn_layer_scale": "prefill/decode: qwen3-next FFN layer scale",
    }
    if func in qwen3_next_descriptions:
        return qwen3_next_descriptions[func]
    beam_attn_descriptions = {
        "attention_forward": "beam attention backend forward entry",
        "segment_requests": "split standard and beam requests for attention",
        "build_std_mappings": "build standard-request attention mappings",
        "build_beam_mappings": "build beam suffix/prefix attention mappings",
        "create_std_tensors": "materialize standard attention metadata tensors",
        "create_beam_tensors": "materialize beam attention metadata tensors",
        "reshape_and_cache_npu": "reshape KV and write KV cache on NPU",
        "cascade_npu_std_attention": "run NPU standard-attention branch",
        "cascade_npu_prefix_attention": "run NPU shared-prefix attention branch",
        "cascade_npu_extract_suffix_kv": "extract suffix KV for beam-only attention",
        "cascade_npu_suffix_attention": "run NPU beam suffix attention branch",
        "cascade_npu_merge_attention": "merge beam attention outputs on NPU",
    }
    if module.startswith("vllm_gr.v1.attention.backends.beam_attn") and func in beam_attn_descriptions:
        return beam_attn_descriptions[func]
    sampler_descriptions = {
        "forward": "sampler entry: prepare logits and produce sampled token",
        "compute_raw_logprobs": "compute raw logprobs before sampling transforms",
        "cast_logits_fp32": "cast logits to fp32 for sampling",
        "apply_logits_processors": "apply bad-words, whitelist, penalties, and processors",
        "sample": "sampler wrapper around greedy/random selection",
        "sample_impl": "main sampling implementation",
        "greedy_sample": "argmax path for greedy requests",
        "apply_temperature": "apply temperature scaling to logits",
        "apply_argmax_invariant_processors": "apply min-p and argmax-invariant processors",
        "topk_topp_sampler": "apply top-k/top-p sampling kernel",
        "mix_greedy_and_random": "merge greedy and random sampled outputs",
        "gather_logprobs": "gather requested logprobs for sampled and top tokens",
        "gather_token_ranks": "compute sampled token ranks",
        "gather_token_ranks_compare": "rank-compare sampled tokens against logits",
        "cast_sampled_long": "cast sampled ids to int64",
        "cast_sampled_int32": "cast sampled ids back to int32",
        "forward_native": "Ascend top-k/top-p sampler forward path",
        "apply_top_k_top_p": "Ascend top-k/top-p filter kernel",
        "logits_log_softmax": "build processed logprobs from filtered logits",
        "softmax_probs": "convert logits to probabilities",
        "wait_async_exponential": "wait for async exponential random tensor",
        "random_sample": "sample token from probability distribution",
    }
    if func in sampler_descriptions and (
        module.startswith("vllm.v1.sample.sampler")
        or module.startswith("vllm_ascend.sample.sampler")
    ):
        return sampler_descriptions[func]
    return DEFAULT_DESCRIPTION


def _stage_bucket_for_event(event: PerfEvent) -> tuple[str, str, str] | None:
    phase_or_func = event.phase or event.func
    if event.event == "request_phase":
        if phase_or_func in QUEUEING_PHASES:
            return (
                "Request / queueing",
                "queueing_time",
                "request-side setup and enqueue before worker execution",
            )
        if phase_or_func in COLLECT_MERGE_PHASES:
            return (
                "Request / collect+merge",
                "result_collect_merge_time",
                "request-side waiting for outputs plus final collect/merge work",
            )
        return None

    if (event.module, event.func) in COLLECT_MERGE_FUNCTIONS:
        return (
            "Request / collect+merge",
            "result_collect_merge_time",
            "request-side waiting for outputs plus final collect/merge work",
        )

    group = _group_for_event(event)
    if group.startswith("Worker /"):
        return (
            "Worker / compute",
            "worker_compute_time",
            "worker-side model execution, sampling, and request-state updates",
        )

    return None


def _sort_key(event: PerfEvent) -> tuple[int, str, str]:
    group_order = {
        "Scheduler / driver": 0,
        "Worker / GPU": 1,
        "Worker / Ascend": 2,
        "Worker / Other": 3,
        "API / client": 4,
    }
    group = _group_for_event(event)
    label = _segment_label(event)
    phase_order = {
        "schedule": 0,
        "update_from_output_prepare": 1,
        "update_from_output_loop_total": 2,
        "update_from_output_request_update": 3,
        "update_from_output_free_stopped": 4,
        "update_from_output_logprobs": 5,
        "update_from_output_structured_accept": 6,
        "update_from_output_build_outputs": 7,
        "update_from_output_remove_stopped": 8,
        "update_from_output_failed_kv": 9,
        "update_from_output_kv_xfer_finished": 10,
        "update_from_output_kv_events": 11,
        "update_from_output_build_engine_outputs": 12,
        "update_from_output_attach_finished": 13,
        "update_from_output_make_stats": 14,
        "update_from_output_total": 15,
        "output_handler_core_to_frontend": 16,
        "output_handler_process_outputs": 17,
        "step": 18,
        "execute_model": 19,
        "prefill_stage_prepare_input": 20,
        "prefill_stage_forward": 21,
        "prefill_stage_post_process": 22,
        "prefill_stage_sample": 23,
        "prefill_stage_draft": 24,
        "prefill_wall_prepare_input": 25,
        "prefill_wall_forward": 26,
        "prefill_wall_post_process": 27,
        "prefill_wall_sample": 28,
        "prefill_wall_bookkeeping": 29,
        "prefill_wall_draft": 30,
        "prefill_wall_build_output": 31,
        "prefill_wall_sample_tokens_body": 32,
        "prefill_wall_profile_event_sync": 33,
        "prefill_wall_async_output_wrap": 34,
        "prefill_wall_perf_log_emit": 35,
        "prefill_wall_execute_model_total": 36,
        "prefill_wall_sample_tokens_total": 37,
        "prefill_wall_total": 38,
        "prefill_exec_submit": 39,
        "prefill_sample_submit": 40,
        "prefill_build_grammar_output": 41,
        "prefill_wait_model_output": 42,
        "prefill_update_from_output": 43,
        "prefill_worker_total": 44,
        "prefill_worker_execute_model": 45,
        "prefill_worker_forward": 46,
        "prefill_worker_sample_tokens_body": 47,
        "prefill_worker_profile_event_sync": 48,
        "prefill_worker_sample_tokens_total": 49,
        "prefill_worker_sample": 50,
        "prefill_worker_bookkeeping": 51,
        "prefill_worker_build_output": 52,
        "prefill_worker_async_output_wrap": 53,
        "prefill_async_output_queue_wait": 54,
        "prefill_async_output_get_output": 55,
        "prefill_async_gpu_output_sync": 56,
        "prefill_async_gpu_output_sampled_token_ids": 57,
        "prefill_async_gpu_output_logprobs_tolist": 58,
        "prefill_async_gpu_output_parse_output": 59,
        "prefill_async_gpu_output_assign_output": 60,
        "prefill_async_gpu_output_total": 61,
        "step_wait_model_output": 62,
        "step_sample_tokens": 63,
        "step_update_from_output": 64,
        "step_total_wall": 65,
        "batch_queue_wait_model_output": 66,
        "batch_queue_build_grammar_output": 67,
        "batch_queue_update_from_output": 68,
        "batch_queue_step_total_wall": 69,
        "batch_queue_deferred_sample_tokens_submit": 70,
        "batch_queue_wait_execute_model_error": 71,
        "future_wrapper_result_total": 72,
        "future_wrapper_wait_for_response": 73,
        "collective_rpc_mq_dequeue": 74,
        "collective_rpc_wait_response": 75,
        "collective_rpc_drain_pending_futures": 76,
        "executor_init": 77,
        "collective_rpc_run_method_sync": 78,
        "collective_rpc_run_method_non_block": 79,
        "async_output_get_output": 80,
        "async_output_get_output_inline": 81,
        "decode_stage_prepare_input": 82,
        "decode_stage_forward": 83,
        "decode_stage_post_process": 84,
        "decode_stage_sample": 85,
        "decode_stage_draft": 86,
        "qwen3_model_input": 87,
        "qwen3_model_intermediate_input": 88,
        "qwen3_model_layers": 89,
        "qwen3_model_norm": 90,
        "qwen3_decoder_layer": 91,
        "qwen3_input_layernorm": 92,
        "qwen3_self_attn": 93,
        "qwen3_attention": 94,
        "qwen3_qk_norm": 95,
        "qwen3_rotary_emb": 96,
        "qwen3_attn_core": 97,
        "qwen3_o_proj": 98,
        "qwen3_post_attention_layernorm": 99,
        "qwen3_mlp": 100,
        "qwen3_next_model_input": 101,
        "qwen3_next_model_intermediate_input": 102,
        "qwen3_next_model_layers": 103,
        "qwen3_next_model_norm": 104,
        "qwen3_next_decoder_layer": 105,
        "qwen3_next_input_layernorm": 106,
        "qwen3_next_linear_attn": 107,
        "qwen3_next_self_attn": 108,
        "qwen3_next_attention": 109,
        "qwen3_next_split_q_gate": 110,
        "qwen3_next_qk_norm": 111,
        "qwen3_next_rotary_emb": 112,
        "qwen3_next_attn_core": 113,
        "qwen3_next_attn_output_gate": 114,
        "qwen3_next_o_proj": 115,
        "qwen3_next_attn_layer_scale": 116,
        "qwen3_next_post_attention_layernorm": 117,
        "qwen3_next_mlp": 118,
        "qwen3_next_ffn_layer_scale": 119,
        "attention_forward": 120,
        "segment_requests": 121,
        "build_std_mappings": 122,
        "build_beam_mappings": 123,
        "create_std_tensors": 124,
        "create_beam_tensors": 125,
        "reshape_and_cache_npu": 126,
        "cascade_npu_std_attention": 127,
        "cascade_npu_prefix_attention": 128,
        "cascade_npu_extract_suffix_kv": 129,
        "cascade_npu_suffix_attention": 130,
        "cascade_npu_merge_attention": 131,
        "forward": 132,
        "compute_raw_logprobs": 133,
        "cast_logits_fp32": 134,
        "apply_logits_processors": 135,
        "sample": 136,
        "sample_impl": 137,
        "greedy_sample": 138,
        "apply_temperature": 139,
        "apply_argmax_invariant_processors": 140,
        "topk_topp_sampler": 141,
        "mix_greedy_and_random": 142,
        "gather_logprobs": 143,
        "gather_token_ranks": 144,
        "gather_token_ranks_compare": 145,
        "cast_sampled_long": 146,
        "cast_sampled_int32": 147,
        "forward_native": 148,
        "apply_top_k_top_p": 149,
        "logits_log_softmax": 150,
        "softmax_probs": 151,
        "wait_async_exponential": 152,
        "random_sample": 153,
        "_prepare_inputs": 154,
        "sample_tokens": 155,
        "_bookkeeping_sync": 156,
        "get_output": 157,
        "add_batch_step": 158,
        "prefill_build_prompts_batch": 159,
        "prefill_build_request_id": 160,
        "prepare_request_batch": 161,
        "add_requests_batch": 162,
        "gather_beam_results": 163,
        "prefill_emit_collector_metrics": 164,
        "beam_request_step": 165,
        "decode_build_prompts_batch": 166,
        "decode_build_request_id": 167,
        "build_fork_info": 168,
        "register_beam_output": 169,
        "beam_request_step_update": 170,
        "collect_beam_result": 171,
        "decode_emit_collector_metrics": 172,
        "decode_engine_queue_wait": 173,
        "decode_engine_compute": 174,
        "decode_exec_submit": 175,
        "decode_sample_submit": 176,
        "decode_build_grammar_output": 177,
        "decode_wait_model_output": 178,
        "decode_update_from_output": 179,
        "decode_worker_total": 180,
        "decode_worker_execute_model": 181,
        "decode_worker_forward": 182,
        "decode_worker_sample_tokens_body": 183,
        "decode_worker_profile_event_sync": 184,
        "decode_worker_sample_tokens_total": 185,
        "decode_worker_sample": 186,
        "decode_worker_bookkeeping": 187,
        "decode_worker_build_output": 188,
        "decode_worker_async_output_wrap": 189,
        "decode_async_output_queue_wait": 190,
        "decode_async_output_get_output": 191,
        "decode_async_gpu_output_sync": 192,
        "decode_async_gpu_output_sampled_token_ids": 193,
        "decode_async_gpu_output_logprobs_tolist": 194,
        "decode_async_gpu_output_parse_output": 195,
        "decode_async_gpu_output_assign_output": 196,
        "decode_async_gpu_output_total": 197,
        "decode_core_to_frontend": 198,
        "decode_frontend_handoff": 199,
        "decode_queue_visible": 200,
        "decode_submit_to_visible": 201,
        "decode_extract_beam_segments": 202,
        "decode_filter_valid_tokens": 203,
        "decode_flatten_candidates": 204,
        "decode_round_total": 205,
        "merge_candidate_scores": 206,
        "materialize_next_beams": 207,
        "gather_result_stats": 208,
        "collect_result_stats": 209,
        "catalog_wait": 210,
        "cleanup": 211,
        "sort_completed": 212,
        "reconstruct_logprobs": 213,
        "decode_best_beams": 214,
        "prefill_core_to_frontend": 215,
    }
    label_key = event.phase or event.func
    return (group_order.get(group, 99), phase_order.get(label_key, 999), label)


def _aggregate_events(events: list[PerfEvent]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    representatives: dict[tuple[str, str], PerfEvent] = {}

    for event in events:
        if event.event not in {"function", "request_phase", "request_summary"}:
            continue
        label = _segment_label(event)
        key = (_group_for_event(event), label)
        buckets[key].append(event.duration_ms)
        representatives.setdefault(key, event)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(buckets.keys(), key=lambda item: (_sort_key(representatives[item]), item[1])):
        group, label = key
        representative = representatives[key]
        durations = buckets[key]
        summary_rows.append(
            {
                "Evidence group": group,
                "Segment / event": label,
                "Avg us": round(sum(durations) * 1000.0 / len(durations), 3),
                "P99 us": round(_percentile(durations, 0.99) * 1000.0, 3),
                "What happens": _description_for_event(representative),
            }
        )
    rows: list[dict[str, Any]] = []
    start_us = 0.0
    for row in summary_rows:
        avg_us = row["Avg us"]
        end_us = round(start_us + avg_us, 3)
        rows.append(
            {
                **row,
                "Start us": round(start_us, 3),
                "End us": end_us,
            }
        )
        start_us = end_us
    return rows


def _request_timeline(events: list[PerfEvent], request_id: str) -> list[dict[str, Any]]:
    filtered = [event for event in events if event.request_id == request_id]
    if not filtered:
        return []

    ordered = [event for event in filtered if event.event in {"request_phase", "function"}]
    ordered.sort(key=_sort_key)

    rows: list[dict[str, Any]] = []
    start_us = 0.0
    for event in ordered:
        avg_us = round(event.duration_ms * 1000.0, 3)
        end_us = round(start_us + avg_us, 3)
        label = _segment_label(event)
        rows.append(
            {
                "Evidence group": _group_for_event(event),
                "Segment / event": label,
                "Start us": round(start_us, 3),
                "End us": end_us,
                "Avg us": avg_us,
                "P99 us": avg_us,
                "What happens": _description_for_event(event),
            }
        )
        start_us = end_us
    return rows


def _build_stage_summary_rows(
    events: list[PerfEvent],
    request_id: str | None = None,
) -> list[dict[str, Any]]:
    relevant_events = [
        event
        for event in events
        if request_id is None or event.request_id == request_id
    ]
    if not relevant_events:
        return []

    category_label_buckets: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for event in relevant_events:
        bucket = _stage_bucket_for_event(event)
        if bucket is None:
            continue
        label = _segment_label(event)
        category_label_buckets[bucket][label].append(event.duration_ms)

    rows: list[dict[str, Any]] = []
    start_us = 0.0
    ordered_buckets = [
        (
            "Request / queueing",
            "queueing_time",
            "request-side setup and enqueue before worker execution",
        ),
        (
            "Worker / compute",
            "worker_compute_time",
            "worker-side model execution, sampling, and request-state updates",
        ),
        (
            "Request / collect+merge",
            "result_collect_merge_time",
            "request-side waiting for outputs plus final collect/merge work",
        ),
    ]
    for bucket in ordered_buckets:
        label_buckets = category_label_buckets.get(bucket)
        if not label_buckets:
            continue
        avg_us = round(
            sum(sum(samples) / len(samples) for samples in label_buckets.values()) * 1000.0,
            3,
        )
        p99_us = round(
            sum(_percentile(samples, 0.99) for samples in label_buckets.values()) * 1000.0,
            3,
        )
        end_us = round(start_us + avg_us, 3)
        rows.append(
            {
                "Evidence group": bucket[0],
                "Segment / event": bucket[1],
                "Start us": round(start_us, 3),
                "End us": end_us,
                "Avg us": avg_us,
                "P99 us": p99_us,
                "What happens": bucket[2],
            }
        )
        start_us = end_us
    return rows


def _write_csv(rows: list[dict[str, Any]], output: Path | None) -> None:
    fieldnames = [
        "Evidence group",
        "Segment / event",
        "Start us",
        "End us",
        "Avg us",
        "P99 us",
        "What happens",
    ]
    out = sys.stdout if output is None else output.open("w", encoding="utf-8", newline="")
    close_after = output is not None
    try:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if close_after:
            out.close()


def build_rows(
    events: list[PerfEvent],
    request_id: str | None = None,
    stage_summary: bool = False,
) -> list[dict[str, Any]]:
    if stage_summary:
        return _build_stage_summary_rows(events, request_id=request_id)
    if request_id is not None:
        rows = _request_timeline(events, request_id)
        if rows:
            return rows
    return _aggregate_events(events)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Path to the server log file. Defaults to stdin.")
    parser.add_argument("--output", type=Path, help="Path to write the CSV. Defaults to stdout.")
    parser.add_argument(
        "--request-id",
        help="If set, only summarize one request_id and reconstruct a single timeline.",
    )
    parser.add_argument(
        "--stage-summary",
        action="store_true",
        help=(
            "Output a 3-way roll-up of queueing time, worker compute time, "
            "and result collect/merge time."
        ),
    )
    args = parser.parse_args(argv)

    events = load_events(args.input)
    rows = build_rows(
        events,
        request_id=args.request_id,
        stage_summary=args.stage_summary,
    )
    _write_csv(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
