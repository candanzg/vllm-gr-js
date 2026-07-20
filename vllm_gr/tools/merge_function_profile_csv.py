# SPDX-License-Identifier: Apache-2.0

"""Merge low-overhead function-profile CSVs into a sample-style summary table.

Input files are the per-process CSVs emitted by
``vllm.v1.metrics.function_profile``.

Example:

    python -m vllm_gr.tools.merge_function_profile_csv \
      --input-dir /tmp/perf \
      --output function_hotspots.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class FunctionRow:
    module: str
    func: str
    count: int
    total_us: float
    avg_us: float
    p99_us: float
    max_us: float


def _iter_input_paths(input_dir: Path | None, inputs: list[Path]) -> list[Path]:
    paths = list(inputs)
    if input_dir is not None:
        paths.extend(sorted(input_dir.glob("vllm_function_profile_*.csv")))
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(resolved)
    return unique_paths


def _load_rows(paths: Iterable[Path]) -> list[FunctionRow]:
    rows: list[FunctionRow] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                p99_raw = raw.get("p99_us")
                if p99_raw is None:
                    p99_raw = raw.get("p50_us")
                if p99_raw is None:
                    p99_raw = raw.get("avg_us")
                rows.append(
                    FunctionRow(
                        module=raw["module"],
                        func=raw["func"],
                        count=int(raw["count"]),
                        total_us=float(raw["total_us"]),
                        avg_us=float(raw["avg_us"]),
                        p99_us=float(p99_raw),
                        max_us=float(raw["max_us"]),
                    )
                )
    return rows


def _group_for_module(module: str) -> str:
    if module.startswith("vllm.v1.core.sched") or module.startswith(
        "vllm.v1.engine"
    ):
        return "Scheduler / driver"
    if module.startswith("vllm.v1.sample"):
        return "Worker / Ascend"
    if module.startswith("vllm.model_executor.models"):
        return "Worker / Ascend"
    if module.startswith("vllm_ascend.worker"):
        return "Worker / Ascend"
    if module.startswith("vllm_ascend.sample"):
        return "Worker / Ascend"
    if module.startswith("vllm_ascend.attention"):
        return "Worker / Ascend"
    if module.startswith("vllm_gr.v1.attention"):
        return "Worker / Ascend"
    if module.startswith("vllm.v1.worker"):
        return "Worker / GPU"
    if module.startswith("vllm_gr.entrypoints"):
        return "API / client"
    if module.startswith("vllm_gr.v1.engine"):
        return "API / client"
    if module.startswith("vllm_gr.ops.npu"):
        return "Worker / Ascend"
    return "Other"


def _segment_label(module: str, func: str) -> str:
    return f"{module}.{func}"


def _description(module: str, func: str) -> str:
    if module == "vllm.v1.core.sched.scheduler" and func == "schedule":
        return "scheduler chooses the next execution step"
    if module == "vllm.v1.engine.llm_engine" and func == "step":
        return "engine step orchestration"
    if module == "vllm.model_executor.models.qwen2" and func == "qwen2_model_forward":
        return "model forward entry"
    if module == "vllm.model_executor.models.qwen2" and func == "qwen2_model_input":
        return "build model input embeddings"
    if module == "vllm.model_executor.models.qwen2" and func == "qwen2_model_intermediate_input":
        return "restore intermediate hidden states"
    if module == "vllm.model_executor.models.qwen2" and func == "qwen2_model_layers":
        return "run the transformer layer stack"
    if module == "vllm.model_executor.models.qwen2" and func == "qwen2_model_norm":
        return "apply final normalization"
    if module == "vllm.model_executor.models.qwen2" and func == "qwen2_for_causal_lm_forward":
        return "causal LM forward wrapper"
    if module == "vllm.model_executor.models.llama" and func == "llama_model_forward":
        return "model forward entry"
    if module == "vllm.model_executor.models.llama" and func == "llama_model_input":
        return "build model input embeddings"
    if module == "vllm.model_executor.models.llama" and func == "llama_model_intermediate_input":
        return "restore intermediate hidden states"
    if module == "vllm.model_executor.models.llama" and func == "llama_model_layers":
        return "run the transformer layer stack"
    if module == "vllm.model_executor.models.llama" and func == "llama_model_norm":
        return "apply final normalization"
    if module == "vllm.model_executor.models.llama" and func == "llama_for_causal_lm_forward":
        return "causal LM forward wrapper"
    if module == "vllm.model_executor.models.qwen3" and func == "qwen3_for_causal_lm_forward":
        return "causal LM forward wrapper"
    if module == "vllm.model_executor.models.qwen3" and func == "qwen3_attention_forward":
        return "attention module forward entry"
    if module == "vllm.model_executor.models.qwen3" and func == "qwen3_model_forward":
        return "model forward entry"
    if module == "vllm.model_executor.models.qwen3" and func == "qwen3_model_input":
        return "build model input embeddings"
    if module == "vllm.model_executor.models.qwen3" and func == "qwen3_model_intermediate_input":
        return "restore intermediate hidden states"
    if module == "vllm.model_executor.models.qwen3" and func == "qwen3_model_layers":
        return "run the transformer layer stack"
    if module == "vllm.model_executor.models.qwen3" and func == "qwen3_model_norm":
        return "apply final normalization"
    if module == "vllm.model_executor.models.qwen3" and func == "qwen3_decoder_layer":
        return "decoder block execution"
    if module == "vllm.model_executor.models.qwen3" and func == "qwen3_decoder_layer_forward":
        return "decoder layer forward entry"
    if module == "vllm.model_executor.models.qwen3" and func == "qwen3_attention":
        return "attention block internals"
    if module == "vllm.model_executor.models.qwen3" and func == "qwen3_qk_norm":
        return "apply q/k RMS normalization"
    if module == "vllm.model_executor.models.qwen3" and func == "qwen3_rotary_emb":
        return "apply rotary position embedding"
    if module == "vllm.model_executor.models.qwen3" and func == "qwen3_attn_core":
        return "run the core attention kernel"
    if module == "vllm.model_executor.models.qwen3" and func == "qwen3_o_proj":
        return "project attention output back to hidden size"
    if module == "vllm.model_executor.models.qwen3" and func == "qwen3_input_layernorm":
        return "apply pre-attention RMSNorm"
    if module == "vllm.model_executor.models.qwen3" and func == "qwen3_self_attn":
        return "self-attention block"
    if module == "vllm.model_executor.models.qwen3" and func == "qwen3_post_attention_layernorm":
        return "apply post-attention RMSNorm"
    if module == "vllm.model_executor.models.qwen3" and func == "qwen3_mlp":
        return "feed-forward block"
    if module == "vllm.v1.sample.sampler" and func == "forward":
        return "sampler pipeline entry for logits post-processing and token selection"
    if module == "vllm.v1.sample.sampler" and func == "sample":
        return "dispatch between greedy and stochastic sampling modes"
    if module == "vllm.v1.sample.sampler" and func == "sample_impl":
        return "core sampler path for greedy or stochastic token selection"
    if module == "vllm.v1.sample.sampler" and func == "greedy_sample":
        return "pick argmax tokens for greedy requests"
    if module == "vllm.v1.sample.sampler" and func == "compute_raw_logprobs":
        return "compute full-vocabulary logprobs before sampling"
    if module == "vllm.v1.sample.sampler" and func == "gather_logprobs":
        return "gather sampled-token and top-k logprob tensors"
    if module == "vllm.v1.sample.sampler" and func == "gather_topk_logprobs":
        return "select top-k logprobs from the full vocabulary tensor"
    if module == "vllm.v1.sample.sampler" and func == "gather_token_logprobs":
        return "gather the chosen token logprob for each request"
    if module == "vllm.v1.sample.sampler" and func == "gather_token_ranks":
        return "rank sampled tokens against the full vocabulary logprobs"
    if module == "vllm.v1.sample.sampler" and func == "gather_token_ranks_compare":
        return "count how many vocabulary logits outrank the chosen token"
    if module == "vllm.v1.sample.sampler" and func == "concat_logprobs_outputs":
        return "merge sampled-token and top-k logprob outputs"
    if module == "vllm.v1.sample.topk_topp_sampler" and func == "forward_native":
        return "native top-k/top-p filtering and random sampling path"
    if module == "vllm.v1.sample.topk_topp_sampler" and func == "apply_top_k_top_p":
        return "apply top-k/top-p masks before random sampling"
    if module == "vllm.v1.sample.topk_topp_sampler" and func == "random_sample":
        return "sample from filtered probabilities with exponential noise"
    if module == "vllm.v1.worker.gpu.async_utils" and func == "get_output":
        return "materialize async output back to host-visible structures"
    if module == "vllm_gr.ops.npu.split_qkv_rmsnorm_rope" and func == "split_qkv_rmsnorm_rope_gr_impl":
        return "fused QKV split, RMSNorm, and RoPE patch path"
    if module == "vllm_gr.ops.npu.split_qkv_rmsnorm_rope" and func == "prepare_kernel_launch":
        return "prepare launch arguments for the fused NPU kernel"
    if module == "vllm_gr.ops.npu.split_qkv_rmsnorm_rope" and func == "launch_split_qkv_rmsnorm_rope_kernel":
        return "launch the fused split-qkv-rmsnorm-rope kernel"
    if module == "vllm_gr.ops.npu.split_qkv_rmsnorm_rope" and func == "convert_sin_cos_cache":
        return "convert rotary cache tensors into kernel input layout"
    if module == "vllm_gr.ops.npu.split_qkv_rmsnorm_rope" and func == "init_triton_device_props":
        return "initialize cached Triton device properties"
    if module == "vllm_gr.entrypoints.openai.serving_engine" and func == "_collect_beam_result":
        return "wait for one beam result queue to finish"
    if module == "vllm_gr.entrypoints.openai.serving_engine" and func == "collect_wait_first_message":
        return "wait for the first message to arrive on a beam result queue"
    if module == "vllm_gr.entrypoints.openai.serving_engine" and func == "collect_wait_tail_messages":
        return "wait for follow-up messages until the beam result finishes"
    if module == "vllm_gr.entrypoints.openai.serving_engine" and func == "_gather_beam_results":
        return "wait for all beam result queues to finish"
    if module == "vllm_gr.entrypoints.openai.serving_engine" and func == "_add_batch_step":
        return "prepare and submit the batch beam requests"
    if module == "vllm_gr.entrypoints.openai.serving_engine" and func == "prefill_build_prompts_batch":
        return "build prompt/token batches for the prefill engine request"
    if module == "vllm_gr.entrypoints.openai.serving_engine" and func == "prefill_build_request_id":
        return "build the internal batch request id for prefill"
    if module == "vllm_gr.entrypoints.openai.serving_engine" and func == "prefill_emit_collector_metrics":
        return "emit derived prefill collector timing metrics into the request log"
    if module == "vllm_gr.entrypoints.openai.serving_engine" and func == "decode_build_prompts_batch":
        return "rebuild per-beam prompt batches before the next decode step"
    if module == "vllm_gr.entrypoints.openai.serving_engine" and func == "decode_build_request_id":
        return "build the internal request id for the decode beam-update step"
    if module == "vllm_gr.entrypoints.openai.serving_engine" and func == "decode_emit_collector_metrics":
        return "emit derived decode collector timing metrics into the request log"
    if module == "vllm_gr.entrypoints.openai.serving_engine" and func == "_beam_request_step":
        return "submit beam-update requests and collect results"
    if module == "vllm_gr.entrypoints.openai.serving_engine" and func == "decode_extract_beam_segments":
        return "slice per-beam candidate segments from the merged flat logprobs"
    if module == "vllm_gr.entrypoints.openai.serving_engine" and func == "decode_filter_valid_tokens":
        return "apply valid-token filtering to each beam candidate list"
    if module == "vllm_gr.entrypoints.openai.serving_engine" and func == "decode_flatten_candidates":
        return "flatten per-beam candidates into the global ranking arrays"
    if module == "vllm_gr.entrypoints.openai.serving_engine" and func == "decode_round_total":
        return "full per-round beam-search loop body"
    if module == "vllm_gr.entrypoints.openai.serving_engine" and func == "_beam_request_cleanup":
        return "cleanup beam request state in the engine"
    if module == "vllm_gr.v1.engine.async_llm" and func == "prepare_request_validate_truncation":
        return "validate request truncation settings"
    if module == "vllm_gr.v1.engine.async_llm" and func == "prepare_request_process_inputs":
        return "convert user input into an engine request"
    if module == "vllm_gr.v1.engine.async_llm" and func == "prepare_request_prompt_text":
        return "extract display text for request logging"
    if module == "vllm_gr.v1.engine.async_llm" and func == "prepare_request_assign_request_id":
        return "assign the engine-visible request identity"
    if module == "vllm_gr.v1.engine.async_llm" and func == "prepare_request_run_output_handler":
        return "ensure the output handler is running"
    if module == "vllm_gr.v1.engine.async_llm" and func == "prepare_request_create_output_collector":
        return "create a per-request output collector"
    if module == "vllm_gr.v1.engine.async_llm" and func == "prepare_request_add_output_request":
        return "register the request in the output processor"
    if module == "vllm_gr.v1.engine.async_llm" and func == "add_requests_batch":
        return "submit a batch of prepared requests to the engine core"
    if module == "vllm_gr.v1.engine.async_llm" and func == "register_beam_output_create_request":
        return "build the beam output placeholder request"
    if module == "vllm_gr.v1.engine.async_llm" and func == "register_beam_output_run_output_handler":
        return "ensure the beam output handler is running"
    if module == "vllm_gr.v1.engine.async_llm" and func == "register_beam_output_create_output_collector":
        return "create the beam output collector"
    if module == "vllm_gr.v1.engine.async_llm" and func == "register_beam_output_add_request":
        return "register the beam output request"
    if module == "vllm_gr.v1.engine.async_llm" and func == "beam_request_step_update":
        return "send a grouped beam step update to the engine core"
    if module == "vllm_gr.v1.attention.backends.beam_attn" and func == "attention_forward":
        return "beam attention forward entry"
    if module == "vllm_gr.v1.attention.backends.beam_attn" and func == "segment_requests":
        return "split requests into standard and beam attention groups"
    if module == "vllm_gr.v1.attention.backends.beam_attn" and func == "build_std_mappings":
        return "build standard request attention mappings"
    if module == "vllm_gr.v1.attention.backends.beam_attn" and func == "build_beam_mappings":
        return "build beam request attention mappings"
    if module == "vllm_gr.v1.attention.backends.beam_attn" and func == "create_std_tensors":
        return "materialize standard attention tensors"
    if module == "vllm_gr.v1.attention.backends.beam_attn" and func == "create_beam_tensors":
        return "materialize beam attention tensors"
    if module == "vllm_gr.v1.attention.backends.beam_attn" and func == "reshape_and_cache_npu":
        return "cache kv tensors on NPU"
    if module == "vllm_gr.v1.attention.backends.beam_attn" and func == "reshape_and_cache_gpu":
        return "cache kv tensors on GPU"
    if module == "vllm_gr.v1.attention.backends.beam_attn" and func == "cascade_npu_std_attention":
        return "NPU standard attention kernel"
    if module == "vllm_gr.v1.attention.backends.beam_attn" and func == "cascade_npu_prefix_attention":
        return "NPU prefix attention kernel"
    if module == "vllm_gr.v1.attention.backends.beam_attn" and func == "cascade_npu_extract_suffix_kv":
        return "NPU suffix KV gather"
    if module == "vllm_gr.v1.attention.backends.beam_attn" and func == "cascade_npu_suffix_attention":
        return "NPU suffix attention kernel"
    if module == "vllm_gr.v1.attention.backends.beam_attn" and func == "cascade_npu_merge_attention":
        return "NPU merge prefix and suffix attention outputs"
    if module == "vllm_gr.v1.attention.backends.beam_attn" and func == "cascade_gpu_std_attention":
        return "GPU standard attention kernel"
    if module == "vllm_gr.v1.attention.backends.beam_attn" and func == "cascade_gpu_prefix_attention":
        return "GPU prefix attention kernel"
    if module == "vllm_gr.v1.attention.backends.beam_attn" and func == "cascade_gpu_extract_suffix_kv":
        return "GPU suffix KV gather"
    if module == "vllm_gr.v1.attention.backends.beam_attn" and func == "cascade_gpu_suffix_attention":
        return "GPU suffix attention kernel"
    if module == "vllm_gr.v1.attention.backends.beam_attn" and func == "cascade_gpu_merge_attention":
        return "GPU merge prefix and suffix attention outputs"
    if module.endswith("worker.worker") and func == "execute_model":
        return "worker-level execute path"
    if module.endswith("worker.worker") and func == "sample_tokens":
        return "worker-level sampling path"
    if module.endswith("worker.model_runner_v1") and func == "execute_model":
        return "model runner forward and post-process path"
    if module.endswith("worker.model_runner_v1") and func == "sample_tokens":
        return "model runner sampling and draft handling path"
    if module.endswith("worker.model_runner_v1") and func == "_sample":
        return "run the sampling kernel and logits post-processing"
    if module.endswith("worker.model_runner_v1") and func == "_prepare_inputs":
        return "build runner inputs and attention metadata"
    if module.endswith("worker.model_runner_v1") and func == "_bookkeeping_sync":
        return "update request state and reconstruct sampled outputs"
    if module.endswith("worker.model_runner_v1") and func == "parse_output":
        return "parse sampled tokens into request-level outputs"
    if module.endswith("worker.model_runner_v1") and func == "writeback_tokens":
        return "write sampled tokens back into request state"
    if module.endswith("worker.model_runner_v1") and func == "writeback_token_arrays":
        return "copy sampled token arrays into per-request buffers"
    if module.endswith("worker.model_runner_v1") and func == "writeback_request_state":
        return "update request bookkeeping after token writeback"
    if module.endswith("worker.model_runner_v1") and func == "prompt_logprobs":
        return "build prompt logprob outputs"
    if module.endswith("worker.model_runner_v1") and func == "compute_prompt_logprobs":
        return "compute prompt logprob tensors for output materialization"
    if module.endswith("worker.model_runner_v1") and func == "execute_forward":
        return "full model forward entry"
    if module.endswith("worker.model_runner_v1") and func == "execute_model_call":
        return "call the model module for hidden-state generation"
    if module.endswith("worker.model_runner_v1") and func == "execute_model_call_forward":
        return "invoke the model forward pass itself"
    if module.endswith("worker.model_runner_v1") and func == "execute_update_full_graph_params":
        return "update full-graph runtime parameters after forward"
    if module.endswith("worker.model_runner_v1") and func == "execute_all_gather_hidden_states":
        return "all-gather hidden states across sequence parallel ranks"
    if module.endswith("worker.model_runner_v1") and func == "execute_restore_hidden_states":
        return "restore hidden states after PCP partitioning"
    if module.endswith("worker.model_runner_v1") and func == "execute_compute_logits":
        return "compute logits from hidden states"
    if module.endswith("worker.model_runner_v1") and func == "execute_prepare_inputs":
        return "prepare input tensors and scheduling metadata"
    if module.endswith("worker.model_runner_v1") and func == "execute_preprocess":
        return "preprocess request inputs before forward"
    if module.endswith("worker.model_runner_v1") and func == "execute_update_states":
        return "advance request state before the forward step"
    if module.endswith("worker.model_runner_v1") and func == "execute_update_cos_sin":
        return "update rotary cosine and sine caches"
    if module.endswith("worker.model_runner_v1") and func == "execute_slice_sample_hidden_states":
        return "slice hidden states for token sampling"
    if module.endswith("worker.model_runner_v1") and func == "sample_main_sampler":
        return "run the main sampler on the logits tensor"
    if module.endswith("worker.model_runner_v1") and func == "sample_prepare_logits":
        return "trim logits to active requests before sampling"
    if module.endswith("worker.model_runner_v1") and func == "sample_prepare_spec_logits":
        return "trim speculative logits to active sampled positions"
    if module.endswith("worker.model_runner_v1") and func == "sample_rejection_sampler":
        return "run rejection sampling for speculative decode"
    if module.endswith("worker.model_runner_v1") and func == "sample_update_accepted_tokens":
        return "update request state from accepted speculative tokens"
    if module.endswith("worker.model_runner_v1") and func == "collect_sampled_updates":
        return "collect sampled-token updates before output assembly"
    if module.endswith("worker.model_runner_v1") and func == "materialize_logprobs_lists":
        return "materialize logprob tensors into Python-side lists"
    if module.endswith("worker.model_runner_v1") and func == "build_async_output":
        return "package async worker output tensors and metadata"
    if module.endswith("worker.model_runner_v1") and func == "build_model_runner_output":
        return "package the final model runner output object"
    return "function-level hotspot"


def _sort_key(row: dict[str, object]) -> tuple[int, float, str]:
    group_order = {
        "Scheduler / driver": 0,
        "Worker / GPU": 1,
        "Worker / Ascend": 2,
        "API / client": 3,
        "Other": 4,
    }
    return (
        group_order.get(str(row["Evidence group"]), 99),
        -float(row["total_us"]),
        str(row["Segment / event"]),
    )


def build_rows(function_rows: list[FunctionRow]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str], list[FunctionRow]] = defaultdict(list)
    for row in function_rows:
        key = (row.module, row.func)
        buckets[key].append(row)

    merged: list[dict[str, object]] = []
    for (module, func), rows in buckets.items():
        total_count = sum(row.count for row in rows)
        total_us = sum(row.total_us for row in rows)
        weighted_avg_sum = sum(row.avg_us * row.count for row in rows)
        max_p99 = max(row.p99_us for row in rows)
        merged.append(
            {
                "Evidence group": _group_for_module(module),
                "Segment / event": _segment_label(module, func),
                "Avg us": round(weighted_avg_sum / total_count, 3)
                if total_count
                else 0.0,
                "P99 us": round(max_p99, 3),
                "What happens": _description(module, func),
                "total_us": round(total_us, 3),
            }
        )

    merged.sort(key=_sort_key)

    start_us = 0.0
    output_rows: list[dict[str, object]] = []
    for row in merged:
        width_us = float(row["Avg us"])
        end_us = round(start_us + width_us, 3)
        output_rows.append(
            {
                "Evidence group": row["Evidence group"],
                "Segment / event": row["Segment / event"],
                "Start us": round(start_us, 3),
                "End us": end_us,
                "Avg us": row["Avg us"],
                "P99 us": row["P99 us"],
                "What happens": row["What happens"],
            }
        )
        start_us = end_us
    return output_rows


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Evidence group",
                "Segment / event",
                "Start us",
                "End us",
                "Avg us",
                "P99 us",
                "What happens",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing vllm_function_profile_*.csv files.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="*",
        default=[],
        help="One or more explicit function-profile CSV files.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    args = parser.parse_args(argv)

    input_paths = _iter_input_paths(args.input_dir, args.input)
    if not input_paths:
        raise SystemExit("No function-profile CSV files found.")

    rows = build_rows(_load_rows(input_paths))
    write_csv(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
