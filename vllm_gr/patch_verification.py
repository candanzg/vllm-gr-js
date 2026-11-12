# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Postconditions for runtime hooks outside the Beam installation plans."""

from __future__ import annotations

import inspect
import sys

from vllm_gr.compat.runtime_v0_30 import resolve_symbol


def _marker(symbol: str, method: str, marker: str) -> str | None:
    target = getattr(resolve_symbol(symbol), method, None)
    if not callable(target) or not getattr(target, marker, False):
        return f"{symbol}.{method} is missing marker {marker}"
    return None


def verify_selector() -> str | None:
    return _marker("selector", "_cached_get_attn_backend", "_gr_attention_selector")


def verify_npu_worker() -> str | None:
    return _marker("NPUWorker", "init_device", "_gr_npu_worker_init")


def verify_penalties() -> str | None:
    return _marker("warmup", "penalties_triton_warmup", "_vllm_gr_oom_guarded")


def verify_pangu() -> str | None:
    return _marker(
        "ModelConfig", "maybe_untie_word_embeddings", "_vllm_gr_pangu_tied_embeddings_patch"
    )


def verify_spawn() -> str | None:
    error = _marker("system_utils", "_maybe_force_spawn", "_vllm_gr_npu_spawn_original")
    if error:
        return error
    if (
        resolve_symbol("system_utils")._maybe_force_spawn
        is not resolve_symbol("multiproc_executor")._maybe_force_spawn
    ):
        return "multiproc_executor._maybe_force_spawn is not the installed wrapper"
    return None


def verify_sliced_logits() -> str | None:
    for symbol, name in (
        ("LogitsProcessor", "__init__"),
        ("LogitsProcessor", "_get_logits"),
        ("Sampler", "gather_logprobs"),
    ):
        error = _marker(symbol, name, "_gr_sliced_logits_original")
        if error:
            return error
    if not isinstance(
        inspect.getattr_static(resolve_symbol("Sampler"), "gather_logprobs"), staticmethod
    ):
        return "Sampler.gather_logprobs must remain a staticmethod"
    from vllm_gr.patch_report import runtime_platform

    if runtime_platform() != "cuda":
        return None
    # Verify loaded aliases too, without importing an unused runner.
    for name in (
        "vllm.v1.worker.gpu.sample.logprob",
        "vllm.v1.worker.gpu.sample.sampler",
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler",
    ):
        module = sys.modules.get(name)
        target = getattr(module, "compute_topk_scores", None)
        if target is not None and not getattr(target, "_vllm_gr_clamped_topk", False):
            return f"{name}.compute_topk_scores is not clamped"
    return None


def verify_config_filter() -> str | None:
    error = _marker("ascend_config", "init_ascend_config", "_vllm_gr_sanitized")
    if error:
        return error
    original = resolve_symbol("ascend_config").init_ascend_config
    for name in ("vllm_ascend.patch.platform.patch_engine_core", "vllm_ascend.worker.worker"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "init_ascend_config", original) is not original:
            return f"{name}.init_ascend_config is not the installed filter"
    return None


def verify_config_entrypoint() -> str | None:
    from vllm_gr.v1.ascend_compat import _sanitize_engine_core_entry

    owner = resolve_symbol("EngineCoreProc")
    if owner.run_engine_core is not _sanitize_engine_core_entry:
        return "EngineCoreProc.run_engine_core is not the configuration entrypoint"
    if not isinstance(inspect.getattr_static(owner, "run_engine_core"), staticmethod):
        return "EngineCoreProc.run_engine_core must remain a staticmethod"
    return None


def verify_npu_kv() -> str | None:
    error = _marker(
        "ascend_attention_utils",
        "requires_contiguous_pa_kv_cache",
        "_vllm_gr_contiguous_prefix_kv_policy",
    )
    if error:
        return error
    first = resolve_symbol("ascend_attention_utils").requires_contiguous_pa_kv_cache
    second = resolve_symbol("ascend_runner_module").requires_contiguous_pa_kv_cache
    if first is not second:
        return "Contiguous KV policy bindings differ"
    return None


def verify_attention_registration() -> str | None:
    registry = resolve_symbol("attention_registry")
    path = registry.AttentionBackendEnum.CUSTOM.get_path()
    if path != "vllm_gr.v1.attention.backends.beam_attn.BeamAttentionBackend":
        return "CUSTOM attention registry path does not select BeamAttentionBackend"
    return None


def verify_scoring_aliases() -> str | None:
    for symbol in ("v2_logprob", "v2_sampler", "v2_rejection_sampler"):
        error = _marker(symbol, "compute_topk_scores", "_vllm_gr_clamped_topk")
        if error:
            return error
    return None


def verify_runner() -> str | None:
    from vllm_gr.patch import _validate_model_runner_patch
    from vllm_gr.patch_report import runtime_platform
    from vllm_gr.v1.worker.model_runner_common import (
        _CONSTRAINT_LOAD_PATCH_MARKER,
        _CONSTRAINT_SHUTDOWN_PATCH_MARKER,
    )
    from vllm_gr.v1.worker.model_runner_patch import discover_active_worker_runners

    error: str | None = _validate_model_runner_patch()
    if error:
        return error
    for name, runner in discover_active_worker_runners():
        for method, marker in (
            ("load_model", _CONSTRAINT_LOAD_PATCH_MARKER),
            ("shutdown", _CONSTRAINT_SHUTDOWN_PATCH_MARKER),
        ):
            if not getattr(getattr(runner, method, None), marker, False):
                return f"{name}.{method} lifecycle hook is not installed"
    error = _marker("WorkerProc", "enqueue_output", "_gr_async_output_guard")
    if error:
        return error
    if runtime_platform() == "cuda":
        return _marker("StagedWriteTensor", "stage_write", "_gr_numpy_tokens") or _verify_utilities(
            "GPUWorker"
        )
    return None


def _verify_utilities(symbol: str) -> str | None:
    from vllm_gr.constraints.worker_inject import set_gr_constraint_resources
    from vllm_gr.v1.worker.model_runner_common import (
        freeze_gc_after_warmup,
        get_beam_graph_runtime_stats,
    )

    owner = resolve_symbol(symbol)
    for name, implementation in (
        ("set_gr_constraint_resources", set_gr_constraint_resources),
        ("get_gr_beam_graph_runtime_stats", get_beam_graph_runtime_stats),
        ("gr_freeze_gc_after_warmup", freeze_gc_after_warmup),
    ):
        if getattr(owner, name, None) is not implementation:
            return f"{symbol}.{name} is not installed"
    return None


def verify_npu_utilities() -> str | None:
    return _verify_utilities("NPUWorker")


def verify_weight_updates() -> str | None:
    from vllm_gr.ops.npu_stage_logits import _UPDATE_PATCH

    for name in ("start_weight_update", "update_weights", "finish_weight_update"):
        error = _marker("NPUWorker", name, _UPDATE_PATCH)
        if error:
            return error
    return None
