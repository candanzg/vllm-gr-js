# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Installation ownership; importing this module never imports device runtimes."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Installation:
    patch_id: str
    name: str
    installer: str
    validator: str
    symbols: tuple[str, ...]
    phases: tuple[str, ...]
    condition: str = "always"
    kind: str = "mutation"
    dependencies: tuple[str, ...] = ()

    def contract(self) -> dict[str, object]:
        return {
            **{
                key: list(value) if isinstance(value, tuple) else value
                for key, value in asdict(self).items()
            },
            "targets": list(BINDINGS[self.name]),
        }


INSTALLATIONS = (
    Installation(
        patch_id="GR-PATCH-000",
        name="runtime_activation",
        installer="vllm_gr.plugin.initialize_runtime",
        validator="",
        symbols=(),
        phases=("runtime",),
        kind="lifecycle",
    ),
    Installation(
        patch_id="GR-PATCH-001",
        name="attention_selector",
        installer="vllm_gr.v1.ascend_compat.apply_ascend_attention_backend_patch",
        validator="vllm_gr.patch_verification.verify_selector",
        symbols=("selector",),
        phases=("runtime", "worker_init"),
    ),
    Installation(
        patch_id="GR-PATCH-002",
        name="npu_worker_init",
        installer="vllm_gr.v1.ascend_compat.patch_npu_worker_init_device",
        validator="vllm_gr.patch_verification.verify_npu_worker",
        symbols=("NPUWorker",),
        phases=("runtime", "worker_init"),
        condition="npu",
    ),
    Installation(
        patch_id="GR-PATCH-002",
        name="npu_worker_utilities",
        installer="vllm_gr.v1.worker.model_runner_patch.install_worker_utility_hooks",
        validator="vllm_gr.patch_verification.verify_npu_utilities",
        symbols=("NPUWorker",),
        phases=("worker_init",),
        condition="npu",
    ),
    Installation(
        patch_id="GR-PATCH-002",
        name="npu_weight_update_hooks",
        installer="vllm_gr.ops.npu_stage_logits.install_npu_stage_weight_update_hooks",
        validator="vllm_gr.patch_verification.verify_weight_updates",
        symbols=("NPUWorker",),
        phases=("worker_init",),
        condition="npu_weight_update_api",
    ),
    Installation(
        patch_id="GR-PATCH-003",
        name="flat_logprobs",
        installer="vllm_gr.logprobs_patch.patch_flat_logprobs",
        validator="vllm_gr.patch._validate_flat_logprobs_patch",
        symbols=("FlatLogprobs",),
        phases=("runtime", "frontend", "worker_init"),
    ),
    Installation(
        patch_id="GR-PATCH-004",
        name="openai_beam_search",
        installer="vllm_gr.entrypoints.openai.beam_search_patch.patch_beam_search",
        validator="vllm_gr.patch._validate_openai_beam_search_patch",
        symbols=(
            "GenerateBaseServing",
            "BaseServing",
            "OutputProcessor",
            "OpenAIServingCompletion",
        ),
        phases=("runtime", "frontend", "worker_init"),
    ),
    Installation(
        patch_id="GR-PATCH-004",
        name="chat_sampling_conversion",
        installer="vllm_gr.entrypoints.openai.beam_search_patch.patch_sampling",
        validator="vllm_gr.patch._validate_chat_sampling_patch",
        symbols=("ChatCompletionRequest", "CompletionRequest"),
        phases=("runtime", "frontend", "worker_init"),
    ),
    Installation(
        patch_id="GR-PATCH-005",
        name="engine_args",
        installer="vllm_gr.engine.arg_utils_gr.patch_add_cli_args",
        validator="vllm_gr.patch._validate_engine_args_patch",
        symbols=("EngineArgs",),
        phases=("runtime", "frontend", "worker_init"),
    ),
    Installation(
        patch_id="GR-PATCH-007",
        name="engine_clients",
        installer="vllm_gr.v1.engine.core_client_patch.apply_engine_client_patches",
        validator="vllm_gr.patch._validate_engine_clients_patch",
        symbols=("AsyncMPClient", "SyncMPClient", "InprocClient"),
        phases=("runtime", "frontend", "worker_init"),
    ),
    Installation(
        patch_id="GR-PATCH-007",
        name="frontend_lifecycle",
        installer="vllm_gr.v1.engine.core_client_patch.apply_async_llm_patches",
        validator="vllm_gr.patch._validate_async_llm_patch",
        symbols=("AsyncLLM", "LLMEngine", "OutputProcessor"),
        phases=("runtime", "frontend", "worker_init"),
    ),
    Installation(
        patch_id="GR-PATCH-008",
        name="request_types",
        installer="vllm_gr.v1.engine.wire.register_engine_core_request_types",
        validator="vllm_gr.patch._validate_request_types_patch",
        symbols=("EngineCoreRequestType",),
        phases=("runtime", "frontend", "worker_init", "engine_core"),
    ),
    Installation(
        patch_id="GR-PATCH-009",
        name="engine_core_requests",
        installer="vllm_gr.v1.engine.engine_core_patch.apply_engine_core_request_patches",
        validator="vllm_gr.patch._validate_engine_core_request_patch",
        symbols=("EngineCore", "EngineCoreProc", "DPEngineCoreProc", "Request", "NewRequestData"),
        phases=("runtime", "frontend", "worker_init", "engine_core"),
        dependencies=("request_types",),
    ),
    Installation(
        patch_id="GR-PATCH-009",
        name="engine_core_entrypoint",
        installer="vllm_gr.entrypoints.openai.beam_search_patch.patch_engine_core_process_entrypoint",
        validator="vllm_gr.patch._validate_engine_core_entrypoint_patch",
        symbols=("EngineCoreProc",),
        phases=("runtime", "frontend", "worker_init"),
    ),
    Installation(
        patch_id="GR-PATCH-010",
        name="scheduler",
        installer="vllm_gr.v1.engine.engine_core_patch.apply_scheduler_patch",
        validator="vllm_gr.patch._validate_scheduler_patch",
        symbols=("Scheduler",),
        phases=("runtime", "frontend", "worker_init", "engine_core"),
    ),
    Installation(
        patch_id="GR-PATCH-011",
        name="cuda_model_runner",
        installer="vllm_gr.v1.worker.model_runner_patch.apply_worker_patches",
        validator="vllm_gr.patch_verification.verify_runner",
        symbols=(
            "GPUModelRunner",
            "GPUModelRunnerV2",
            "WorkerProc",
            "GPUWorker",
            "StagedWriteTensor",
            "buffer_utils",
        ),
        phases=("runtime", "frontend", "worker_init", "engine_core"),
        condition="cuda",
    ),
    Installation(
        patch_id="GR-PATCH-012",
        name="npu_model_runner",
        installer="vllm_gr.v1.worker.model_runner_patch.apply_worker_patches",
        validator="vllm_gr.patch_verification.verify_runner",
        symbols=("NPUModelRunner", "WorkerProc"),
        phases=("runtime", "frontend", "worker_init", "engine_core"),
        condition="npu",
    ),
    Installation(
        patch_id="GR-PATCH-013",
        name="attention_registration",
        installer="vllm_gr.v1.attention.backends.beam_attn.register_backend",
        validator="vllm_gr.patch_verification.verify_attention_registration",
        symbols=("attention_registry",),
        phases=("registration",),
        kind="registration",
    ),
    Installation(
        patch_id="GR-PATCH-015",
        name="sliced_logits",
        installer="vllm_gr.ops.sliced_logits_patch.apply_sliced_logits_patch",
        validator="vllm_gr.patch_verification.verify_sliced_logits",
        symbols=("LogitsProcessor", "Sampler"),
        phases=("runtime", "worker_init"),
    ),
    Installation(
        patch_id="GR-PATCH-015",
        name="sliced_scoring_aliases",
        installer="vllm_gr.ops.sliced_logits_patch.apply_sliced_logits_patch",
        validator="vllm_gr.patch_verification.verify_scoring_aliases",
        symbols=("v2_logprob", "v2_sampler", "v2_rejection_sampler"),
        phases=("runtime", "worker_init"),
        condition="cuda",
    ),
    Installation(
        patch_id="GR-PATCH-016",
        name="npu_spawn",
        installer="vllm_gr.npu_spawn_patch.apply_npu_spawn_patch",
        validator="vllm_gr.patch_verification.verify_spawn",
        symbols=("system_utils", "multiproc_executor"),
        phases=("runtime", "worker_init"),
    ),
    Installation(
        patch_id="GR-PATCH-018",
        name="penalties_warmup",
        installer="vllm_gr.v1.ascend_compat.apply_penalties_warmup_oom_guard",
        validator="vllm_gr.patch_verification.verify_penalties",
        symbols=("warmup",),
        phases=("worker_init",),
        condition="npu",
    ),
    Installation(
        patch_id="GR-PATCH-019",
        name="ascend_config_filter",
        installer="vllm_gr.v1.ascend_compat._install_init_ascend_config_sanitizer",
        validator="vllm_gr.patch_verification.verify_config_filter",
        symbols=("ascend_config",),
        phases=("runtime", "worker_init", "configuration"),
        condition="npu",
    ),
    Installation(
        patch_id="GR-PATCH-019",
        name="ascend_core_entrypoint",
        installer="vllm_gr.v1.ascend_compat.apply_engine_core_config_sanitizer",
        validator="vllm_gr.patch_verification.verify_config_entrypoint",
        symbols=("EngineCoreProc",),
        phases=("configuration",),
        condition="npu",
    ),
    Installation(
        patch_id="GR-PATCH-020",
        name="pangu_tied_embeddings",
        installer="vllm_gr.pangu_compat.apply_pangu_tied_embeddings_patch",
        validator="vllm_gr.patch_verification.verify_pangu",
        symbols=("ModelConfig",),
        phases=("runtime", "worker_init"),
    ),
    Installation(
        patch_id="GR-PATCH-021",
        name="npu_kv_layout",
        installer="vllm_gr.v1.npu_kv_layout.apply_npu_contiguous_kv_cache_patch",
        validator="vllm_gr.patch_verification.verify_npu_kv",
        symbols=("ascend_attention_utils", "ascend_runner_module"),
        phases=("worker_init",),
        condition="supported_npu_kv",
    ),
    Installation(
        patch_id="GR-PATCH-022",
        name="kv_hash_memo",
        installer="vllm_gr.v1.engine.engine_core_patch.apply_kv_hash_memo_patch",
        validator="vllm_gr.patch._validate_kv_hash_memo_patch",
        symbols=("kv_cache_utils", "engine_core_module"),
        phases=("runtime", "frontend", "worker_init", "engine_core"),
    ),
)
BY_NAME = {item.name: item for item in INSTALLATIONS}
RETIRED_PATCH_IDS = frozenset({"GR-PATCH-006", "GR-PATCH-014", "GR-PATCH-017"})


def for_installer(path: str) -> tuple[Installation, ...]:
    return tuple(item for item in INSTALLATIONS if item.installer == path)


# Concrete bindings include consumer aliases as well as defining attributes.
BINDINGS: dict[str, tuple[str, ...]] = {
    "runtime_activation": (),
    "attention_selector": ("selector._cached_get_attn_backend",),
    "npu_worker_init": ("NPUWorker.init_device",),
    "flat_logprobs": ("FlatLogprobs.append_fast",),
    "openai_beam_search": (
        "GenerateBaseServing.beam_search",
        "GenerateBaseServing.beam_search_v1",
        "GenerateBaseServing._raise_if_error",
        "BaseServing._base_request_id",
        "OutputProcessor.add_request",
        "OpenAIServingCompletion._create_completion",
    ),
    "chat_sampling_conversion": (
        "ChatCompletionRequest.to_beam_search_params",
        "CompletionRequest.to_beam_search_params",
        "CompletionRequest.model_fields",
    ),
    "engine_args": (
        "EngineArgs.__init__",
        "EngineArgs.add_cli_args",
        "EngineArgs.from_cli_args",
        "EngineArgs.create_engine_config",
    ),
    "engine_clients": (
        "AsyncMPClient.submit_gr_session_async",
        "SyncMPClient.submit_gr_session",
        "InprocClient.submit_gr_session",
        "InprocClient.get_output",
        "InprocClient.add_request",
        "InprocClient.abort_requests",
        "InprocClient.shutdown",
    ),
    "frontend_lifecycle": (
        "OutputProcessor.process_outputs",
        "LLMEngine.add_request",
        "AsyncLLM._add_request",
    )
    + tuple(
        f"{owner}.{method}_gr_session"
        for owner in ("LLMEngine", "AsyncLLM")
        for method in ("submit", "wait", "abort", "release")
    ),
    "request_types": ("EngineCoreRequestType.SUBMIT_GR_SESSION",),
    "engine_core_requests": (
        "EngineCore.__init__",
        "EngineCore.is_gr_session_retired",
        "EngineCore._handle_submit_gr_session",
        "EngineCoreProc._handle_submit_gr_session",
        "EngineCoreProc._handle_client_request",
        "DPEngineCoreProc._handle_client_request",
        "EngineCoreProc.process_input_sockets",
        "EngineCoreProc._process_engine_step",
        "Request.from_engine_core_request",
        "NewRequestData.from_request",
    ),
    "engine_core_entrypoint": ("EngineCoreProc.run_engine_core",),
    "scheduler": (
        "Scheduler.schedule",
        "Scheduler.add_request",
        "Scheduler.finish_requests",
        "Scheduler.has_requests",
        "Scheduler.update_from_output",
        "Scheduler.reset_prefix_cache",
    ),
    "cuda_model_runner": tuple(
        f"{owner}.{method}"
        for owner in ("GPUModelRunner", "GPUModelRunnerV2")
        for method in ("execute_model", "load_model", "shutdown", "capture_model")
    ),
    "npu_model_runner": tuple(
        f"NPUModelRunner.{method}"
        for method in ("execute_model", "load_model", "shutdown", "capture_model")
    ),
    "attention_registration": ("attention_registry.AttentionBackendEnum",),
    "sliced_logits": (
        "LogitsProcessor.__init__",
        "LogitsProcessor._get_logits",
        "Sampler.gather_logprobs",
    ),
    "npu_spawn": ("system_utils._maybe_force_spawn", "multiproc_executor._maybe_force_spawn"),
    "penalties_warmup": ("warmup.penalties_triton_warmup",),
    "ascend_config_filter": ("ascend_config.init_ascend_config",),
    "ascend_core_entrypoint": ("EngineCoreProc.run_engine_core",),
    "pangu_tied_embeddings": ("ModelConfig.maybe_untie_word_embeddings",),
    "npu_kv_layout": (
        "ascend_attention_utils.requires_contiguous_pa_kv_cache",
        "ascend_runner_module.requires_contiguous_pa_kv_cache",
    ),
    "kv_hash_memo": (
        "kv_cache_utils.hash_block_tokens",
        "kv_cache_utils.get_request_block_hasher",
        "engine_core_module.get_request_block_hasher",
    ),
}

BINDINGS["scheduler"] += tuple(
    f"Scheduler.{method}"
    for method in (
        "add_gr_session",
        "can_issue_gr_stage",
        "attach_gr_stage_metadata",
        "update_gr_from_output",
        "finish_gr_session",
        "abort_gr_session",
        "release_gr_session",
    )
)
BINDINGS["cuda_model_runner"] += (
    tuple(
        f"GPUModelRunner.{method}"
        for method in (
            "_prepare_inputs",
            "_bookkeeping_sync",
            "_determine_batch_execution_and_padding",
        )
    )
    + tuple(
        f"GPUModelRunnerV2.{method}"
        for method in (
            "prepare_inputs",
            "sample_tokens",
            "add_requests",
            "initialize_kv_cache",
            "profile_cudagraph_memory",
        )
    )
    + (
        "WorkerProc.enqueue_output",
        "GPUWorker.set_gr_constraint_resources",
        "GPUWorker.get_gr_beam_graph_runtime_stats",
        "GPUWorker.gr_freeze_gc_after_warmup",
        "StagedWriteTensor.stage_write",
        "buffer_utils.async_tensor_h2d",
    )
)
BINDINGS["npu_model_runner"] += (
    "WorkerProc.enqueue_output",
    "NPUModelRunner._prepare_inputs",
    "NPUModelRunner._bookkeeping_sync",
)
BINDINGS["sliced_scoring_aliases"] = tuple(
    f"{module}.compute_topk_scores"
    for module in ("v2_logprob", "v2_sampler", "v2_rejection_sampler")
)

BINDINGS["npu_worker_utilities"] = (
    "NPUWorker.set_gr_constraint_resources",
    "NPUWorker.get_gr_beam_graph_runtime_stats",
    "NPUWorker.gr_freeze_gc_after_warmup",
)
BINDINGS["npu_weight_update_hooks"] = (
    "NPUWorker.start_weight_update",
    "NPUWorker.update_weights",
    "NPUWorker.finish_weight_update",
)
