# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Symbol locations and structural contracts for the supported runtime."""

from __future__ import annotations

import importlib
import inspect
import sys
from dataclasses import dataclass
from typing import Any


class RuntimeContractError(RuntimeError):
    """A selected runtime target does not satisfy its installation contract."""


@dataclass(frozen=True)
class Symbol:
    module: str
    attribute: str = ""

    @property
    def path(self) -> str:
        return self.module + ("." + self.attribute if self.attribute else "")


SYMBOLS: dict[str, Symbol] = {
    "WorkerProc": Symbol("vllm.v1.executor.multiproc_executor", "WorkerProc"),
    "GPUWorker": Symbol("vllm.v1.worker.gpu_worker", "Worker"),
    "StagedWriteTensor": Symbol("vllm.v1.worker.gpu.buffer_utils", "StagedWriteTensor"),
    "buffer_utils": Symbol("vllm.v1.worker.gpu.buffer_utils"),
    "v2_logprob": Symbol("vllm.v1.worker.gpu.sample.logprob"),
    "v2_sampler": Symbol("vllm.v1.worker.gpu.sample.sampler"),
    "v2_rejection_sampler": Symbol("vllm.v1.worker.gpu.spec_decode.rejection_sampler"),
    "engine_core_module": Symbol("vllm.v1.engine.core"),
    "FlatLogprobs": Symbol("vllm.logprobs", "FlatLogprobs"),
    "GenerateBaseServing": Symbol("vllm.entrypoints.generate.base.serving", "GenerateBaseServing"),
    "BaseServing": Symbol("vllm.entrypoints.serve.engine.serving", "BaseServing"),
    "OutputProcessor": Symbol("vllm.v1.engine.output_processor", "OutputProcessor"),
    "ChatCompletionRequest": Symbol(
        "vllm.entrypoints.openai.chat_completion.protocol", "ChatCompletionRequest"
    ),
    "EngineArgs": Symbol("vllm.engine.arg_utils", "EngineArgs"),
    "EngineCoreRequestType": Symbol("vllm.v1.engine", "EngineCoreRequestType"),
    "AsyncMPClient": Symbol("vllm.v1.engine.core_client", "AsyncMPClient"),
    "InprocClient": Symbol("vllm.v1.engine.core_client", "InprocClient"),
    "SyncMPClient": Symbol("vllm.v1.engine.core_client", "SyncMPClient"),
    "AsyncLLM": Symbol("vllm.v1.engine.async_llm", "AsyncLLM"),
    "LLMEngine": Symbol("vllm.v1.engine.llm_engine", "LLMEngine"),
    "EngineCoreProc": Symbol("vllm.v1.engine.core", "EngineCoreProc"),
    "EngineCore": Symbol("vllm.v1.engine.core", "EngineCore"),
    "Scheduler": Symbol("vllm.v1.core.sched.scheduler", "Scheduler"),
    "kv_cache_utils": Symbol("vllm.v1.core.kv_cache_utils", ""),
    "CompletionRequest": Symbol("vllm.entrypoints.openai.completion.protocol", "CompletionRequest"),
    "OpenAIServingCompletion": Symbol(
        "vllm.entrypoints.openai.completion.serving", "OpenAIServingCompletion"
    ),
    "selector": Symbol("vllm.v1.attention.selector", ""),
    "ModelConfig": Symbol("vllm.config", "ModelConfig"),
    "system_utils": Symbol("vllm.utils.system_utils", ""),
    "multiproc_executor": Symbol("vllm.v1.executor.multiproc_executor", ""),
    "LogitsProcessor": Symbol("vllm.model_executor.layers.logits_processor", "LogitsProcessor"),
    "Sampler": Symbol("vllm.v1.sample.sampler", "Sampler"),
    "GPUModelRunner": Symbol("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
    "GPUModelRunnerV2": Symbol("vllm.v1.worker.gpu.model_runner", "GPUModelRunner"),
    "NPUModelRunner": Symbol("vllm_ascend.worker.model_runner_v1", "NPUModelRunner"),
    "NPUWorker": Symbol("vllm_ascend.worker.worker", "NPUWorker"),
    "warmup": Symbol("vllm_ascend.model_executor.warmup.kernel_warmup", ""),
    "ascend_config": Symbol("vllm_ascend.ascend_config", ""),
    "ascend_attention_utils": Symbol("vllm_ascend.attention.utils", ""),
    "ascend_runner_module": Symbol("vllm_ascend.worker.model_runner_v1", ""),
    "attention_registry": Symbol("vllm.v1.attention.backends.registry", ""),
    "Request": Symbol("vllm.v1.request", "Request"),
    "NewRequestData": Symbol("vllm.v1.core.sched.output", "NewRequestData"),
    "DPEngineCoreProc": Symbol("vllm.v1.engine.core", "DPEngineCoreProc"),
}


def resolve_symbol(name: str, *, loaded_only: bool = False) -> Any:
    """Resolve on demand; never cache objects that an installer can replace."""
    spec = SYMBOLS[name]
    module = sys.modules.get(spec.module) if loaded_only else importlib.import_module(spec.module)
    if module is None:
        return None
    value: Any = module
    try:
        for part in spec.attribute.split(".") if spec.attribute else ():
            value = getattr(value, part)
    except AttributeError as exc:
        if loaded_only:
            return None
        raise RuntimeContractError(f"Missing runtime symbol {spec.path}") from exc
    return value


def check_method(
    owner: Any,
    name: str,
    *,
    descriptor: type | None = None,
    parameters: tuple[str, ...] = (),
) -> None:
    """Check a callable without invoking it or binding a descriptor accidentally."""
    raw = inspect.getattr_static(owner, name, None)
    if descriptor is not None and not isinstance(raw, descriptor):
        raise RuntimeContractError(f"{owner.__name__}.{name} must be a {descriptor.__name__}")
    function = getattr(owner, name, None)
    if not callable(function):
        raise RuntimeContractError(f"{owner.__name__}.{name} must be callable")
    signature = inspect.signature(function)
    if not set(parameters) <= signature.parameters.keys():
        raise RuntimeContractError(f"{owner.__name__}.{name} requires parameters {parameters}")


# Only native methods are checked before installation; GR-added methods belong
# to postconditions. Keep these probes free of device execution.
STRUCTURES: dict[str, tuple[tuple[str, str, type | None, tuple[str, ...]], ...]] = {
    "flat_logprobs": (("FlatLogprobs", "append_fast", None, ("self",)),),
    "attention_selector": (
        (
            "selector",
            "_cached_get_attn_backend",
            None,
            ("backend", "attn_selector_config", "num_heads"),
        ),
    ),
    "npu_worker_init": (("NPUWorker", "init_device", None, ("self",)),),
    "pangu_tied_embeddings": (("ModelConfig", "maybe_untie_word_embeddings", None, ("self",)),),
    "npu_spawn": (("system_utils", "_maybe_force_spawn", None, ()),),
    "engine_args": (
        ("EngineArgs", "add_cli_args", staticmethod, ("parser",)),
        ("EngineArgs", "from_cli_args", classmethod, ("args",)),
        ("EngineArgs", "create_engine_config", None, ("self",)),
    ),
    "engine_core_entrypoint": (
        ("EngineCoreProc", "run_engine_core", staticmethod, ("dp_rank", "local_dp_rank")),
    ),
    "engine_core_requests": (
        ("EngineCoreProc", "process_input_sockets", None, ("self",)),
        ("EngineCoreProc", "_handle_client_request", None, ("self",)),
        ("DPEngineCoreProc", "_handle_client_request", None, ("self",)),
    ),
    "scheduler": (("Scheduler", "schedule", None, ("self", "throttle_prefills")),),
    "sliced_logits": (
        ("LogitsProcessor", "_get_logits", None, ("self", "hidden_states")),
        ("Sampler", "gather_logprobs", staticmethod, ("logprobs", "num_logprobs")),
    ),
    "penalties_warmup": (("warmup", "penalties_triton_warmup", None, ()),),
    "ascend_config_filter": (("ascend_config", "init_ascend_config", None, ()),),
    "kv_hash_memo": (
        ("kv_cache_utils", "hash_block_tokens", None, ("parent_block_hash",)),
        ("kv_cache_utils", "get_request_block_hasher", None, ("caching_hash_fn",)),
    ),
}


def check_structure(installation: str) -> None:
    """Reject changed native descriptors/signatures before replacing them."""
    for symbol, method, descriptor, parameters in STRUCTURES.get(installation, ()):
        check_method(resolve_symbol(symbol), method, descriptor=descriptor, parameters=parameters)
