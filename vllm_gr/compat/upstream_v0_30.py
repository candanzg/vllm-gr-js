# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Upstream symbol locations for the supported vLLM runtime.

Runtime patches reach into vLLM internals. Those targets live here so a runtime
upgrade touches one file instead of every validator in :mod:`vllm_gr.patch`.
Resolving is deliberately uncached: installers replace module attributes, and a
cached object would keep referring to the pre-install function.
"""

from __future__ import annotations

import importlib
from typing import Any, NamedTuple

UPSTREAM_VERSION = "0.30.0"


class Symbol(NamedTuple):
    module: str
    attribute: str = ""

    def __str__(self) -> str:
        return f"{self.module}.{self.attribute}" if self.attribute else self.module


SYMBOLS: dict[str, Symbol] = {
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
    "SyncMPClient": Symbol("vllm.v1.engine.core_client", "SyncMPClient"),
    "InprocClient": Symbol("vllm.v1.engine.core_client", "InprocClient"),
    "AsyncLLM": Symbol("vllm.v1.engine.async_llm", "AsyncLLM"),
    "LLMEngine": Symbol("vllm.v1.engine.llm_engine", "LLMEngine"),
    "EngineCore": Symbol("vllm.v1.engine.core", "EngineCore"),
    "EngineCoreProc": Symbol("vllm.v1.engine.core", "EngineCoreProc"),
    "Scheduler": Symbol("vllm.v1.core.sched.scheduler", "Scheduler"),
    "kv_cache_utils": Symbol("vllm.v1.core.kv_cache_utils"),
}


def resolve(name: str) -> Any:
    """Return the current upstream object ``name`` refers to."""
    symbol = SYMBOLS[name]
    value: Any = importlib.import_module(symbol.module)
    for part in symbol.attribute.split(".") if symbol.attribute else ():
        value = getattr(value, part)
    return value
