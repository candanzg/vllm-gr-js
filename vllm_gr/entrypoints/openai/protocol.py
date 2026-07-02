# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py

import importlib
import pkgutil

from vllm_gr.sampling_params import BeamSearchParams

_OPENAI_CANDIDATE_MODULES = (
    "vllm.entrypoints.openai.protocol",
    "vllm.entrypoints.openai.engine.protocol",
    "vllm.entrypoints.openai.engine.serving",
    "vllm.entrypoints.openai.models.protocol",
    "vllm.entrypoints.openai.models.serving",
    "vllm.entrypoints.openai.chat_completion.protocol",
    "vllm.entrypoints.openai.chat_completion.serving",
    "vllm.entrypoints.openai.completion.protocol",
    "vllm.entrypoints.openai.serving_engine",
    "vllm.entrypoints.openai.serving_models",
    "vllm.entrypoints.openai.schemas",
    "vllm.entrypoints.openai.types",
    "vllm.entrypoints.openai.api_types",
    "vllm.entrypoints.openai.api_server",
)
_OPENAI_DISCOVERY_TOKENS = ("protocol", "schema", "type", "api", "serving")


def resolve_openai_symbol(symbol_name: str):
    for module_name in _OPENAI_CANDIDATE_MODULES:
        try:
            module = importlib.import_module(module_name)
        except (ModuleNotFoundError, ImportError):
            continue
        if hasattr(module, symbol_name):
            return getattr(module, symbol_name)

    openai_pkg = importlib.import_module("vllm.entrypoints.openai")
    for module_info in pkgutil.walk_packages(openai_pkg.__path__, openai_pkg.__name__ + "."):
        name_parts = module_info.name.lower().split(".")
        leaf_name = name_parts[-1]
        parent_name = name_parts[-2] if len(name_parts) > 1 else ""
        if not any(token in leaf_name for token in _OPENAI_DISCOVERY_TOKENS):
            if not any(token in parent_name for token in _OPENAI_DISCOVERY_TOKENS):
                continue
        try:
            module = importlib.import_module(module_info.name)
        except Exception:
            continue
        if hasattr(module, symbol_name):
            return getattr(module, symbol_name)

    raise ModuleNotFoundError(
        f"Unable to resolve {symbol_name!r} from installed vLLM OpenAI modules."
    )


def to_beam_search_params(self, max_tokens: int, default_sampling_params: dict) -> BeamSearchParams:
    extra_args = self.vllm_xargs or {}
    n = self.n if self.n is not None else 1
    if (temperature := self.temperature) is None:
        temperature = default_sampling_params.get(
            "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"]
        )
    return BeamSearchParams(
        beam_width=n,
        max_tokens=max_tokens,
        ignore_eos=self.ignore_eos,
        temperature=temperature,
        length_penalty=self.length_penalty,
        include_stop_str_in_output=self.include_stop_str_in_output,
        begin_token=extra_args.get("begin_token"),
        end_token=extra_args.get("end_token"),
    )
