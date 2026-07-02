# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_gr.entrypoints.openai.protocol import resolve_openai_symbol, to_beam_search_params
from vllm_gr.entrypoints.openai.serving_engine import beam_search

OpenAIServing = resolve_openai_symbol("OpenAIServing")
ChatCompletionRequest = resolve_openai_symbol("ChatCompletionRequest")


def patch_beam_search():
    OpenAIServing.beam_search = beam_search


def patch_sampling():
    ChatCompletionRequest.to_beam_search_params = to_beam_search_params


def patch_batch_and_fork():
    """Wire ADD_BATCH + BEAM_FORK into the engine pipeline."""
    from vllm.v1.engine.core import EngineCoreProc

    from vllm_gr.v1.engine.core_client_patch import apply_batch_fork_patches
    from vllm_gr.v1.engine.engine_core_patch import run_engine_core

    apply_batch_fork_patches()
    current_run_engine_core = EngineCoreProc.run_engine_core
    if current_run_engine_core is run_engine_core:
        return

    # Always delegate to the implementation that is active at patch time.
    # This preserves wrappers installed by other runtimes, notably
    # vllm-ascend's balance-scheduler patch on NPU.
    run_engine_core._original_run_engine_core = current_run_engine_core
    run_engine_core._delegate_run_engine_core_module = getattr(
        current_run_engine_core,
        "__module__",
        type(current_run_engine_core).__module__,
    )
    run_engine_core._delegate_run_engine_core_name = getattr(
        current_run_engine_core,
        "__qualname__",
        getattr(
            current_run_engine_core,
            "__name__",
            type(current_run_engine_core).__name__,
        ),
    )
    run_engine_core._vllm_gr_wrapper = True
    EngineCoreProc.run_engine_core = run_engine_core
