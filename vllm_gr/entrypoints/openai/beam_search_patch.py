# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.serving import OpenAIServing

from vllm_gr.entrypoints.openai.protocol import to_beam_search_params
from vllm_gr.entrypoints.openai.serving_engine import beam_search


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
    # Always delegate to the current wrapper/original. This keeps any
    # vllm-ascend wrapper that may have been installed after a previous GR
    # patch application instead of pinning an older saved original.
    run_engine_core._original_run_engine_core = current_run_engine_core
    EngineCoreProc.run_engine_core = run_engine_core
