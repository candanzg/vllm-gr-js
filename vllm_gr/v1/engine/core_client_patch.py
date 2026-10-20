# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install atomic GR session transport and frontend lifecycle adapters."""

from __future__ import annotations

from vllm_gr.v1.engine.async_llm import make_patched_process_outputs
from vllm_gr.v1.engine.core_client import submit_gr_session_async


def sync_mp_submit_gr_session(self, batch_request):
    from vllm.v1.engine import EngineCoreRequestType

    requests = tuple(item.request for item in batch_request.items)
    if not requests:
        raise ValueError("GR batch session must not be empty")
    for request in requests:
        request.client_index = getattr(self, "client_index", 0)
        if hasattr(self, "current_wave"):
            request.current_wave = self.current_wave
    if getattr(self, "is_dp", False):
        self.engines_running = True
    self._send_input(EngineCoreRequestType.SUBMIT_GR_SESSION, batch_request)


def inproc_submit_gr_session(self, batch_request):
    self.engine_core._handle_submit_gr_session(batch_request)


def apply_request_type_patches() -> None:
    from vllm_gr.v1.engine.wire import register_engine_core_request_types

    register_engine_core_request_types()


def apply_engine_client_patches() -> None:
    """Install V1 submission on every supported EngineCore transport."""
    from vllm.v1.engine.core_client import AsyncMPClient, InprocClient, SyncMPClient

    AsyncMPClient.submit_gr_session_async = submit_gr_session_async
    SyncMPClient.submit_gr_session = sync_mp_submit_gr_session
    InprocClient.submit_gr_session = inproc_submit_gr_session


def apply_async_llm_patches() -> None:
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.engine.output_processor import OutputProcessor

    from vllm_gr.v1.engine.gr_session_client import install_gr_frontend_methods

    OutputProcessor.process_outputs = make_patched_process_outputs(OutputProcessor.process_outputs)
    install_gr_frontend_methods(LLMEngine, AsyncLLM)


def apply_gr_session_patches() -> None:
    """Install GR transport in the frontend, including in-process engines."""
    apply_request_type_patches()
    apply_engine_client_patches()
    apply_async_llm_patches()

    from vllm_gr.v1.engine.engine_core_patch import apply_engine_core_child_patches

    apply_engine_core_child_patches()
