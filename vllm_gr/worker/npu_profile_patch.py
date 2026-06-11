# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Optional NPU runtime profiling hooks.

This patch is intentionally independent from GRNPUModelRunner.  Online
OpenAI-compatible serving for Qwen-style generative recommendation models can
run through vLLM-Ascend's standard NPUModelRunner while still using vLLM-GR's
beam-search serving patches.  In that path GRFeatureRunnerMixin is not present,
so the profile hook must live on the underlying NPUModelRunner.
"""

from __future__ import annotations

import os
from functools import wraps
from typing import Any

from vllm_gr.logger import init_logger

logger = init_logger(__name__)


def _is_profile_enabled() -> bool:
    return os.getenv("VLLM_GR_NPU_PROFILE", "").lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
        "debug",
    }


def _extract_total_scheduled_tokens(scheduler_output: Any) -> int | None:
    if scheduler_output is None:
        return None

    total = getattr(scheduler_output, "total_num_scheduled_tokens", None)
    if total is not None:
        try:
            return int(total)
        except (TypeError, ValueError):
            return None

    scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    if isinstance(scheduled, dict):
        try:
            return int(sum(scheduled.values()))
        except (TypeError, ValueError):
            return None

    return None


def _extract_num_reqs(runner: Any) -> int:
    input_batch = getattr(runner, "input_batch", None)
    num_reqs = getattr(input_batch, "num_reqs", None)
    if num_reqs is not None:
        try:
            return int(num_reqs)
        except (TypeError, ValueError):
            return 0

    req_ids = getattr(input_batch, "req_ids", None)
    if req_ids is not None:
        try:
            return len(req_ids)
        except TypeError:
            return 0

    return 0


def _extract_beam_group_stats(scheduler_output: Any) -> tuple[int, int]:
    beam_groups = getattr(scheduler_output, "beam_prefix_groups", None)
    if not isinstance(beam_groups, list):
        return 0, 0

    beam_group_reqs = 0
    for _, req_ids in beam_groups:
        try:
            beam_group_reqs += len(req_ids)
        except TypeError:
            pass
    return len(beam_groups), beam_group_reqs


def _maybe_log_prepare_inputs_profile(runner: Any, scheduler_output: Any, result: Any) -> None:
    if not _is_profile_enabled():
        return

    # GRNPUModelRunner already logs through GRFeatureRunnerMixin after the
    # parent _prepare_inputs call.  Avoid duplicate lines for that path.
    if runner.__class__.__module__ == "vllm_gr.worker.gr_npu_model_runner":
        return

    if not isinstance(result, tuple) or len(result) < 7:
        return

    try:
        num_input_tokens = int(result[2])
        max_query_len = int(result[6])
    except (TypeError, ValueError):
        return

    total_num_scheduled_tokens = _extract_total_scheduled_tokens(scheduler_output)
    if total_num_scheduled_tokens is None:
        return

    num_reqs = _extract_num_reqs(runner)
    uniform_decode_query_len = getattr(runner, "uniform_decode_query_len", None)
    uniform_decode = bool(
        uniform_decode_query_len is not None
        and max_query_len == uniform_decode_query_len
        and num_reqs > 0
        and total_num_scheduled_tokens == num_reqs * max_query_len
    )

    beam_group_count, beam_group_reqs = _extract_beam_group_stats(scheduler_output)
    padding_ratio = (
        float(num_input_tokens) / float(total_num_scheduled_tokens)
        if total_num_scheduled_tokens > 0
        else 0.0
    )

    compilation_config = getattr(
        runner,
        "compilation_config",
        getattr(getattr(runner, "vllm_config", None), "compilation_config", None),
    )
    cudagraph_mode = getattr(compilation_config, "cudagraph_mode", None)

    logger.info(
        "[GR-NPU-PROFILE] runner=%s scheduled_tokens=%d num_input_tokens=%d "
        "padding_ratio=%.4f num_reqs=%d max_query_len=%d uniform_decode=%s "
        "use_aclgraph=%s cudagraph_mode=%s beam_group_count=%d beam_group_reqs=%d",
        runner.__class__.__name__,
        total_num_scheduled_tokens,
        num_input_tokens,
        padding_ratio,
        num_reqs,
        max_query_len,
        uniform_decode,
        getattr(runner, "use_aclgraph", None),
        cudagraph_mode,
        beam_group_count,
        beam_group_reqs,
    )


def _patch_runner_cls(runner_cls: Any, target_name: str) -> bool:
    if getattr(runner_cls, "_vllm_gr_npu_profile_patched", False):
        return False

    original_prepare_inputs = getattr(runner_cls, "_prepare_inputs", None)
    if original_prepare_inputs is None:
        return False

    @wraps(original_prepare_inputs)
    def patched_prepare_inputs(self, *args, **kwargs):
        scheduler_output = args[0] if args else kwargs.get("scheduler_output")
        result = original_prepare_inputs(self, *args, **kwargs)
        _maybe_log_prepare_inputs_profile(self, scheduler_output, result)
        return result

    runner_cls._prepare_inputs = patched_prepare_inputs
    runner_cls._vllm_gr_npu_profile_patched = True
    logger.info("vllm-gr %s profile patch installed.", target_name)
    return True


def _patch_import_target(module_name: str, class_name: str) -> bool:
    try:
        module = __import__(module_name, fromlist=[class_name])
    except (ImportError, ModuleNotFoundError):
        return False

    runner_cls = getattr(module, class_name, None)
    if runner_cls is None:
        return False

    return _patch_runner_cls(runner_cls, f"{module_name}.{class_name}")


def patch_npu_model_runner_profile() -> bool:
    """Patch online runner classes with optional GR/NPU profile logging.

    Qwen-style ``serve --gr`` traffic can still run through the standard
    vLLM/vLLM-Ascend generate path.  Depending on the vLLM-Ascend version,
    the called ``_prepare_inputs`` method may live on NPUModelRunner itself or
    on vLLM's GPUModelRunner base class, so patch both import targets when
    they are available.

    Returns True when at least one patch is installed.
    """

    patched = False
    targets = (
        ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner"),
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
    )
    for module_name, class_name in targets:
        patched = _patch_import_target(module_name, class_name) or patched
    return patched
