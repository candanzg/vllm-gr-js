# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support
ADD_BATCH and BEAM_REQUEST_STEP_UPDATE."""

from __future__ import annotations

import threading
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple
import os
import sys

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequestType

from vllm_gr.v1.attention.backends.beam_attn import _compute_beam_bounds

logger = init_logger(__name__)

ADD_BATCH_WIRE_BYTE = b"\x06"
BEAM_REQUEST_STEP_UPDATE_WIRE_BYTE = b"\x07"

# ---------------------------------------------------------------------------
# EngineCore.__init__ wrapper
# ---------------------------------------------------------------------------


def make_patched_engine_core_init(original_init):
    """Wrap EngineCore.__init__ to add beam state."""

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.beam_cache: dict[str, dict] = {}
        self.beam_cache_lock = threading.Lock()

    return patched_init


def make_patched_engine_core_proc_init(original_init):
    """Wrap EngineCoreProc.__init__ to add beam state for socket-thread helpers."""

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.beam_cache: dict[str, dict] = {}
        self.beam_cache_lock = threading.Lock()

    return patched_init


# ---------------------------------------------------------------------------
# Helpers: add enum members to EngineCoreRequestType at runtime
# ---------------------------------------------------------------------------


def _add_enum_member(name: str, value: bytes) -> None:
    """Dynamically add a member to EngineCoreRequestType if absent.

    This uses object.__new__() to bypass enum's normal creation pathway,
    which is necessary because we're monkey-patching at runtime and cannot
    modify the original class definition in vLLM's codebase.

    While this approach manipulates enum internals, it:
    - Is a standard technique used by enum extension libraries (e.g. aenum)
    - Includes validation to detect incompatibilities early
    - Is contained to initialization time (not runtime performance critical)
    - Is tested to ensure correctness

    Args:
        name: The name of the new enum member (e.g., "ADD_BATCH")
        value: The byte value for the enum member (e.g., b"\\x06")

    Raises:
        RuntimeError: If the enum's internal structure is incompatible
    """
    # Skip if already added
    if hasattr(EngineCoreRequestType, name):
        existing_member = getattr(EngineCoreRequestType, name)
        if existing_member.value != value:
            raise ValueError(
                f"EngineCoreRequestType.{name} already exists with value "
                f"{existing_member.value!r}, cannot reassign to {value!r}"
            )
        return

    # Validate that the enum has expected internal structure
    required_attrs = ["_value2member_map_", "_member_map_", "_member_names_"]
    for attr in required_attrs:
        if not hasattr(EngineCoreRequestType, attr):
            raise RuntimeError(
                f"EngineCoreRequestType missing expected attribute '{attr}'. "
                f"This indicates an incompatible Python/vLLM version. "
                f"Please report this issue with your Python version."
            )

    # Verify the value type matches existing members
    if EngineCoreRequestType._value2member_map_:
        existing_value_type = type(next(iter(EngineCoreRequestType._value2member_map_.keys())))
        if not isinstance(value, existing_value_type):
            raise TypeError(
                f"Value type mismatch: expected {existing_value_type.__name__}, "
                f"got {type(value).__name__}"
            )

    existing_member = EngineCoreRequestType._value2member_map_.get(value)
    if existing_member is not None:
        raise ValueError(
            f"EngineCoreRequestType value {value!r} is already used by {existing_member.name!r}"
        )

    # Create the member instance bypassing enum's __new__
    # (This is the standard approach for runtime enum extension)
    member = object.__new__(EngineCoreRequestType)
    member._name_ = name
    member._value_ = value

    # Register in enum's internal maps
    EngineCoreRequestType._value2member_map_[value] = member
    EngineCoreRequestType._member_map_[name] = member
    EngineCoreRequestType._member_names_.append(name)

    # Set as class attribute (must use type.__setattr__ to bypass enum's __setattr__)
    type.__setattr__(EngineCoreRequestType, name, member)


# ---------------------------------------------------------------------------
# Child-process patch applicator — called from patched run_engine_core
# ---------------------------------------------------------------------------


def apply_engine_core_child_patches():
    """Apply all patches inside the EngineCore child process.

    This must be called at the start of run_engine_core (the child process
    entry point) because multiprocessing 'spawn' starts a fresh Python
    interpreter that does not inherit monkey patches from the parent.
    """
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    if getattr(EngineCore, "_patched_for_beam_requests", False):
        _log_beam_debug("apply_engine_core_child_patches already applied")
        return

    from vllm_gr.v1.engine.core import (
        _cache_beam_request,
        _handle_beam_request_step_update,
        process_input_sockets,
    )

    logger.info("[vllm-gr] apply_engine_core_child_patches called.")
    _log_beam_debug("apply_engine_core_child_patches called")

    # Add new enum members: ADD_BATCH, BEAM_REQUEST_STEP_UPDATE
    _add_enum_member("ADD_BATCH", ADD_BATCH_WIRE_BYTE)
    _add_enum_member("BEAM_REQUEST_STEP_UPDATE", BEAM_REQUEST_STEP_UPDATE_WIRE_BYTE)

    # Wrap EngineCore / EngineCoreProc.__init__ to add beam state
    EngineCore.__init__ = make_patched_engine_core_init(EngineCore.__init__)
    EngineCoreProc.__init__ = make_patched_engine_core_proc_init(EngineCoreProc.__init__)

    # Bind _cache_beam_request and _handle_beam_request_step_update on both classes
    EngineCore._cache_beam_request = _cache_beam_request
    EngineCore._handle_beam_request_step_update = _handle_beam_request_step_update

    # Replace EngineCoreProc.process_input_sockets
    EngineCoreProc.process_input_sockets = process_input_sockets

    from vllm.sampling_params import SamplingParams

    # Force the greedy verification check to be a no-op to allow n > 1 under greedy beam search parameters
    SamplingParams._verify_greedy_sampling = lambda self: None

    # Apply scheduler patches in the child process
    apply_scheduler_patch()

    # Apply worker patches (e.g., GPUModelRunner)
    apply_worker_patches()

    EngineCore._patched_for_beam_requests = True
    logger.debug("Engine-core patches applied in child process.")


# Re-entry guard to detect recursive calls
_run_engine_core_active = False


def run_engine_core(*args, **kwargs):
    """Wrapper for EngineCoreProc.run_engine_core that applies patches
    before the engine starts.  This is a module-level function so that
    multiprocessing 'spawn' can pickle it by qualified name.

    In the spawned child process, all modules are freshly imported, so
    EngineCoreProc.run_engine_core is the original unpatched version.
    We apply our patches first, then delegate to the original.
    """
    global _run_engine_core_active

    # Re-entry guard: detect if we're already inside run_engine_core
    if _run_engine_core_active:
        raise RuntimeError(
            "run_engine_core re-entry detected. This indicates a recursive "
            "call, likely due to patching being applied multiple times. "
            "Check that patch_batch_and_fork() is only called once."
        )

    from vllm.v1.engine.core import EngineCoreProc

    # Guard against recursion: if vllm_gr/init.py was auto-imported in
    # the child process, run_patch() will have replaced
    # EngineCoreProc.run_engine_core with *this* function again.
    # Use the saved original stashed by patch_batch_and_fork() if
    # available; otherwise fall back to the current class attribute
    # (which is the unpatched original in a fresh child interpreter).
    original_run = getattr(run_engine_core, "_original_run_engine_core", None)

    # Validate the saved original
    if original_run is not None and original_run is not run_engine_core:
        # We have a saved original that isn't ourselves - validate it
        if not callable(original_run):
            raise RuntimeError(
                f"run_engine_core._original_run_engine_core is not callable: "
                f"type={type(original_run)}"
            )
    else:
        # No saved original (or it points to ourselves, which is invalid)
        # Fall back to the current class attribute (should be the original in a fresh process)
        if original_run is run_engine_core:
            logger.warning(
                "run_engine_core._original_run_engine_core points to itself. "
                "This may indicate corrupted patch state from a previous run. "
                "Falling back to EngineCoreProc.run_engine_core."
            )
        original_run = EngineCoreProc.run_engine_core

    # Final validation: ensure we didn't end up with ourselves
    if original_run is run_engine_core:
        raise RuntimeError(
            "run_engine_core recursion detected: EngineCoreProc.run_engine_core "
            "points to this wrapper. This indicates that patching was applied "
            "but the original was not saved, or the child process imported "
            "vllm_gr before starting."
        )

    # Validate that original_run looks like a real function from vllm
    # (Skip validation for test mocks)
    if hasattr(original_run, "__module__"):
        original_module = original_run.__module__
        expected_module = "vllm.v1.engine.core"
        # Skip warning for unittest mocks (used in tests)
        if original_module not in (expected_module, "unittest.mock", "mock"):
            logger.warning(
                "Unexpected module for original run_engine_core: %s (expected %s). "
                "Proceeding anyway, but this may indicate an issue.",
                original_module,
                expected_module,
            )

    # Set re-entry guard and apply patches
    _run_engine_core_active = True
    try:
        apply_engine_core_child_patches()
        original_run(*args, **kwargs)
    finally:
        _run_engine_core_active = False


def apply_scheduler_patch():
    """Monkey-patch Scheduler.schedule to propagate custom attributes via SchedulerOutput."""
    from functools import wraps

    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "_patched_for_beam_requests", False):
        return

    _original_schedule = Scheduler.schedule

    @wraps(_original_schedule)
    def patched_schedule(self):
        _original_get_computed_blocks = self.kv_cache_manager.get_computed_blocks
        block_size = self.cache_config.block_size

        def patch_get_computed_blocks(req):
            bn, tn = _original_get_computed_blocks(req)
            if req and getattr(req, "is_beam_decode", False):
                pl = getattr(req, "prefix_len", 0) // block_size * block_size
                if tn > pl:
                    sliced_blocks = tuple(group[: pl // block_size] for group in bn.blocks)
                    return self.kv_cache_manager.create_kv_cache_blocks(sliced_blocks), pl
            return bn, tn

        self.kv_cache_manager.get_computed_blocks = patch_get_computed_blocks
        try:
            output = _original_schedule(self)

            # Propagate beam request attributes from Request objects to SchedulerOutput
            beam_data = {}
            if hasattr(output, "num_scheduled_tokens"):
                for req_id in output.num_scheduled_tokens.keys():
                    req = self.requests.get(req_id)
                    if req and getattr(req, "is_beam_decode", False):
                        beam_data[req_id] = {
                            "is_beam_decode": True,
                            "beam_width": getattr(req, "beam_width", 1),
                            "beam_decode_steps": getattr(req, "beam_decode_steps", 0),
                            "prefix_len": getattr(req, "prefix_len", 0),
                            "cache_len": req.num_computed_tokens
                            - output.num_scheduled_tokens[req_id],
                        }
            output.beam_data = beam_data
            return output
        finally:
            self.kv_cache_manager.get_computed_blocks = _original_get_computed_blocks

    Scheduler.schedule = patched_schedule
    Scheduler._patched_for_beam_requests = True
    logger.debug("Scheduler.schedule patched for beam-requests propagation.")


# ============================================================================
# TENSOR AND BUFFER GEOMETRY EXTRACTION HELPERS
# ============================================================================


def _get_positions_buffers(runner: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Safely extracts underlying hardware and host indexing position tensors."""
    gpu_positions = None
    cpu_positions = None
    if hasattr(runner, "positions") and runner.positions is not None:
        positions = runner.positions
        gpu_candidate = getattr(positions, "gpu", None)
        cpu_candidate = getattr(positions, "cpu", None)

        if isinstance(gpu_candidate, torch.Tensor):
            gpu_positions = gpu_candidate
        elif isinstance(positions, torch.Tensor):
            gpu_positions = positions

        if isinstance(cpu_candidate, torch.Tensor):
            cpu_positions = cpu_candidate
        elif hasattr(runner, "_positions_cpu_buf") and isinstance(
            runner._positions_cpu_buf, torch.Tensor
        ):
            cpu_positions = runner._positions_cpu_buf
    return gpu_positions, cpu_positions


def _write_position_segments(
    token_offset: int,
    seg_starts: List[int],
    seg_lens: List[int],
    gpu_pos: Optional[torch.Tensor],
    cpu_pos: Optional[torch.Tensor],
) -> None:
    """Write concatenated per-beam position-id segments in one vectorized pass.

    Each active beam contributes ``range(st_b, st_b + len_b)``. The previous
    implementation emitted one ``torch.arange`` per beam (up to ``beam_width``
    GPU kernel launches per call); here we build the whole concatenation on CPU
    with a handful of vectorized ops and a single host->device copy for
    ``gpu_pos``. The result is element-wise identical to the per-beam writes.
    """
    if not seg_lens:
        return
    lens = torch.tensor(seg_lens, dtype=torch.int64)
    total = int(lens.sum().item())
    if total == 0:
        return
    starts = torch.tensor(seg_starts, dtype=torch.int64)
    # Output offset where each segment begins (prefix sum minus own length).
    seg_off = torch.cumsum(lens, dim=0) - lens
    gidx = torch.arange(total, dtype=torch.int64)
    seg_off_rep = torch.repeat_interleave(seg_off, lens)
    starts_rep = torch.repeat_interleave(starts, lens)
    positions = starts_rep + (gidx - seg_off_rep)
    if gpu_pos is not None:
        gpu_pos[token_offset : token_offset + total] = positions
    if cpu_pos is not None:
        cpu_pos[token_offset : token_offset + total] = positions


def _process_beam_decode_request(
    mdata: Dict[str, Any],
    seq_len: int,
    token_offset: int,
    gpu_pos: Optional[torch.Tensor],
    cpu_pos: Optional[torch.Tensor],
    new_logits_indices: List[int],
) -> Tuple[int, int, int, List[int]]:
    """Traces allocation arrays and mutates position buffers for a speculative request."""
    prefix_len = mdata["prefix_len"]
    beam_width = mdata["beam_width"]
    decode_steps = mdata["beam_decode_steps"]
    cache_len = mdata["cache_len"]
    active_beam_ids = []  # Track structural beam locations

    chunk_budget = seq_len
    curr_offset = token_offset
    valid_rows_for_req = 0
    logits_row_cnt = 0
    b_cache_len = cache_len
    seg_starts: List[int] = []
    seg_lens: List[int] = []
    for b in range(beam_width):
        if chunk_budget <= 0:
            break

        b_uncached, b_suffix_len, b_past_suffix, b_physical = _compute_beam_bounds(
            b, prefix_len, b_cache_len, decode_steps, chunk_budget
        )

        if b_uncached > 0:
            st = cache_len if (prefix_len >= b_cache_len) else (prefix_len + b_past_suffix)
            seg_starts.append(st)
            seg_lens.append(b_uncached)

            if b_past_suffix + b_uncached >= decode_steps:
                new_logits_indices.append(curr_offset + b_uncached - 1)
                active_beam_ids.append(b_physical)

                logits_row_cnt += 1

            valid_rows_for_req += 1
            curr_offset += b_uncached
            chunk_budget -= b_uncached
            b_cache_len += b_uncached

    # Batched position-id write: one vectorized build (CPU) + single H2D copy
    # instead of one torch.arange per beam (up to beam_width GPU launches/call).
    _write_position_segments(token_offset, seg_starts, seg_lens, gpu_pos, cpu_pos)

    return valid_rows_for_req, logits_row_cnt, curr_offset, active_beam_ids


# ============================================================================
# CORE WORKER PIPELINE WELDER LOGIC
# ============================================================================


def _remux_logprobs_tensors(
    request_row_configs: List[Tuple[Any, int, int]],
    max_chained_dim: int,
    target_K: int,
    orig_lps: torch.Tensor,
    orig_lp_ids: Optional[torch.Tensor],
    orig_ranks: Optional[torch.Tensor],
    st_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Merges and aligns sampler tensor slices sequentially across runtime waves."""
    if max_chained_dim == 0 or orig_lps.numel() == 0:
        return st_ids, orig_lps, orig_lp_ids, orig_ranks

    new_st_ids, new_lps, new_lp_ids, new_ranks = [], [], [], []
    gpu_row_cursor = 0

    for req_id, _, W_logits, *_ in request_row_configs:
        # --------------------------------------------------------------------
        # CASE A: INTERMEDIATE CHUNKED PREFILL (No output rows generated)
        # --------------------------------------------------------------------
        if W_logits == 0:
            dummy_st = torch.tensor([[-1]], dtype=st_ids.dtype, device=st_ids.device)
            new_st_ids.append(dummy_st)

            new_lps.append(
                torch.full(
                    (1, max_chained_dim),
                    float("-inf"),
                    dtype=orig_lps.dtype,
                    device=orig_lps.device,
                )
            )
            if orig_lp_ids is not None:
                new_lp_ids.append(
                    torch.zeros(
                        (1, max_chained_dim), dtype=orig_lp_ids.dtype, device=orig_lp_ids.device
                    )
                )
            if orig_ranks is not None:
                new_ranks.append(
                    torch.zeros(
                        (1, max_chained_dim), dtype=orig_ranks.dtype, device=orig_ranks.device
                    )
                )
            continue

        # --------------------------------------------------------------------
        # CASE B: ACTIVE SAMPLING PATH (Captures the full multi-beam footprint)
        # --------------------------------------------------------------------
        # Retain leader token track identity
        new_st_ids.append(st_ids[gpu_row_cursor : gpu_row_cursor + 1, :])

        req_lps = orig_lps[gpu_row_cursor : gpu_row_cursor + W_logits, :target_K].reshape(
            1, W_logits * target_K
        )
        if W_logits * target_K < max_chained_dim:
            req_lps = torch.nn.functional.pad(
                req_lps, (0, max_chained_dim - (W_logits * target_K)), value=float("-inf")
            )
        new_lps.append(req_lps)

        if orig_lp_ids is not None:
            req_lp_ids = orig_lp_ids[gpu_row_cursor : gpu_row_cursor + W_logits, :target_K].reshape(
                1, W_logits * target_K
            )
            if W_logits * target_K < max_chained_dim:
                req_lp_ids = torch.nn.functional.pad(
                    req_lp_ids, (0, max_chained_dim - (W_logits * target_K)), value=0
                )
            new_lp_ids.append(req_lp_ids)

        if orig_ranks is not None:
            req_ranks = orig_ranks[gpu_row_cursor : gpu_row_cursor + W_logits].reshape(1, W_logits)
            if W_logits < max_chained_dim:
                req_ranks = torch.nn.functional.pad(
                    req_ranks, (0, max_chained_dim - W_logits), value=0
                )
            new_ranks.append(req_ranks)

        # Advance the pointer cursor past the complete multi-beam block size
        gpu_row_cursor += W_logits

    return (
        torch.cat(new_st_ids, dim=0),
        torch.cat(new_lps, dim=0),
        torch.cat(new_lp_ids, dim=0) if orig_lp_ids is not None else None,
        torch.cat(new_ranks, dim=0) if orig_ranks is not None else None,
    )


# ============================================================================
# INLINE COMPILATION PROXIES FOR HIGH-PERFORMANCE INTERCEPTION
# ============================================================================


class BeamReqIdsProxy(list):
    def __init__(self, original_list, req_id_map):
        super().__init__(original_list)
        self.req_id_map = req_id_map

    def __getitem__(self, index):
        try:
            return self.req_id_map[index]
        except IndexError as e:
            logger.error("Out of bounds in BeamReqIdsProxy for index %s", index)
            raise e

    def copy(self):
        return list(self.req_id_map)


class Mock1DArray:
    def __init__(self, original, batch_idx_map):
        self.original = original
        self.batch_idx_map = batch_idx_map

    def __getitem__(self, idx):
        try:
            return self.original[self.batch_idx_map[idx]]
        except IndexError as e:
            logger.error("Out of bounds in Mock1DArray getitem for index %s", idx)
            raise e

    def __setitem__(self, idx, value):
        try:
            self.original[self.batch_idx_map[idx]] = value
        except IndexError as e:
            logger.error("Out of bounds in Mock1DArray setitem for index %s", idx)
            raise e


class Mock2DArray:
    def __init__(self, original, batch_idx_map):
        self.original = original
        self.batch_idx_map = batch_idx_map

    def __setitem__(self, key, value):
        idx, slc = key
        try:
            self.original[self.batch_idx_map[idx], slc] = value
        except IndexError as e:
            logger.error("Out of bounds in Mock2DArray setitem for key %s", key)
            raise e

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __getitem__(self, key):
        idx, slc = key
        try:
            return self.original[self.batch_idx_map[idx], slc]
        except IndexError as e:
            logger.error("Out of bounds in Mock2DArray getitem for key %s", key)
            raise e


class InputBatchProxy:
    def __init__(self, obj, req_map, idx_map, logits_map):
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_req_id_map", req_map)
        object.__setattr__(self, "_batch_idx_map", idx_map)
        object.__setattr__(self, "_logit_idx_map", logits_map)

    def __getattr__(self, name):
        obj = object.__getattribute__(self, "_obj")
        idx_map = object.__getattribute__(self, "_batch_idx_map")
        logits_map = object.__getattribute__(self, "_logit_idx_map")
        req_map = object.__getattribute__(self, "_req_id_map")

        if name == "sampling_metadata":
            s_meta = getattr(obj, "sampling_metadata", None)
            if s_meta is None or logits_map is None:
                return s_meta

            orig_req_len = len(obj.req_ids) if hasattr(obj, "req_ids") else 1
            for attr in [
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "frequency_penalties",
                "presence_penalties",
                "repetition_penalties",
            ]:
                val = getattr(s_meta, attr, None)
                if isinstance(val, torch.Tensor) and val.numel() > 0:
                    if val.shape[0] == orig_req_len and val.shape[0] != len(logits_map):
                        object.__setattr__(s_meta, attr, val[logits_map])
            return s_meta

        elif name == "req_ids":
            return BeamReqIdsProxy(obj.req_ids, req_map)
        elif name == "num_tokens_no_spec":
            return Mock1DArray(obj.num_tokens_no_spec, idx_map)
        elif name == "num_tokens":
            # NPU's _bookkeeping_sync writeback loop also updates num_tokens
            # (the GPU path does not). Map it to the logical batch slot like
            # num_tokens_no_spec. Inert when never accessed (e.g. GPU).
            return Mock1DArray(obj.num_tokens, idx_map)
        elif name == "token_ids_cpu":
            return Mock2DArray(obj.token_ids_cpu, idx_map)
        elif name == "is_token_ids":
            return (
                Mock2DArray(obj.is_token_ids, idx_map)
                if hasattr(obj, "is_token_ids")
                else Mock2DArray(obj.token_ids_cpu, idx_map)
            )
        return getattr(obj, name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_obj"), name, value)


# ============================================================================
# MONKEY PATCH APPLICATOR REGISTRY
# ============================================================================


def _beam_remap_inputs(runner, scheduler_output, num_scheduled_tokens) -> List[int]:
    """Expand a packed beam-decode batch into per-beam rows.

    Device-neutral. Overwrites per-beam RoPE position ids (Job B), builds the
    per-beam ``logits_indices`` list, installs an ``InputBatchProxy`` on
    ``runner.input_batch`` and stashes ``scheduler_output._gr_request_row_configs``
    (Job C). Job A (populating ``beam_info["req_ids"]``) is left to the caller
    because its ordering relative to the original ``_prepare_inputs`` differs
    between GPU and NPU.

    Assumes ``scheduler_output.beam_data`` is non-empty (caller checks).

    Returns:
        The new per-beam logits-index list (caller converts to a tensor).
    """
    beam_data = getattr(scheduler_output, "beam_data", {})

    runner.input_batch.logprob_token_ids = {}
    new_logits_indices: List[int] = []
    orig_req_ids = runner.input_batch.req_ids

    token_offset = 0
    row_to_req_id, row_to_batch_idx, logits_to_batch_idx, request_row_configs = ([], [], [], [])
    input_row_idx = 0

    gpu_positions, cpu_positions = _get_positions_buffers(runner)

    for batch_idx, req_id in enumerate(orig_req_ids):
        seq_len = num_scheduled_tokens[batch_idx] if batch_idx < len(num_scheduled_tokens) else 0
        if req_id is None:
            row_to_req_id.append(None)
            row_to_batch_idx.append(batch_idx)
            input_row_idx += 1
            token_offset += seq_len
            continue

        mdata = beam_data.get(req_id)
        if mdata and mdata.get("is_beam_decode", False):
            beams_count, beams_count_with_logits, next_offset, active_beam_ids = (
                _process_beam_decode_request(
                    mdata,
                    seq_len,
                    token_offset,
                    gpu_positions,
                    cpu_positions,
                    new_logits_indices,
                )
            )
            row_to_req_id.extend([req_id] * beams_count)
            row_to_batch_idx.extend([batch_idx] * beams_count)
            logits_to_batch_idx.extend([batch_idx] * beams_count_with_logits)

            request_row_configs.append(
                (req_id, input_row_idx, beams_count_with_logits, active_beam_ids)
            )
            input_row_idx += beams_count
        else:
            num_computed_tokens = runner.input_batch.num_computed_tokens_cpu[batch_idx]
            num_prompt_tokens = runner.input_batch.num_prompt_tokens[batch_idx]

            is_last_chunk_prefill = bool(
                num_computed_tokens < num_prompt_tokens
                and num_computed_tokens + seq_len >= num_prompt_tokens
            )
            beams_count_with_logits = 0
            beams_count = 1
            if is_last_chunk_prefill:
                new_logits_indices.append(token_offset + seq_len - 1)
                beams_count_with_logits = 1
                logits_to_batch_idx.extend([batch_idx] * beams_count_with_logits)
            row_to_req_id.extend([req_id] * beams_count)
            row_to_batch_idx.extend([batch_idx] * beams_count)

            request_row_configs.append((req_id, input_row_idx, beams_count_with_logits, []))
            input_row_idx += beams_count

        token_offset += seq_len

    runner.input_batch = InputBatchProxy(
        runner.input_batch, row_to_req_id, row_to_batch_idx, logits_to_batch_idx
    )

    # vLLM 0.22.1 threads expanded-logits bookkeeping through explicit
    # InputBatch fields. The old beam patch only had to rewrite
    # logits_indices; now we must keep cu_num_logits / expanded_idx_mapping in
    # sync so the sampler and logprob gather path can see every beam row.
    cu_num_logits_np = np.empty(len(request_row_configs) + 1, dtype=np.int32)
    cu_num_logits_np[0] = 0
    expanded_req_state_indices: list[int] = []
    expanded_local_pos: list[int] = []
    orig_input_batch = object.__getattribute__(runner.input_batch, "_obj")
    for req_pos, (_req_id, input_row_idx, beams_count_with_logits, _active_beams) in enumerate(
        request_row_configs
    ):
        cu_num_logits_np[req_pos + 1] = cu_num_logits_np[req_pos] + beams_count_with_logits
        if beams_count_with_logits <= 0:
            continue
        batch_idx = row_to_batch_idx[input_row_idx]
        req_state_idx = int(orig_input_batch.idx_mapping_np[batch_idx])
        expanded_req_state_indices.extend([req_state_idx] * beams_count_with_logits)
        expanded_local_pos.extend(range(beams_count_with_logits))

    runner.input_batch.cu_num_logits_np = cu_num_logits_np
    runner.input_batch.cu_num_logits = torch.tensor(
        cu_num_logits_np, dtype=torch.int32, device=runner.device
    )
    runner.input_batch.expanded_idx_mapping = torch.tensor(
        expanded_req_state_indices,
        dtype=torch.int32,
        device=runner.device,
    )
    runner.input_batch.expanded_local_pos = torch.zeros(
        len(expanded_local_pos),
        dtype=torch.int32,
        device=runner.device,
    )
    if expanded_local_pos:
        runner.input_batch.expanded_local_pos.copy_(
            torch.tensor(expanded_local_pos, dtype=torch.int32, device=runner.device)
        )
    scheduler_output._gr_request_row_configs = request_row_configs

    return new_logits_indices


def _beam_remap_inputs_v2(runner, input_batch, scheduler_output) -> None:
    """Expand V2 ``InputBatch`` logits rows for merged beam decode.

    Unlike the V1 runner, ModelRunner V2 does not expose a mutable
    ``runner.input_batch`` with the legacy helper fields. The beam-search
    scheduler output still carries the same logical request metadata, so we
    rebuild only the sampling-facing pieces of ``InputBatch`` in place:
    ``logits_indices``, ``cu_num_logits``, ``expanded_idx_mapping``,
    ``expanded_local_pos``, and per-beam position ids.
    """
    beam_data = getattr(scheduler_output, "beam_data", {})
    if not beam_data:
        return

    new_logits_indices: list[int] = []
    request_row_configs: list[tuple[Any, int, int, list[int]]] = []
    token_offset = 0
    input_row_idx = 0
    gpu_positions = input_batch.positions
    row_to_batch_idx: list[int] = []

    req_state_indices = input_batch.idx_mapping_np
    num_computed_tokens_np = runner.req_states.num_computed_tokens_np
    prompt_len_np = runner.req_states.prompt_len.np

    for batch_idx, req_id in enumerate(input_batch.req_ids):
        seq_len = int(input_batch.num_scheduled_tokens[batch_idx])
        if req_id is None:
            input_row_idx += 1
            token_offset += seq_len
            continue

        mdata = beam_data.get(req_id)
        if mdata and mdata.get("is_beam_decode", False):
            beams_count, beams_count_with_logits, _next_offset, active_beam_ids = (
                _process_beam_decode_request(
                    mdata,
                    seq_len,
                    token_offset,
                    gpu_positions,
                    None,
                    new_logits_indices,
                )
            )
            request_row_configs.append(
                (req_id, input_row_idx, beams_count_with_logits, active_beam_ids)
            )
            row_to_batch_idx.append(batch_idx)
            input_row_idx += beams_count
        else:
            req_state_idx = int(req_state_indices[batch_idx])
            num_computed_tokens = int(num_computed_tokens_np[req_state_idx])
            num_prompt_tokens = int(prompt_len_np[req_state_idx])
            is_last_chunk_prefill = bool(
                num_computed_tokens < num_prompt_tokens
                and num_computed_tokens + seq_len >= num_prompt_tokens
            )
            beams_count_with_logits = 1 if is_last_chunk_prefill else 0
            if is_last_chunk_prefill:
                new_logits_indices.append(token_offset + seq_len - 1)
            request_row_configs.append((req_id, input_row_idx, beams_count_with_logits, []))
            row_to_batch_idx.append(batch_idx)
            input_row_idx += 1

        token_offset += seq_len

    cu_num_logits_np = np.empty(len(request_row_configs) + 1, dtype=np.int32)
    cu_num_logits_np[0] = 0
    expanded_req_state_indices: list[int] = []
    expanded_local_pos: list[int] = []
    per_req_num_sampled: list[int] = []
    for req_pos, (_req_id, input_row_idx, beams_count_with_logits, _active_beams) in enumerate(
        request_row_configs
    ):
        cu_num_logits_np[req_pos + 1] = cu_num_logits_np[req_pos] + beams_count_with_logits
        per_req_num_sampled.append(1 if beams_count_with_logits > 0 else 0)
        if beams_count_with_logits <= 0:
            continue
        batch_idx = row_to_batch_idx[req_pos]
        req_state_idx = int(req_state_indices[batch_idx])
        expanded_req_state_indices.extend([req_state_idx] * beams_count_with_logits)
        expanded_local_pos.extend(range(beams_count_with_logits))

    input_batch.logits_indices = torch.tensor(
        new_logits_indices, dtype=torch.int32, device=runner.device
    )
    input_batch.cu_num_logits_np = cu_num_logits_np
    input_batch.cu_num_logits = torch.tensor(
        cu_num_logits_np, dtype=torch.int32, device=runner.device
    )
    input_batch.expanded_idx_mapping = torch.tensor(
        expanded_req_state_indices,
        dtype=torch.int32,
        device=runner.device,
    )
    input_batch.expanded_local_pos = torch.tensor(
        expanded_local_pos,
        dtype=torch.int32,
        device=runner.device,
    )
    input_batch._gr_request_row_configs = request_row_configs
    input_batch._gr_num_sampled_per_req = per_req_num_sampled


def _beam_collapse_bookkeeping(
    runner, orig_input_batch, request_row_configs, beam_data, sampler_output
) -> Tuple[List[Any], Dict[Any, int], int]:
    """Collapse per-beam sampler output back onto logical (session) requests.

    Device-neutral. Remuxes the sampler logprobs tensors in place so that each
    logical request owns one flattened ``W * K`` row, and resizes the sampler
    output tensors accordingly.

    Returns:
        ``(collapsed_req_ids, collapsed_req_id_to_index, num_logical_requests)``.
    """
    collapsed_req_ids_copy = [r[0] for r in request_row_configs if r[0] is not None]
    collapsed_req_id_to_index = {req_id: idx for idx, req_id in enumerate(collapsed_req_ids_copy)}
    num_logical_requests = len(collapsed_req_ids_copy)

    if sampler_output is not None:
        st_ids = sampler_output.sampled_token_ids
        lp_tensors = sampler_output.logprobs_tensors
        if st_ids is not None and lp_tensors is not None and lp_tensors.logprobs is not None:
            orig_lps = lp_tensors.logprobs
            orig_lp_ids = lp_tensors.logprob_token_ids
            orig_ranks = getattr(lp_tensors, "selected_token_ranks", None)
            actual_K = orig_lps.shape[1]

            max_chained_dim = 0
            for req_id, _, W, *_ in request_row_configs:
                if req_id is not None and req_id in beam_data:
                    m_width = beam_data[req_id].get("beam_width", 1)
                    needed_dim = m_width * actual_K
                else:
                    needed_dim = W * actual_K
                if needed_dim > max_chained_dim:
                    max_chained_dim = needed_dim

            final_st_ids, final_lps, final_lp_ids, final_ranks = _remux_logprobs_tensors(
                request_row_configs,
                max_chained_dim,
                actual_K,
                orig_lps,
                orig_lp_ids,
                orig_ranks,
                st_ids,
            )
            if not hasattr(runner, "_beam_logprobs_cache"):
                runner._beam_logprobs_cache = {}

            req_id_to_config = {r[0]: (r[2], r[3]) for r in request_row_configs if r[0] is not None}

            for idx, req_id in enumerate(collapsed_req_ids_copy):
                mdata = beam_data.get(req_id)
                if mdata and mdata.get("is_beam_decode", False):
                    logits_num, active_beams = req_id_to_config.get(req_id, (0, []))
                    total_beam_width = mdata.get("beam_width", 1)

                    if logits_num > 0:
                        # ---------------------------------------------------------------------
                        # FAST-PATH: This chunk contains ALL beams directly (No Split occurred)
                        # ---------------------------------------------------------------------
                        if logits_num == total_beam_width:
                            # Clean up stale entries left behind by older preempted/aborted steps
                            if req_id in runner._beam_logprobs_cache:
                                del runner._beam_logprobs_cache[req_id]

                            # Skip all caching loops! final_lps[idx] is already fully populated.
                            continue

                        # ---------------------------------------------------------------------
                        # SLOW-PATH: Handled split chunked prefills across steps
                        # ---------------------------------------------------------------------
                        # Extract this chunk's output elements
                        actual_lps = final_lps[idx, : logits_num * actual_K].view(
                            logits_num, actual_K
                        )
                        actual_lp_ids = (
                            final_lp_ids[idx, : logits_num * actual_K].view(logits_num, actual_K)
                            if final_lp_ids is not None
                            else None
                        )
                        actual_ranks = (
                            final_ranks[idx, :logits_num] if final_ranks is not None else None
                        )

                        # Initialize pre-allocated tensor cache matrix structures only if needed
                        if req_id not in runner._beam_logprobs_cache:
                            runner._beam_logprobs_cache[req_id] = {
                                "lps": torch.full(
                                    (total_beam_width, actual_K),
                                    float("-inf"),
                                    device=orig_lps.device,
                                    dtype=orig_lps.dtype,
                                ),
                                "lp_ids": torch.zeros(
                                    (total_beam_width, actual_K),
                                    dtype=orig_lp_ids.tensor.dtype
                                    if hasattr(orig_lp_ids, "tensor")
                                    else torch.long,
                                    device=orig_lps.device,
                                )
                                if orig_lp_ids is not None
                                else None,
                                "ranks": torch.zeros(
                                    total_beam_width, dtype=torch.long, device=orig_lps.device
                                )
                                if orig_ranks is not None
                                else None,
                                "completed_beams": set(),
                            }

                        cache_entry = runner._beam_logprobs_cache[req_id]

                        # Scatter write chunk data lines directly to their physical beam slots
                        for local_row_idx, physical_beam_id in enumerate(active_beams):
                            if physical_beam_id not in cache_entry["completed_beams"]:
                                cache_entry["lps"][physical_beam_id] = actual_lps[local_row_idx]
                                if actual_lp_ids is not None:
                                    cache_entry["lp_ids"][physical_beam_id] = actual_lp_ids[
                                        local_row_idx
                                    ]
                                if actual_ranks is not None:
                                    cache_entry["ranks"][physical_beam_id] = actual_ranks[
                                        local_row_idx
                                    ]
                                cache_entry["completed_beams"].add(physical_beam_id)

                        # Evict and overwrite target boundaries only when ALL unique structural slots are filled
                        if len(cache_entry["completed_beams"]) >= total_beam_width:
                            final_lps[idx].copy_(cache_entry["lps"].view(-1))
                            if final_lp_ids is not None:
                                final_lp_ids[idx].copy_(cache_entry["lp_ids"].view(-1))
                            if final_ranks is not None:
                                final_ranks[idx, :total_beam_width].copy_(cache_entry["ranks"])

                            del runner._beam_logprobs_cache[req_id]

            try:
                st_ids.resize_(final_st_ids.shape).copy_(final_st_ids)
                lp_tensors.logprobs.resize_(final_lps.shape).copy_(final_lps)

                if lp_tensors.logprob_token_ids is not None and final_lp_ids is not None:
                    lp_tensors.logprob_token_ids.resize_(final_lp_ids.shape).copy_(final_lp_ids)
                if orig_ranks is not None and final_ranks is not None:
                    clean_ranks_slice = final_ranks[:num_logical_requests, 0].contiguous()
                    lp_tensors.selected_token_ranks.resize_(clean_ranks_slice.shape).copy_(
                        clean_ranks_slice
                    )
            except Exception as e:
                logger.critical("Inplace copy failed during beam collapse: %s", e)
                raise RuntimeError("Inplace copy failed during beam collapse") from e

    return collapsed_req_ids_copy, collapsed_req_id_to_index, num_logical_requests


def _snapshot_beam_runner_state(orig_input_batch, request_row_configs, beam_data, requests):
    """Capture mutable per-request state that merged beam decode must not persist.

    vLLM's native bookkeeping assumes one sampled row per logical request and
    mutates request/output-token state in place. Our merged beam decode expands
    one logical request into multiple rows only for sampling/logprob
    computation; the authoritative beam state still lives in GRLLM's external
    beam-search loop. We therefore restore these fields after calling the
    original bookkeeping implementation.
    """
    beam_batch_indices: list[int] = []
    beam_req_ids: list[Any] = []
    seen_batch_indices: set[int] = set()
    seen_req_ids: set[Any] = set()

    for req_id, input_row_idx, _logits_num, _active_beams in request_row_configs:
        mdata = beam_data.get(req_id)
        if not (mdata and mdata.get("is_beam_decode", False)):
            continue
        batch_idx = int(orig_input_batch.req_id_to_index[req_id])
        if batch_idx not in seen_batch_indices:
            seen_batch_indices.add(batch_idx)
            beam_batch_indices.append(batch_idx)
        if req_id not in seen_req_ids:
            seen_req_ids.add(req_id)
            beam_req_ids.append(req_id)

    if not beam_batch_indices and not beam_req_ids:
        return None

    token_ids_cpu = orig_input_batch.token_ids_cpu
    is_token_ids = getattr(orig_input_batch, "is_token_ids", None)

    return {
        "beam_batch_indices": beam_batch_indices,
        "beam_req_ids": beam_req_ids,
        "token_ids_cpu": {
            batch_idx: token_ids_cpu[batch_idx].copy() for batch_idx in beam_batch_indices
        },
        "is_token_ids": (
            {batch_idx: is_token_ids[batch_idx].copy() for batch_idx in beam_batch_indices}
            if is_token_ids is not None
            else None
        ),
        "num_tokens_no_spec": {
            batch_idx: int(orig_input_batch.num_tokens_no_spec[batch_idx])
            for batch_idx in beam_batch_indices
        },
        "num_tokens": (
            {
                batch_idx: int(orig_input_batch.num_tokens[batch_idx])
                for batch_idx in beam_batch_indices
            }
            if hasattr(orig_input_batch, "num_tokens")
            else None
        ),
        "request_output_token_ids": {
            req_id: list(requests[req_id].output_token_ids) for req_id in beam_req_ids if req_id in requests
        },
    }


def _restore_beam_runner_state(orig_input_batch, requests, snapshot) -> None:
    if snapshot is None:
        return

    for batch_idx, saved in snapshot["token_ids_cpu"].items():
        orig_input_batch.token_ids_cpu[batch_idx] = saved

    saved_is_token_ids = snapshot.get("is_token_ids")
    if saved_is_token_ids is not None:
        is_token_ids = getattr(orig_input_batch, "is_token_ids", None)
        if is_token_ids is not None:
            for batch_idx, saved in saved_is_token_ids.items():
                is_token_ids[batch_idx] = saved

    for batch_idx, saved in snapshot["num_tokens_no_spec"].items():
        orig_input_batch.num_tokens_no_spec[batch_idx] = saved

    saved_num_tokens = snapshot.get("num_tokens")
    if saved_num_tokens is not None and hasattr(orig_input_batch, "num_tokens"):
        for batch_idx, saved in saved_num_tokens.items():
            orig_input_batch.num_tokens[batch_idx] = saved

    for req_id, saved in snapshot["request_output_token_ids"].items():
        if req_id in requests:
            requests[req_id].output_token_ids = list(saved)


def _sampler_output_to_valid_token_lists(sampler_output) -> list[list[int]]:
    if sampler_output is None or sampler_output.sampled_token_ids is None:
        return []

    sampled = sampler_output.sampled_token_ids
    if sampled.ndim == 1:
        sampled = sampled.unsqueeze(-1)

    valid_sampled_token_ids: list[list[int]] = []
    for row in sampled.tolist():
        valid_sampled_token_ids.append([int(tok) for tok in row if int(tok) != -1])
    return valid_sampled_token_ids


def _beam_debug_enabled() -> bool:
    return os.environ.get("VLLM_GR_BEAM_DEBUG", "0") == "1"


def _log_beam_debug(message: str, *args: Any) -> None:
    if not _beam_debug_enabled():
        return

    if args:
        try:
            rendered = message % args
        except Exception:
            rendered = " ".join([message, *[str(arg) for arg in args]])
    else:
        rendered = message

    rendered = "[beam-debug] " + rendered
    logger.warning(rendered)
    try:
        print(rendered, file=sys.stderr, flush=True)
    except Exception:
        pass


def _beam_trace_enabled() -> bool:
    """Return whether verbose per-step beam trace logs are enabled."""
    return os.environ.get("VLLM_GR_BEAM_TRACE", "0") == "1"


def _log_beam_trace(message: str, *args: Any) -> None:
    """Emit high-volume beam tracing only when explicitly requested."""
    if not _beam_trace_enabled():
        return
    _log_beam_debug(message, *args)


# ---------------------------------------------------------------------------
# Factory functions (module-level so each captures its own original via
# closure argument, avoiding late-binding bugs when patching multiple runners).
# ---------------------------------------------------------------------------


def make_patched_execute_model(_orig_execute_model):
    """Wrap execute_model to publish beam-request context for the run.

    Shared by GPU and NPU runners.
    """
    from vllm_gr.v1.attention.backends.beam_attn import BEAM_DATA_VAR

    @wraps(_orig_execute_model)
    def patched_execute_model(self, scheduler_output, *args, **kwargs):
        self._current_beam_data = getattr(scheduler_output, "beam_data", {})
        beam_info = {"beam_data": self._current_beam_data, "req_ids": []}
        token = BEAM_DATA_VAR.set(beam_info)
        self._current_beam_info = beam_info
        try:
            return _orig_execute_model(self, scheduler_output, *args, **kwargs)
        finally:
            self._current_beam_data = {}
            self._current_beam_info = None
            BEAM_DATA_VAR.reset(token)

    return patched_execute_model


def make_patched_execute_model_v2(_orig_execute_model):
    """Wrap V2 ``execute_model`` without clearing beam state too early.

    ModelRunner V2 splits execution into ``execute_model()`` and
    ``sample_tokens()``. Unlike V1, sampling does not happen before
    ``execute_model`` returns, so clearing ``_current_beam_data`` in the
    execute-model wrapper drops the beam context before sampling/logprob
    handling begins. Keep the context alive until the paired
    ``sample_tokens()`` wrapper finishes.
    """
    from vllm_gr.v1.attention.backends.beam_attn import BEAM_DATA_VAR

    @wraps(_orig_execute_model)
    def patched_execute_model(self, scheduler_output, *args, **kwargs):
        self._current_beam_data = getattr(scheduler_output, "beam_data", {})
        beam_info = {"beam_data": self._current_beam_data, "req_ids": []}
        token = BEAM_DATA_VAR.set(beam_info)
        self._current_beam_info = beam_info
        self._current_beam_var_token = token
        return _orig_execute_model(self, scheduler_output, *args, **kwargs)

    return patched_execute_model


def make_patched_prepare_inputs_gpu(_orig_prepare_inputs):
    """GPU ``_prepare_inputs`` wrapper: ``(scheduler_output, num_scheduled_tokens)``
    returning ``(logits_indices, spec_decode_metadata)``."""

    @wraps(_orig_prepare_inputs)
    def patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        logits_indices, spec_decode_metadata = _orig_prepare_inputs(
            self, scheduler_output, num_scheduled_tokens
        )

        # Job A: the GPU attention builder runs after _prepare_inputs returns, so
        # populating req_ids here (post-orig) is in time.
        if getattr(self, "_current_beam_info", None) is not None:
            self._current_beam_info["req_ids"] = self.input_batch.req_ids

        beam_data = getattr(scheduler_output, "beam_data", {})
        if not beam_data:
            return logits_indices, spec_decode_metadata

        new_logits_indices = _beam_remap_inputs(self, scheduler_output, num_scheduled_tokens)
        return (
            torch.tensor(new_logits_indices, dtype=torch.int32, device=self.device),
            spec_decode_metadata,
        )

    return patched_prepare_inputs


def make_patched_bookkeeping_sync_gpu(_orig_bookkeeping_sync):
    """GPU ``_bookkeeping_sync`` wrapper: returns the 7-tuple including
    ``num_nans_in_logits``."""

    @wraps(_orig_bookkeeping_sync)
    def patched_bookkeeping_sync(self, scheduler_output, *args, **kwargs):
        if hasattr(self, "_beam_logprobs_cache") and self._beam_logprobs_cache:
            if hasattr(scheduler_output, "finished_req_ids"):
                for req_id in scheduler_output.finished_req_ids:
                    self._beam_logprobs_cache.pop(req_id, None)

        beam_data = getattr(scheduler_output, "beam_data", {})
        if not beam_data or not hasattr(scheduler_output, "_gr_request_row_configs"):
            return _orig_bookkeeping_sync(self, scheduler_output, *args, **kwargs)

        request_row_configs = scheduler_output._gr_request_row_configs
        orig_input_batch = object.__getattribute__(self.input_batch, "_obj")
        beam_state_snapshot = _snapshot_beam_runner_state(
            orig_input_batch,
            request_row_configs,
            beam_data,
            self.requests,
        )

        try:
            res = _orig_bookkeeping_sync(self, scheduler_output, *args, **kwargs)
        finally:
            _restore_beam_runner_state(orig_input_batch, self.requests, beam_state_snapshot)
            self.input_batch = orig_input_batch

        (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = res

        sampler_output = kwargs.get("sampler_output") or (args[0] if args else None)
        collapsed_req_ids, collapsed_idx, n = _beam_collapse_bookkeeping(
            self, orig_input_batch, request_row_configs, beam_data, sampler_output
        )
        collapsed_logprobs_lists = None
        if sampler_output is not None and sampler_output.logprobs_tensors is not None:
            # After beam collapse we have exactly one flattened logprob row per
            # logical request. The original cu_num_generated_tokens still
            # describes the expanded per-beam sampler rows, so reusing it here
            # makes downstream slice_request() index into the collapsed tensor
            # with stale offsets and can attach another request's beam block.
            collapsed_logprobs_lists = sampler_output.logprobs_tensors.tolists(list(range(n)))
        collapsed_valid_sampled_token_ids = _sampler_output_to_valid_token_lists(sampler_output)
        _log_beam_trace(
            "gpu collapse reqs=%s rows=%s sampled_shape=%s logprobs_shape=%s cu_gen=%s",
            collapsed_req_ids,
            [cfg[2] for cfg in request_row_configs],
            tuple(sampler_output.sampled_token_ids.shape)
            if sampler_output is not None and sampler_output.sampled_token_ids is not None
            else None,
            tuple(sampler_output.logprobs_tensors.logprobs.shape)
            if sampler_output is not None
            and sampler_output.logprobs_tensors is not None
            and sampler_output.logprobs_tensors.logprobs is not None
            else None,
            (
                sampler_output.logprobs_tensors.cu_num_generated_tokens
                if sampler_output is not None and sampler_output.logprobs_tensors is not None
                else None
            ),
        )
        if collapsed_logprobs_lists is not None:
            _log_beam_trace(
                "gpu collapsed lists token_ids_shape=%s logprobs_shape=%s ranks_shape=%s cu_gen=%s",
                collapsed_logprobs_lists.logprob_token_ids.shape,
                collapsed_logprobs_lists.logprobs.shape,
                collapsed_logprobs_lists.sampled_token_ranks.shape,
                collapsed_logprobs_lists.cu_num_generated_tokens,
            )

        return (
            num_nans_in_logits,
            collapsed_logprobs_lists,
            collapsed_valid_sampled_token_ids
            if collapsed_valid_sampled_token_ids
            else [[] for _ in range(n)],
            prompt_logprobs_dict,
            collapsed_req_ids,
            collapsed_idx,
            invalid_req_indices,
        )

    return patched_bookkeeping_sync


def make_patched_sample_gpu(_orig_sample):
    """GPU ``sample`` wrapper for merged beam decode on vLLM 0.22.1.

    vLLM 0.22.1 adds ``SamplerOutput.num_sampled`` and expects it to encode
    the number of sampled logits rows per logical request. Beam remapping can
    expand one logical request into multiple logits rows, so we derive
    ``num_sampled`` from ``cu_num_logits`` when beam_data is active.
    """

    @wraps(_orig_sample)
    def patched_sample(self, hidden_states, input_batch, grammar_output):
        sampler_output, num_sampled, num_rejected = _orig_sample(
            self, hidden_states, input_batch, grammar_output
        )

        beam_data = getattr(self, "_current_beam_data", {})
        if not beam_data:
            return sampler_output, num_sampled, num_rejected

        request_row_configs = getattr(input_batch, "_gr_request_row_configs", None)
        if request_row_configs is not None:
            collapsed_req_ids, _collapsed_idx, n = _beam_collapse_bookkeeping(
                self,
                input_batch,
                request_row_configs,
                beam_data,
                sampler_output,
            )
            per_req_num_sampled = getattr(
                input_batch,
                "_gr_num_sampled_per_req",
                [1] * n,
            )
            logical_num_sampled = torch.tensor(
                per_req_num_sampled,
                dtype=num_sampled.dtype,
                device=num_sampled.device,
            )
            sampler_output.num_sampled = logical_num_sampled
            num_sampled = logical_num_sampled
            num_rejected = torch.zeros_like(logical_num_sampled)
            _log_beam_trace(
                "gpu-v2 collapsed reqs=%s rows=%s sampled_shape=%s logprobs_shape=%s cu_gen=%s",
                collapsed_req_ids,
                [cfg[2] for cfg in request_row_configs],
                tuple(sampler_output.sampled_token_ids.shape)
                if sampler_output.sampled_token_ids is not None
                else None,
                tuple(sampler_output.logprobs_tensors.logprobs.shape)
                if sampler_output.logprobs_tensors is not None
                and sampler_output.logprobs_tensors.logprobs is not None
                else None,
                (
                    sampler_output.logprobs_tensors.cu_num_generated_tokens
                    if sampler_output.logprobs_tensors is not None
                    else None
                ),
            )
            return sampler_output, num_sampled, num_rejected

        cu_num_logits = getattr(input_batch, "cu_num_logits", None)
        if cu_num_logits is None or cu_num_logits.numel() <= 1:
            return sampler_output, num_sampled, num_rejected

        logical_num_sampled = (cu_num_logits[1:] - cu_num_logits[:-1]).to(
            dtype=num_sampled.dtype, device=num_sampled.device
        )
        sampler_output.num_sampled = logical_num_sampled
        num_sampled = logical_num_sampled
        num_rejected = torch.zeros_like(logical_num_sampled)
        return sampler_output, num_sampled, num_rejected

    return patched_sample


def make_patched_prepare_inputs_gpu_v2(_orig_prepare_inputs):
    """GPU ModelRunner V2 ``prepare_inputs`` wrapper.

    vLLM 0.22.1 can switch the GPU worker over to ModelRunner V2
    (``vllm.v1.worker.gpu.model_runner.GPUModelRunner``). The old GR beam patch
    only targeted the V1 runner, which meant the real runtime path on newer
    builds never published request ids to the beam attention context and never
    emitted any beam-debug logs. Keep the patch narrow by reusing the existing
    execute_model/sample wrappers and only bridging the request-id publication
    point that moved from ``_prepare_inputs`` to ``prepare_inputs`` in V2.
    """

    @wraps(_orig_prepare_inputs)
    def patched_prepare_inputs(self, scheduler_output, batch_desc):
        input_batch = _orig_prepare_inputs(self, scheduler_output, batch_desc)

        if getattr(self, "_current_beam_info", None) is not None:
            self._current_beam_info["req_ids"] = input_batch.req_ids

        beam_data = getattr(scheduler_output, "beam_data", {})
        if beam_data:
            _beam_remap_inputs_v2(self, input_batch, scheduler_output)
            _log_beam_trace(
                "gpu-v2 prepare_inputs req_ids=%s beam_reqs=%s logits=%s cu=%s",
                input_batch.req_ids,
                list(beam_data.keys()),
                input_batch.logits_indices,
                input_batch.cu_num_logits_np.tolist(),
            )

        return input_batch

    return patched_prepare_inputs


def make_patched_sample_tokens_gpu_v2(_orig_sample_tokens):
    """GPU ModelRunner V2 ``sample_tokens`` wrapper.

    This wrapper is intentionally light-touch: it does not invent V2-specific
    beam behavior, it only surfaces a positive signal that the actual 0.22.1
    runtime path is executing under our GR beam patch so follow-up debugging and
    any remaining fixes can target the correct runner.
    """

    @wraps(_orig_sample_tokens)
    def patched_sample_tokens(self, grammar_output):
        execute_model_state = getattr(self, "execute_model_state", None)
        input_batch = getattr(execute_model_state, "input_batch", None)
        if input_batch is not None and getattr(self, "_current_beam_data", {}):
            _log_beam_trace(
                "gpu-v2 sample_tokens req_ids=%s logits=%s",
                getattr(input_batch, "req_ids", None),
                getattr(input_batch, "logits_indices", None),
            )
        try:
            return _orig_sample_tokens(self, grammar_output)
        finally:
            from vllm_gr.v1.attention.backends.beam_attn import BEAM_DATA_VAR

            token = getattr(self, "_current_beam_var_token", None)
            if token is not None:
                BEAM_DATA_VAR.reset(token)
                self._current_beam_var_token = None
            self._current_beam_data = {}
            self._current_beam_info = None

    return patched_sample_tokens


def make_patched_prepare_inputs_npu(_orig_prepare_inputs):
    """NPU ``_prepare_inputs`` wrapper.

    Signature: ``(scheduler_output, intermediate_tensors=None)``.

    vLLM-Ascend changed the return shape of ``_prepare_inputs`` between
    releases:

    - ``0.14.x`` returned a 7-tuple whose 5th element is ``logits_indices``.
    - ``0.22.1rc1`` returns a 4-tuple whose 1st element is ``logits_indices``.

    Two NPU-specific differences vs the GPU wrapper:

    - The NPU attention metadata builder runs *inside* the original
      ``_prepare_inputs``, so Job A (publishing ``req_ids`` to the contextvar)
      must happen *before* delegating, not after.
    - ``_prepare_inputs`` is called from ``execute_model``, while sampling and
      ``_bookkeeping_sync`` run in a separate ``sample_tokens`` call. If an
      early return skips bookkeeping, a stale ``InputBatchProxy`` could survive
      to the next step, so we defensively un-proxy at entry.

    The per-beam RoPE position overwrite needs no extra device copy: NPU
    ``self.positions`` is a ``CpuGpuBuffer`` and ``_overwrite_position_ids``
    writes the ``.gpu`` (device) tensor in place, which ``_preprocess`` reads.
    The physical ``slot_mapping`` is materialized from linear positions inside
    the original before the overwrite, so it is unaffected.
    """

    @wraps(_orig_prepare_inputs)
    def patched_prepare_inputs(self, scheduler_output, intermediate_tensors=None):
        # Drop a stale proxy left over from a prior step (see docstring).
        if isinstance(self.input_batch, InputBatchProxy):
            self.input_batch = object.__getattribute__(self.input_batch, "_obj")

        # Job A: publish req_ids BEFORE the original builds attention metadata.
        if getattr(self, "_current_beam_info", None) is not None:
            self._current_beam_info["req_ids"] = self.input_batch.req_ids

        out = _orig_prepare_inputs(self, scheduler_output, intermediate_tensors)

        beam_data = getattr(scheduler_output, "beam_data", {})
        if not beam_data:
            return out

        if not isinstance(out, tuple):
            raise TypeError(f"Unexpected NPU _prepare_inputs return type: {type(out).__name__}")

        if len(out) == 7:
            num_scheduled_tokens = out[1]
            logits_index = 4
        elif len(out) == 4:
            req_ids = list(getattr(self.input_batch, "req_ids", []))
            num_scheduled_tokens = [
                scheduler_output.num_scheduled_tokens[req_id] for req_id in req_ids
            ]
            logits_index = 0
        else:
            raise ValueError(f"Unexpected NPU _prepare_inputs return tuple length: {len(out)}")

        new_logits_indices = _beam_remap_inputs(self, scheduler_output, num_scheduled_tokens)
        logits_indices = torch.tensor(new_logits_indices, dtype=torch.int32, device=self.device)

        out_list = list(out)
        out_list[logits_index] = logits_indices
        return tuple(out_list)

    return patched_prepare_inputs


def make_patched_bookkeeping_sync_npu(_orig_bookkeeping_sync):
    """NPU ``_bookkeeping_sync`` wrapper: returns the 6-tuple (no
    ``num_nans_in_logits``).

    NPU calls ``_bookkeeping_sync(scheduler_output, sampler_output, logits,
    hidden_states, num_scheduled_tokens, spec_decode_metadata)`` positionally,
    so ``sampler_output`` is ``args[0]``.
    """

    @wraps(_orig_bookkeeping_sync)
    def patched_bookkeeping_sync(self, scheduler_output, *args, **kwargs):
        if hasattr(self, "_beam_logprobs_cache") and self._beam_logprobs_cache:
            if hasattr(scheduler_output, "finished_req_ids"):
                for req_id in scheduler_output.finished_req_ids:
                    self._beam_logprobs_cache.pop(req_id, None)

        beam_data = getattr(scheduler_output, "beam_data", {})
        if not beam_data or not hasattr(scheduler_output, "_gr_request_row_configs"):
            res = _orig_bookkeeping_sync(self, scheduler_output, *args, **kwargs)
            sampler_output = kwargs.get("sampler_output") or (args[0] if args else None)
            return res

        request_row_configs = scheduler_output._gr_request_row_configs
        orig_input_batch = object.__getattribute__(self.input_batch, "_obj")
        beam_state_snapshot = _snapshot_beam_runner_state(
            orig_input_batch,
            request_row_configs,
            beam_data,
            self.requests,
        )

        try:
            res = _orig_bookkeeping_sync(self, scheduler_output, *args, **kwargs)
        finally:
            _restore_beam_runner_state(orig_input_batch, self.requests, beam_state_snapshot)
            self.input_batch = orig_input_batch

        (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = res

        sampler_output = kwargs.get("sampler_output") or (args[0] if args else None)
        collapsed_req_ids, collapsed_idx, n = _beam_collapse_bookkeeping(
            self, orig_input_batch, request_row_configs, beam_data, sampler_output
        )
        collapsed_logprobs_lists = None
        if sampler_output is not None and sampler_output.logprobs_tensors is not None:
            # After beam collapse we have exactly one flattened logprob row per
            # logical request. The original cu_num_generated_tokens still
            # describes the expanded per-beam sampler rows, so reusing it here
            # makes downstream slice_request() index into the collapsed tensor
            # with stale offsets and can attach another request's beam block.
            collapsed_logprobs_lists = sampler_output.logprobs_tensors.tolists(list(range(n)))
        collapsed_valid_sampled_token_ids = _sampler_output_to_valid_token_lists(sampler_output)
        _log_beam_trace(
            "npu collapse reqs=%s rows=%s sampled_shape=%s logprobs_shape=%s cu_gen=%s",
            collapsed_req_ids,
            [cfg[2] for cfg in request_row_configs],
            tuple(sampler_output.sampled_token_ids.shape)
            if sampler_output is not None and sampler_output.sampled_token_ids is not None
            else None,
            tuple(sampler_output.logprobs_tensors.logprobs.shape)
            if sampler_output is not None
            and sampler_output.logprobs_tensors is not None
            and sampler_output.logprobs_tensors.logprobs is not None
            else None,
            (
                sampler_output.logprobs_tensors.cu_num_generated_tokens
                if sampler_output is not None and sampler_output.logprobs_tensors is not None
                else None
            ),
        )
        if collapsed_logprobs_lists is not None:
            _log_beam_trace(
                "npu collapsed lists token_ids_shape=%s logprobs_shape=%s ranks_shape=%s cu_gen=%s",
                collapsed_logprobs_lists.logprob_token_ids.shape,
                collapsed_logprobs_lists.logprobs.shape,
                collapsed_logprobs_lists.sampled_token_ranks.shape,
                collapsed_logprobs_lists.cu_num_generated_tokens,
            )

        return (
            collapsed_logprobs_lists,
            collapsed_valid_sampled_token_ids
            if collapsed_valid_sampled_token_ids
            else [[] for _ in range(n)],
            prompt_logprobs_dict,
            collapsed_req_ids,
            collapsed_idx,
            invalid_req_indices,
        )

    return patched_bookkeeping_sync


def apply_worker_patches():
    """Applies runtime overrides onto GPUModelRunner and NPUModelRunner execution entries."""
    from vllm_gr.utils.npu_adapter import is_npu_available

    _log_beam_debug("apply_worker_patches called")

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1
    except ImportError:
        logger.warning("Failed to import GPUModelRunner segments, skipping GPU patches.")
        GPUModelRunnerV1 = None

    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    except ImportError:
        GPUModelRunnerV2 = None

    npu_available = is_npu_available()
    if npu_available:
        NPUModelRunner = None
        import_errors: list[tuple[str, Exception]] = []
        for module_path in (
            "vllm_ascend.worker.model_runner_v1",
            "vllm_ascend.worker.worker",
        ):
            try:
                module = __import__(module_path, fromlist=["NPUModelRunner"])
                NPUModelRunner = getattr(module, "NPUModelRunner")
                break
            except (ImportError, AttributeError) as exc:
                import_errors.append((module_path, exc))
        if NPUModelRunner is None:
            details = "; ".join(
                f"{module_path}: {type(exc).__name__}: {exc}" for module_path, exc in import_errors
            )
            logger.warning(
                "Failed to import NPUModelRunner, skipping NPU patches. Details: %s",
                details,
            )
    else:
        NPUModelRunner = None

    # Return early if nothing to patch
    if GPUModelRunnerV1 is None and GPUModelRunnerV2 is None and NPUModelRunner is None:
        logger.warning("No supported model runner found for patching.")
        return

    # Track which runners have already been patched
    runners_to_patch = []
    if GPUModelRunnerV1 is not None and not getattr(
        GPUModelRunnerV1, "_patched_for_beam_pos_ids", False
    ):
        runners_to_patch.append(("GPUModelRunnerV1", GPUModelRunnerV1))
    if GPUModelRunnerV2 is not None and not getattr(
        GPUModelRunnerV2, "_patched_for_beam_pos_ids", False
    ):
        runners_to_patch.append(("GPUModelRunnerV2", GPUModelRunnerV2))
    if NPUModelRunner is not None and not getattr(
        NPUModelRunner, "_patched_for_beam_pos_ids", False
    ):
        runners_to_patch.append(("NPUModelRunner", NPUModelRunner))

    if not runners_to_patch:
        _log_beam_debug("apply_worker_patches found no new runners to patch")
        return

    # Apply patches to each runner
    for runner_name, runner_class in runners_to_patch:
        if runner_name == "GPUModelRunnerV1":
            runner_class.execute_model = make_patched_execute_model(runner_class.execute_model)
            runner_class._prepare_inputs = make_patched_prepare_inputs_gpu(
                runner_class._prepare_inputs
            )
            if hasattr(runner_class, "sample"):
                runner_class.sample = make_patched_sample_gpu(runner_class.sample)
            elif hasattr(runner_class, "_sample"):
                runner_class._sample = make_patched_sample_gpu(runner_class._sample)
            runner_class._bookkeeping_sync = make_patched_bookkeeping_sync_gpu(
                runner_class._bookkeeping_sync
            )
        elif runner_name == "GPUModelRunnerV2":
            runner_class.execute_model = make_patched_execute_model_v2(
                runner_class.execute_model
            )
            runner_class.prepare_inputs = make_patched_prepare_inputs_gpu_v2(
                runner_class.prepare_inputs
            )
            runner_class.sample = make_patched_sample_gpu(runner_class.sample)
            runner_class.sample_tokens = make_patched_sample_tokens_gpu_v2(
                runner_class.sample_tokens
            )
        elif runner_name == "NPUModelRunner":
            runner_class.execute_model = make_patched_execute_model(runner_class.execute_model)
            runner_class._prepare_inputs = make_patched_prepare_inputs_npu(
                runner_class._prepare_inputs
            )
            runner_class._bookkeeping_sync = make_patched_bookkeeping_sync_npu(
                runner_class._bookkeeping_sync
            )

        runner_class._patched_for_beam_pos_ids = True
        logger.info("%s hooks patched successfully.", runner_name)
        _log_beam_debug("%s hooks patched successfully", runner_name)
