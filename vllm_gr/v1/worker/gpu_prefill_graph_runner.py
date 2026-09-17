# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA Graph Runner for GPU prefill execution."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_gr.utils.npu_adapter import is_cuda_available
from vllm_gr.v1.worker.prefill_graph_common import (
    ENV_ENABLE_PREFILL_GRAPH,
    BasePrefillGraphRunner,
    is_prefill_graph_enabled,
)

logger = init_logger(__name__)
GPU_PREFILL_CAPTURE_NUM_REQS = (1,)


class GPUPrefillGraphRunner(BasePrefillGraphRunner):
    """Manages CUDA graph capture and replay on NVIDIA GPUs."""

    def __init__(
        self,
        model_runner: Any,
        buckets: list[int] | None = None,
    ) -> None:
        super().__init__(
            model_runner=model_runner,
            buckets=buckets,
        )
        # CPU-only diagnostic for the most recent replay attempt; never reads
        # device state or changes the stream ordering.
        self.last_replay_reason: str | None = None
        # One-shot direct-staging token: the bucket whose static input
        # buffers try_direct_stage_prefill() staged for the CURRENT step.
        # Reset at every _prepare_inputs (see reset_direct_staging) and
        # consumed by the next execute_model_prefill, so a replay can never
        # observe a staging left over from an earlier step.
        self._direct_staged_bucket: int | None = None
        # Cumulative count of fast-path stagings (regression diagnostics).
        self.direct_stage_count: int = 0
        logger.info(
            "GPUPrefillGraphRunner initialized on device %s with buckets %s",
            self.device,
            self.buckets,
        )

    def reset_direct_staging(self) -> None:
        """Invalidate the previous step's direct-staging token.

        Called at the START of every ``_prepare_inputs`` (fast or generic)
        so a replay can never consume a staging left over from an earlier
        step (e.g. a fast-staged batch that fell back to eager upstream of
        the replay).
        """
        self._direct_staged_bucket = None

    def capture_graphs(self, runner: Any) -> None:
        """Capture one-request CUDA graphs for every configured prefill bucket."""
        if not is_prefill_graph_enabled():
            logger.info(
                "Prefill CUDA Graph capture is disabled via environment variable (%s=0).",
                ENV_ENABLE_PREFILL_GRAPH,
            )
            return

        if not is_cuda_available():
            logger.info("CUDA not available; skipping CUDA graph capture for prefill.")
            return

        from tqdm import tqdm
        from vllm.config import CUDAGraphMode
        from vllm.distributed.device_communicators.pynccl_allocator import (
            set_graph_pool_id,
        )
        from vllm.forward_context import set_forward_context
        from vllm.platforms import current_platform

        logger.info(
            "Beginning Prefill CUDA graph capture for buckets: %s (batch sizes: %s)",
            self.buckets,
            GPU_PREFILL_CAPTURE_NUM_REQS,
        )

        pool = current_platform.get_global_graph_pool()
        if pool is not None:
            set_graph_pool_id(pool)
        else:
            set_graph_pool_id(current_platform.graph_pool_handle())

        sorted_buckets = sorted(self.buckets, reverse=True)

        for bucket in tqdm(sorted_buckets, desc="Capturing Prefill CUDA graphs (FULL)"):
            buf = self.buffers[bucket]

            # Adjust positions_buf shape for mrope / xdrope if needed
            if getattr(runner, "uses_mrope", False):
                buf.positions_buf = torch.zeros(
                    (3, bucket), dtype=torch.int64, device=runner.device
                )
                buf.positions_buf[:] = torch.arange(bucket, dtype=torch.int64, device=runner.device)
            elif getattr(runner, "uses_xdrope_dim", 0) > 0:
                buf.positions_buf = torch.zeros(
                    (2, bucket), dtype=torch.int64, device=runner.device
                )
                buf.positions_buf[:] = torch.arange(bucket, dtype=torch.int64, device=runner.device)
            else:
                buf.positions_buf = torch.arange(bucket, dtype=torch.int64, device=runner.device)

            buf.input_ids_buf.fill_(buf.pad_token_id)
            buf.slot_mapping_buf.fill_(-1)
            slot_mappings_by_group = {0: buf.slot_mapping_buf}
            # Per-layer mapping for vllm's unified KV-cache update op. Point
            # every attention layer at the static buffer: the graph bakes the
            # pointer and execute_model_prefill() refreshes the contents
            # before each replay. Without this the op silently skips KV
            # writes and replays read an empty KV cache.
            slot_mappings_by_layer: dict[str, torch.Tensor] = {}
            kv_cache_config = getattr(runner, "kv_cache_config", None)
            if kv_cache_config is not None and len(kv_cache_config.kv_cache_groups) > 0:
                for layer_name in kv_cache_config.kv_cache_groups[0].layer_names:
                    slot_mappings_by_layer[layer_name] = buf.slot_mapping_buf

            for num_reqs in GPU_PREFILL_CAPTURE_NUM_REQS:
                if bucket < num_reqs:
                    continue

                try:
                    # Distribute bucket tokens evenly across num_reqs for capture
                    base_len = bucket // num_reqs
                    rem = bucket % num_reqs
                    req_lens = [base_len + (1 if i < rem else 0) for i in range(num_reqs)]

                    # Set up seq_lens and query_start_loc for (bucket, num_reqs)
                    if hasattr(runner, "seq_lens") and hasattr(runner, "optimistic_seq_lens_cpu"):
                        for i, rlen in enumerate(req_lens):
                            runner.optimistic_seq_lens_cpu[i] = rlen
                        runner.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
                        runner.seq_lens.copy_(runner.optimistic_seq_lens_cpu, non_blocking=True)

                    if hasattr(runner, "query_start_loc"):
                        runner.query_start_loc.np[0] = 0
                        curr = 0
                        for i, rlen in enumerate(req_lens):
                            curr += rlen
                            runner.query_start_loc.np[i + 1] = curr
                        runner.query_start_loc.np[num_reqs + 1 :] = 0
                        runner.query_start_loc.copy_to_gpu()

                    # Build attention metadata for (bucket, num_reqs)
                    attn_metadata, _ = runner._build_attention_metadata(
                        num_tokens=bucket,
                        num_tokens_padded=bucket,
                        num_reqs=num_reqs,
                        max_query_len=bucket,
                        ubatch_slices=None,
                        for_cudagraph_capture=True,
                        slot_mappings=slot_mappings_by_group,
                    )

                    # The captured flash-attention call bakes max_seqlen_k from
                    # this metadata scalar; the replay then admits requests
                    # whose KV length (num_computed + scheduled) fits within
                    # 2x the bucket. Requests beyond that bound fall back to
                    # eager (checked in execute_model_prefill).
                    bound = 2 * bucket
                    self._apply_max_seq_len_bound(attn_metadata, bound)
                    buf.max_seq_len_bounds[num_reqs] = bound

                    # Rebind the block table, seq_lens, and query_start_loc
                    # to static per-bucket buffers: a captured graph bakes the
                    # pointer address, so the tensor at that address must stay
                    # stable for the lifetime of the graph. The runner's live
                    # per-batch tensors are overwritten (or reallocated) for
                    # the next batch, so the graph must never bake them.
                    bt_metas = (
                        list(attn_metadata.values())
                        if isinstance(attn_metadata, dict)
                        else [attn_metadata]
                    )
                    for meta in bt_metas:
                        if not hasattr(meta, "block_table"):
                            continue
                        bt_shape = meta.block_table.shape
                        # Pre-allocated at init from the model config; re-size
                        # only if the metadata's width exceeds it (should not
                        # happen — config-derived widths match).
                        if (
                            buf.block_table_buf is None
                            or bt_shape[1] > buf.block_table_buf.shape[1]
                        ):
                            buf.block_table_buf = torch.zeros(
                                (GPU_PREFILL_CAPTURE_NUM_REQS[-1], bt_shape[1]),
                                dtype=meta.block_table.dtype,
                                device=runner.device,
                            )
                        meta.block_table = buf.block_table_buf[: bt_shape[0], : bt_shape[1]]
                        # Rebind seq_lens / query_start_loc to the static
                        # buffers pre-allocated at init: the replay refreshes
                        # them before every execution (see execute_model_prefill).
                        if hasattr(meta, "seq_lens") and meta.seq_lens is not None:
                            static_seq_lens = buf.seq_lens_buf[: meta.seq_lens.shape[0]]
                            static_seq_lens.copy_(meta.seq_lens)
                            meta.seq_lens = static_seq_lens
                        if hasattr(meta, "query_start_loc") and meta.query_start_loc is not None:
                            static_query_start_loc = buf.query_start_loc_buf[
                                : meta.query_start_loc.shape[0]
                            ]
                            static_query_start_loc.copy_(meta.query_start_loc)
                            meta.query_start_loc = static_query_start_loc

                    if hasattr(attn_metadata, "is_prefill_graph_exec"):
                        attn_metadata.is_prefill_graph_exec = True

                    # Warmup passes
                    for _ in range(2):
                        with set_forward_context(
                            attn_metadata,
                            runner.vllm_config,
                            num_tokens=bucket,
                            cudagraph_runtime_mode=CUDAGraphMode.NONE,
                            slot_mapping=slot_mappings_by_layer,
                        ):
                            runner.model(
                                input_ids=buf.input_ids_buf,
                                positions=buf.positions_buf,
                            )
                    torch.accelerator.synchronize()

                    # Capture CUDA graph
                    graph = torch.cuda.CUDAGraph()
                    with (
                        torch.cuda.graph(graph, pool=pool, stream=torch.cuda.current_stream()),
                        set_forward_context(
                            attn_metadata,
                            runner.vllm_config,
                            num_tokens=bucket,
                            cudagraph_runtime_mode=CUDAGraphMode.NONE,
                            slot_mapping=slot_mappings_by_layer,
                        ),
                    ):
                        output = runner.model(
                            input_ids=buf.input_ids_buf,
                            positions=buf.positions_buf,
                        )

                    torch.accelerator.synchronize()
                    buf.captured_graphs[num_reqs] = graph
                    buf.output_bufs[num_reqs] = output
                    buf.static_attn_metadatas[num_reqs] = attn_metadata
                    # Snapshot the capture-time positions (arange for the
                    # plain-RoPE case). A fresh exact-bucket single-request
                    # prefill replays with exactly these values, so the
                    # direct-staging path can skip re-copying positions (and
                    # restore them after a differently-shaped replay).
                    buf.positions_template = buf.positions_buf.clone()
                    buf.capture_invariants_fresh = True

                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Failed to capture Prefill CUDA graph for bucket %d (num_reqs=%d): %r.",
                        bucket,
                        num_reqs,
                        e,
                    )

        captured_summary = {
            b: list(self.buffers[b].captured_graphs.keys())
            for b in self.buckets
            if self.buffers[b].captured_graphs
        }
        msg = f"Prefill CUDA Graphs capture completed. Captured (bucket -> req_counts): {captured_summary}"
        logger.info(msg)

    def has_captured_graph(self, bucket: int, num_reqs: int = 1) -> bool:
        """Returns True if a Graph has been captured for this bucket and num_reqs."""
        buf = self.buffers.get(bucket)
        return buf is not None and buf.has_captured_graph(num_reqs)

    @staticmethod
    def _apply_max_seq_len_bound(attn_metadata: Any, bound: int) -> None:
        """Bind the per-layer max_seq_len scalar to the replay KV-length bound.

        The captured flash-attention call bakes max_seqlen_k from this scalar;
        the runner only replays batches whose per-request KV lengths stay
        within ``bound`` (2x the bucket). 0 keeps the dummy value.
        """
        if not bound:
            return
        if isinstance(attn_metadata, dict):
            for meta in attn_metadata.values():
                if hasattr(meta, "max_seq_len"):
                    meta.max_seq_len = bound
        elif hasattr(attn_metadata, "max_seq_len"):
            attn_metadata.max_seq_len = bound

    def _max_scheduled_kvlen(self, num_scheduled_tokens: list[int] | dict[Any, int]) -> int | None:
        """Max per-request KV length (num_computed + scheduled) in the batch.

        Resolves each scheduled request ID through the InputBatch row mapping
        (num_scheduled_tokens is req-id-keyed). Returns None when the
        resolution is unavailable (legacy list form or missing arrays).
        """
        if not isinstance(num_scheduled_tokens, dict):
            return None
        input_batch = getattr(self.model_runner, "input_batch", None)
        if input_batch is None:
            return None
        req_id_to_index = getattr(input_batch, "req_id_to_index", None)
        computed = getattr(input_batch, "num_computed_tokens_cpu", None)
        if req_id_to_index is None or computed is None:
            return None
        max_kvlen = 0
        for req_id, count in num_scheduled_tokens.items():
            row = req_id_to_index.get(req_id)
            if row is None:
                return None
            max_kvlen = max(max_kvlen, int(computed[row]) + int(count))
        return max_kvlen

    def try_direct_stage_prefill(
        self,
        runner: Any,
        scheduler_output: Any,
    ) -> tuple[torch.Tensor, None] | None:
        """Specialized ``_prepare_inputs`` for a GR V1 single-request prefill.

        A GR V1 asynchronous-Prefill stage with B=1 whose scheduled token
        count exactly fills its bucket has a fully static input layout:
        positions are the capture-time arange, query_start_loc is
        [0, bucket], seq_lens is [bucket], and the slot mapping is the
        within-block offset template expanded over the request's kernel
        block ids. Stage every live buffer the rest of the step consumes
        (attention-metadata build, eager fallback, logits indexing) AND
        the captured graph's static input buffers in one pass, skipping
        the generic path's cumsum/arange/index_select machinery and its
        triton slot-mapping launch.

        Returns ``(logits_indices, None)`` when the fast path ran, else
        ``None`` (the caller must fall back to the generic
        ``_prepare_inputs``). When the fast path runs, the replay is
        guaranteed to accept the batch, so the eagerly-populated live
        tensors are only a defensive fallback.
        """
        stages = getattr(scheduler_output, "gr_stage_metadata", None)
        if not stages or len(stages) != 1:
            return None
        from vllm_gr.v1.engine.beam_search_v1 import GRStageKind

        if next(iter(stages.values())).kind is not GRStageKind.PREFILL:
            return None
        # Legacy beam batches keep their own _prepare_inputs post-processing
        # (_beam_remap_inputs / CUDA prefill constraint preparation).
        if getattr(scheduler_output, "beam_data", None):
            return None
        if getattr(runner.model_config, "enforce_eager", False):
            return None

        input_batch = getattr(runner, "input_batch", None)
        if input_batch is None or input_batch.num_reqs != 1:
            return None
        sched = getattr(scheduler_output, "num_scheduled_tokens", None)
        if not isinstance(sched, dict) or len(sched) != 1:
            return None
        ((req_id, count),) = sched.items()
        if input_batch.req_id_to_index.get(req_id) != 0:
            return None
        total = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
        if total <= 0 or int(count) != total:
            return None
        bucket = self.match_bucket(total)
        if bucket is None or bucket != total:
            return None
        buf = self.buffers.get(bucket)
        if buf is None or not buf.has_captured_graph(1):
            return None
        if buf.positions_template is None or buf.block_table_buf is None:
            return None
        computed = getattr(input_batch, "num_computed_tokens_cpu", None)
        if computed is None or int(computed[0]) != 0:
            return None

        # Model features the generic path handles but this specialized
        # shape does not model.
        if getattr(runner, "uses_mrope", False) or getattr(runner, "uses_xdrope_dim", 0):
            return None
        if runner.lora_config is not None:
            return None
        if getattr(runner, "num_accepted_tokens_event", None) is not None:
            return None  # Speculative decode keeps its own input contract.
        if getattr(runner, "use_async_spec_decode", False):
            return None
        if getattr(runner, "mamba_prev_last_scheduled_idx", None) is not None:
            return None
        if getattr(runner, "enable_prompt_embeds", False) or input_batch.req_prompt_embeds:
            return None
        if getattr(scheduler_output, "scheduled_spec_decode_tokens", None):
            return None
        # Single-lane topologies only: pipeline ranks exchange intermediate
        # tensors through the eager path, and DP ubatching slices the live
        # buffers differently than this whole-batch staging assumes.
        parallel_config = getattr(runner, "parallel_config", None)
        if int(getattr(parallel_config, "pipeline_parallel_size", 1) or 1) != 1:
            return None
        if getattr(parallel_config, "use_ubatching", False):
            return None
        # Context-parallel ranks interleave KV slots and require per-rank
        # seq lens that this path does not stage.
        if int(getattr(runner, "dcp_world_size", 1) or 1) > 1:
            return None
        kv_groups = getattr(getattr(runner, "kv_cache_config", None), "kv_cache_groups", None)
        if not kv_groups or len(kv_groups) != 1:
            return None
        # A fresh prefill never continues a previous iteration's batch slot
        # (the generic async input_ids path would gather its token from
        # prev_sampled_token_ids instead).
        prev_map = getattr(input_batch, "prev_req_id_to_index", None)
        if prev_map and prev_map.get(req_id, -1) >= 0:
            return None
        req_state = runner.requests.get(req_id)
        if req_state is None:
            return None

        token_ids = getattr(input_batch, "token_ids_cpu_tensor", None)
        if token_ids is None or token_ids.shape[1] < bucket:
            return None
        blk = input_batch.block_table[0]
        kbs = int(getattr(blk, "block_size", 0))
        if kbs <= 0:
            return None
        # Context-parallel block tables interleave KV slots across ranks;
        # this path models the plain slot formula only.
        if (
            int(getattr(blk, "pcp_world_size", 1) or 1) > 1
            or int(getattr(blk, "dcp_world_size", 1) or 1) > 1
            or int(getattr(blk, "cp_kv_cache_interleave_size", 1) or 1) > 1
        ):
            return None
        nblocks = int(blk.num_blocks_per_row[0])
        if nblocks * kbs < bucket:
            return None
        live_bt = blk.block_table.gpu
        if live_bt is None or live_bt.shape[1] > buf.block_table_buf.shape[1]:
            return None

        # ---- Live buffer state (the generic _prepare_inputs contract,
        # specialized for B=1 / computed=0 / scheduled=bucket). ----
        # Block table upload first, overlapping the CPU work below (same
        # ordering as the generic path).
        input_batch.block_table.commit_block_table(1)

        # input_ids: for a fresh single-request prefill the generic gather
        # (token_indices == arange) is the identity, so one row copy into
        # the pinned staging buffer suffices; the unchanged pinned bytes
        # then feed BOTH the live GPU tensor and the graph's static input
        # buffer through two async reads.
        runner.input_ids.cpu[:bucket].copy_(token_ids[0, :bucket])
        runner.input_ids.gpu[:bucket].copy_(runner.input_ids.cpu[:bucket], non_blocking=True)

        qsl = runner.query_start_loc
        qsl.np[0] = 0
        qsl.np[1] = bucket
        # Pad non-decreasing for kernels that require it (generic parity).
        qsl.np[2:].fill(bucket)
        qsl.copy_to_gpu()

        runner.optimistic_seq_lens_cpu[0] = bucket
        runner.optimistic_seq_lens_cpu[1:].fill_(0)

        runner.discard_request_mask.np[0] = bucket < int(req_state.num_tokens)
        runner.discard_request_mask.copy_to_gpu(1)

        # Fresh request: no previous-iteration slot to gather from.
        runner.prev_positions.np[0] = -1

        runner.num_computed_tokens[:1].copy_(
            input_batch.num_computed_tokens_cpu_tensor[:1], non_blocking=True
        )
        # positions == arange(bucket) for computed=0: the capture template
        # IS the value (one D2D copy instead of gather + add).
        runner.positions[:bucket].copy_(buf.positions_template)
        runner.seq_lens[:1].fill_(bucket)
        runner.seq_lens[1:].fill_(0)

        # Slot mapping without the triton kernel:
        # slot[i] = kernel_block_ids[i // kbs] * kbs + (i % kbs)
        offsets = buf.slot_offsets_np
        nblk = (bucket + kbs - 1) // kbs
        if offsets is None or len(offsets) < bucket:
            offsets = np.tile(np.arange(kbs, dtype=np.int64), nblk)[:bucket]
            buf.slot_offsets_np = offsets
        blocks = blk.block_table.np[0, :nblk].astype(np.int64)
        blk.slot_mapping.np[:bucket] = np.repeat(blocks, kbs)[:bucket] * kbs + offsets
        blk.slot_mapping.gpu[:bucket].copy_(blk.slot_mapping.cpu[:bucket], non_blocking=True)

        # ---- Captured graph static input buffers (direct staging). ----
        buf.input_ids_buf[:bucket].copy_(runner.input_ids.cpu[:bucket], non_blocking=True)
        buf.slot_mapping_buf[:bucket].copy_(blk.slot_mapping.cpu[:bucket], non_blocking=True)
        num_cols = live_bt.shape[1]
        buf.block_table_buf[0, :num_cols].copy_(live_bt[0, :num_cols], non_blocking=True)

        if not buf.capture_invariants_fresh:
            # A differently-shaped generic replay overwrote the static
            # positions / query_start_loc / seq_lens buffers; restore the
            # capture values (identical to this batch's live values).
            buf.positions_buf.copy_(buf.positions_template)
            buf.query_start_loc_buf[:2].copy_(qsl.gpu[:2])
            buf.seq_lens_buf[:1].copy_(runner.seq_lens[:1])
        buf.capture_invariants_fresh = True
        # Publish the one-shot token consumed by execute_model_prefill's
        # replay: the static input buffers now hold this dispatch's inputs.
        self._direct_staged_bucket = bucket
        self.direct_stage_count += 1

        # Constant [bucket-1] for this bucket's B=1 exact-fill shape.
        if buf.prefill_logits_indices is None:
            buf.prefill_logits_indices = qsl.gpu[1:2] - 1
        return buf.prefill_logits_indices, None

    def execute_model_prefill(
        self,
        bucket: int,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        slot_mapping: torch.Tensor | None,
        num_scheduled_tokens: list[int] | dict[Any, int],
    ) -> torch.Tensor | None:
        """Populates static device buffers, replays captured CUDA Graph for num_reqs, and returns output."""
        token_counts = (
            list(num_scheduled_tokens.values())
            if isinstance(num_scheduled_tokens, dict)
            else list(num_scheduled_tokens)
        )
        num_reqs = len(token_counts)
        total_actual = sum(token_counts)
        self.last_replay_reason = "graph_not_captured"
        buf = self.buffers.get(bucket)
        if buf is None or not buf.has_captured_graph(num_reqs):
            return None

        # One-shot direct-staging handshake: try_direct_stage_prefill(),
        # run from the patched _prepare_inputs for an exact-bucket B=1
        # prefill, already staged this dispatch's inputs into the static
        # buffers (and restored the capture-invariant positions /
        # query_start_loc / seq_lens values, which equal this batch's live
        # values for that shape). Consume the token before any early
        # return so no later dispatch can observe this staging.
        direct_staged = num_reqs == 1 and self._direct_staged_bucket == bucket
        self._direct_staged_bucket = None

        # Never replay with a missing slot mapping: the captured
        # reshape-and-cache would skip or misdirect the current KV writes.
        if slot_mapping is None:
            self.last_replay_reason = "missing_slot_mapping"
            logger.warning(
                "GPU prefill graph replay: no current slot mapping; falling back to eager prefill."
            )
            return None

        # The captured flash-attention bound (max_seqlen_k) is baked from
        # 2x the bucket; reject batches whose real per-request KV lengths
        # exceed it (the scalar cannot change at replay).
        bound = buf.max_seq_len_bounds.get(num_reqs)
        max_kvlen = self._max_scheduled_kvlen(num_scheduled_tokens)
        if bound is not None and max_kvlen is not None and max_kvlen > bound:
            self.last_replay_reason = "kv_length_exceeds_bound"
            logger.debug(
                "GPU prefill graph replay: batch max KV length %d exceeds the "
                "captured bound %d; falling back to eager prefill.",
                max_kvlen,
                bound,
            )
            return None

        if input_ids is None or positions is None:
            self.last_replay_reason = "missing_inputs"
            logger.warning(
                "GPU prefill graph replay: missing live inputs; falling back to eager prefill."
            )
            return None

        if not direct_staged:
            # Refresh the static buffers with plain on-stream D2D copies from the
            # live per-batch tensors. _prepare_inputs has already computed all of
            # them (positions, slot mapping, seq_lens, query start loc) for THIS
            # batch on the current stream, and the next batch's writes are
            # enqueued strictly after these copies on the same stream (a CUDA
            # stream is FIFO), so reading the live tensors here can never observe
            # the next batch's data. No host-side recomputation or allocations.
            if total_actual < bucket:
                buf.input_ids_buf[total_actual:].fill_(buf.pad_token_id)
                buf.positions_buf[..., total_actual:].fill_(0)
                buf.slot_mapping_buf[total_actual:].fill_(-1)
            buf.input_ids_buf[:total_actual].copy_(input_ids[:total_actual], non_blocking=True)
            if positions.ndim == buf.positions_buf.ndim == 1:
                buf.positions_buf[:total_actual].copy_(positions[:total_actual], non_blocking=True)
            else:
                buf.positions_buf[..., :total_actual].copy_(
                    positions[..., :total_actual], non_blocking=True
                )
            buf.slot_mapping_buf[:total_actual].copy_(
                slot_mapping[:total_actual], non_blocking=True
            )

            # Per-request attention metadata: copy the live GPU tensors
            # _prepare_inputs just computed, in the InputBatch row order.
            qsl_gpu = getattr(self.model_runner, "query_start_loc", None)
            if qsl_gpu is None or not hasattr(qsl_gpu, "gpu"):
                self.last_replay_reason = "missing_query_start_loc"
                logger.warning(
                    "GPU prefill graph replay: live query_start_loc unavailable; "
                    "falling back to eager prefill."
                )
                return None
            buf.query_start_loc_buf[: num_reqs + 1].copy_(
                qsl_gpu.gpu[: num_reqs + 1], non_blocking=True
            )
            seq_gpu = getattr(self.model_runner, "seq_lens", None)
            if seq_gpu is None:
                self.last_replay_reason = "missing_seq_lens"
                logger.warning(
                    "GPU prefill graph replay: live seq_lens unavailable; "
                    "falling back to eager prefill."
                )
                return None
            buf.seq_lens_buf[:num_reqs].copy_(seq_gpu[:num_reqs], non_blocking=True)

            # Refresh the static block table (baked pointer) from the live GPU
            # block table, which the runner keeps current for the batch. vLLM
            # pre-allocates the block table at a fixed width (max_model_len /
            # block_size), so the live and capture widths are equal in practice;
            # the explicit column slice keeps the copy correct if that ever
            # changes upstream (copy_ requires matching shapes).
            if buf.block_table_buf is not None:
                try:
                    live_bt = self.model_runner.input_batch.block_table[0].block_table.gpu
                    if live_bt is None:
                        raise AttributeError("live block table is None")
                except (AttributeError, IndexError, TypeError):
                    self.last_replay_reason = "missing_block_table"
                    logger.warning(
                        "GPU prefill graph replay: live block table unavailable; "
                        "falling back to eager prefill."
                    )
                    return None
                if live_bt.shape[0] < num_reqs or live_bt.shape[1] > buf.block_table_buf.shape[1]:
                    self.last_replay_reason = "block_table_shape"
                    logger.warning(
                        "GPU prefill graph replay: block table shape %s exceeds the "
                        "staged buffer; falling back to eager prefill.",
                        tuple(live_bt.shape),
                    )
                    return None
                num_cols = min(live_bt.shape[1], buf.block_table_buf.shape[1])
                buf.block_table_buf[:num_reqs, :num_cols].copy_(
                    live_bt[:num_reqs, :num_cols], non_blocking=True
                )

            # The generic staging overwrote the capture-invariant buffers
            # with this batch's values; a later direct-stage must restore
            # the capture values before it may skip its own copies.
            buf.capture_invariants_fresh = False
        # else: try_direct_stage_prefill() staged input_ids_buf /
        # slot_mapping_buf / block_table_buf directly from the pinned input
        # tensors and left positions / query_start_loc / seq_lens at their
        # capture-invariant values (identical to this batch's live values
        # for the B=1 exact-bucket shape), so every staging copy above is
        # redundant for this dispatch.

        # Replay the captured graph for num_reqs.
        graph = buf.captured_graphs[num_reqs]
        graph.replay()
        self.last_replay_reason = "replay"

        return buf.output_bufs[num_reqs]
