# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare graph inputs directly for a complete, admitted GR Prefill."""

from dataclasses import dataclass
from typing import Any

import torch

from vllm_gr.v1.engine.beam_search_v1 import GRStageKind


@dataclass(slots=True)
class PreparedGRPrefillInputs:
    """One-use proof that this request's static graph buffers were enqueued."""

    buffer: Any
    bucket: int
    req_id: str
    scheduled: int
    consumed: bool = False
    kv_len: int | None = None

    def consume(self, buffer: Any, bucket: int, counts: Any) -> None:
        if (
            self.consumed
            or self.buffer is not buffer
            or self.bucket != bucket
            or counts != {self.req_id: self.scheduled}
        ):
            raise RuntimeError("GR Prefill prepared inputs are stale or belong to another replay")
        self.consumed = True


def try_prepare_gr_prefill_graph_inputs(
    runner: Any, scheduler_output: Any, buf: Any, bucket: int, *, new_request: Any = None
) -> PreparedGRPrefillInputs | None:
    """Prepare inside the native host-input fence; decline before any writes.

    GR sampling and Decode do not consume native Prefill GPU metadata. Read
    either an updated native row or an admitted new request while the native
    batch is empty. A later ordinary execute prepares its own GPU inputs as
    usual. This is not a _prepare_inputs ABI.
    """
    if runner.input_ids.gpu.device.type != "cuda":
        return None
    batch = runner.input_batch
    stages = getattr(scheduler_output, "gr_stage_metadata", None)
    if not stages or len(stages) != 1:
        return None
    if new_request is None:
        if batch.num_reqs != 1:
            return None
        req_id = batch.req_ids[0]
        computed = int(batch.num_computed_tokens_cpu[0])
        prompt_len = batch.num_prompt_tokens[0]
    else:
        if batch.num_reqs or len(new_request.block_ids) != 1:
            return None
        req_id = new_request.req_id
        computed = new_request.num_computed_tokens
        prompt_len = len(new_request.prompt_token_ids)
    stage = stages.get(req_id)
    scheduled = scheduler_output.total_num_scheduled_tokens
    end = computed + scheduled
    if (
        stage is None
        or stage.kind is not GRStageKind.PREFILL
        or scheduler_output.num_scheduled_tokens != {req_id: scheduled}
        or not 0 < scheduled <= bucket
        or computed < 0
        or end != prompt_len
        or end != stage.num_prompt_tokens
        or runner.uses_mrope
        or runner.uses_xdrope_dim > 0
        or runner.enable_prompt_embeds
        or batch.req_prompt_embeds
        or runner.use_async_spec_decode
        or runner.lora_config is not None
        or runner.num_accepted_tokens_event is not None
        or runner.mamba_prev_last_scheduled_idx is not None
        or batch.prev_sampled_token_ids is not None
        or scheduler_output.scheduled_spec_decode_tokens
        or len(batch.block_table.block_tables) != 1
    ):
        return None
    table = batch.block_table[0]
    if table.use_hybrid_blocks or table.pcp_world_size != 1 or table.dcp_world_size != 1:
        return None
    block_size = table.block_size
    cpu_table = table.block_table.cpu
    cpu_ids = runner.input_ids.cpu
    device = runner.input_ids.gpu.device
    block_count = (
        table.num_blocks_per_row[0] if new_request is None else len(new_request.block_ids[0])
    )
    if (
        block_size <= 0
        or block_count < (end + block_size - 1) // block_size
        or cpu_ids.dtype != torch.int32
        or cpu_ids.ndim != 1
        or cpu_ids.numel() < scheduled
        or not cpu_ids.is_pinned()
        or not cpu_ids.is_contiguous()
        or cpu_table.dtype != torch.int32
        or cpu_table.ndim != 2
        or not cpu_table.is_pinned()
        or not cpu_table.is_contiguous()
        or cpu_table.shape[0] < 1
        or buf.block_table_buf is None
    ):
        return None
    columns = cpu_table.shape[1]
    for tensor, dtype, size, ndim in (
        (buf.input_ids_buf, torch.int32, bucket, 1),
        (buf.positions_buf, torch.int64, bucket, 1),
        (buf.slot_mapping_buf, torch.int64, bucket, 1),
        (buf.query_start_loc_buf, torch.int32, 2, 1),
        (buf.seq_lens_buf, torch.int32, 1, 1),
        (buf.block_table_buf, torch.int32, 1, 2),
    ):
        if (
            tensor.device != device
            or tensor.dtype != dtype
            or tensor.ndim != ndim
            or tensor.shape[0] < size
            or not tensor.is_contiguous()
        ):
            return None
    if columns > buf.block_table_buf.shape[1] or block_count > columns:
        return None

    from vllm_gr.ops.cuda.gr_prefill_graph_inputs import prepare_gr_prefill_graph_metadata

    if new_request is None:
        runner.input_ids.np[:scheduled] = batch.token_ids_cpu[0, computed:end]
    else:
        runner.input_ids.np[:scheduled] = new_request.prompt_token_ids[computed:end]
        # No native row is active. add_row overwrites these inactive values
        # on the next ordinary request; all host writes share its input fence.
        table.block_table.np[0, :block_count] = new_request.block_ids[0]
    buf.input_ids_buf[:scheduled].copy_(cpu_ids[:scheduled], non_blocking=True)
    buf.block_table_buf[:1, :columns].copy_(cpu_table[:1], non_blocking=True)
    prepare_gr_prefill_graph_metadata[((max(bucket, 2) + 255) // 256,)](
        buf.input_ids_buf,
        buf.positions_buf,
        buf.slot_mapping_buf,
        buf.query_start_loc_buf,
        buf.seq_lens_buf,
        buf.block_table_buf,
        computed,
        scheduled,
        buf.pad_token_id,
        BUCKET=bucket,
        BLOCK_SIZE=block_size,
        BLOCK=256,
    )
    return PreparedGRPrefillInputs(buf, bucket, req_id, scheduled, kv_len=end)
