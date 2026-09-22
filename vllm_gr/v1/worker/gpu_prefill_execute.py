# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute an admitted GR Prefill using its already captured forward graph."""

from typing import Any

import numpy as np
import torch

from vllm_gr.v1.engine.beam_search_v1 import GRStageKind
from vllm_gr.v1.worker.prefill_graph_common import is_prefill_graph_enabled


@torch.inference_mode()  # type: ignore[misc]
def try_execute_gr_prefill(
    runner: Any, scheduler_output: Any, intermediate_tensors: Any = None
) -> bool:
    """Decline before mutation, or publish the native execute/sample state.

    Called only after GPUBeamStageRunner admission (single GPU, text decoder,
    Beam attention, one KV group, no speculative decoding or KV transfer).
    The normal native execute_model remains the fallback for other batches.
    """
    stages = getattr(scheduler_output, "gr_stage_metadata", None)
    graph_runner = getattr(runner, "_prefill_graph_runner", None)
    if (
        intermediate_tensors is not None
        or not stages
        or len(stages) != 1
        or graph_runner is None
        or not is_prefill_graph_enabled()
        or runner.model_config.enforce_eager
        or runner.calculate_kv_scales
        or runner.routed_experts_initialized
        or runner.is_pooling_model
        or runner.use_aux_hidden_state_outputs
        or runner.broadcast_pp_output
        or runner.uses_mrope
        or runner.uses_xdrope_dim > 0
        or runner.enable_prompt_embeds
        or getattr(runner.vllm_config, "ec_transfer_config", None) is not None
        or runner.parallel_config.use_ubatching
        or scheduler_output.scheduled_spec_decode_tokens
        or scheduler_output.scheduled_encoder_inputs
    ):
        return False
    if len(scheduler_output.scheduled_new_reqs) != 1:
        return False
    request = scheduler_output.scheduled_new_reqs[0]
    stage = stages.get(request.req_id)
    scheduled = scheduler_output.total_num_scheduled_tokens
    if (
        stage is None
        or stage.kind is not GRStageKind.PREFILL
        or len(scheduler_output.num_scheduled_tokens) != 1
        or scheduler_output.num_scheduled_tokens.get(request.req_id) != scheduled
        or scheduler_output.scheduled_cached_reqs.req_ids
        or request.prompt_token_ids is None
        or request.prompt_embeds is not None
        or request.mm_features
        or request.lora_request is not None
        or request.pooling_params is not None
        or scheduled <= 0
        or scheduled > runner.max_num_tokens
        or request.num_computed_tokens < 0
        or request.num_computed_tokens + scheduled != len(request.prompt_token_ids)
        or stage.num_prompt_tokens != len(request.prompt_token_ids)
    ):
        return False
    bucket = graph_runner.match_bucket(scheduled)
    buf = graph_runner.buffers.get(bucket)
    if buf is None or not buf.has_captured_graph(1):
        return False
    bound = buf.max_seq_len_bounds.get(1)
    if bound is None or len(request.prompt_token_ids) > bound:
        return False

    # All replay rejection conditions are checked before _update_states. In
    # particular, a KV-bound miss must not update the batch twice on fallback.
    table = runner.input_batch.block_table[0]
    slot_mapping = table.slot_mapping.gpu
    live_bt = table.block_table.gpu
    qsl = getattr(runner.query_start_loc, "gpu", None)
    if (
        slot_mapping is None
        or slot_mapping.numel() < scheduled
        or runner.input_ids.gpu is None
        or runner.input_ids.gpu.numel() < scheduled
        or runner.positions.ndim != 1
        or runner.positions.numel() < scheduled
        or buf.positions_buf.ndim != 1
        or qsl is None
        or qsl.numel() < 2
        or runner.seq_lens is None
        or runner.seq_lens.numel() < 1
        or buf.block_table_buf is None
        or live_bt is None
        or live_bt.shape[0] < 1
        or live_bt.shape[1] > buf.block_table_buf.shape[1]
    ):
        return False
    if runner.execute_model_state is not None:
        raise RuntimeError("State error: sample_tokens() must follow execute_model().")

    from vllm.v1.worker.gpu_model_runner import ExecuteModelState

    from vllm_gr.v1.worker.gpu_prefill_graph_inputs import try_prepare_gr_prefill_graph_inputs
    from vllm_gr.v1.worker.gr_worker_request import try_begin_gr_worker_request

    with runner.synchronize_input_prep():
        prepared = try_begin_gr_worker_request(runner, scheduler_output, buf, bucket)
        corrections = None
        if prepared is None:
            corrections = runner._update_states(scheduler_output)
            prepared = try_prepare_gr_prefill_graph_inputs(runner, scheduler_output, buf, bucket)
        if prepared is None:
            counts = np.array([scheduled], dtype=np.int32)
            logits_indices, spec_metadata = runner._prepare_inputs(scheduler_output, counts)
    if prepared is None:
        hidden_states = graph_runner.execute_model_prefill(
            bucket,
            runner.input_ids.gpu,
            runner.positions,
            slot_mapping,
            scheduler_output.num_scheduled_tokens,
        )
    else:
        hidden_states = graph_runner.execute_model_prefill(
            bucket,
            buf.input_ids_buf,
            buf.positions_buf,
            buf.slot_mapping_buf,
            scheduler_output.num_scheduled_tokens,
            prepared_inputs=prepared,
        )
        spec_metadata = None
    if hidden_states is None:
        raise RuntimeError("Admitted GR Prefill graph unexpectedly declined replay")
    sample_hidden_states = (
        hidden_states[logits_indices]
        if prepared is None
        else hidden_states[scheduled - 1 : scheduled]
    )
    logits = runner.model.compute_logits(sample_hidden_states)
    runner.execute_model_state = ExecuteModelState(
        scheduler_output,
        logits,
        spec_metadata,
        None,
        hidden_states,
        sample_hidden_states,
        None,
        None,
        None,
        None,
    )
    runner.kv_connector_output = None
    if corrections:
        corrections()
    return True
