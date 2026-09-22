# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission, native state contract and host-buffer lifetime for GR Prefill."""

from contextlib import contextmanager
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_gr.v1.engine.beam_search_v1 import GRStageKind
from vllm_gr.v1.worker.gpu_prefill_execute import try_execute_gr_prefill


def runtime(computed=0, scheduled=16):
    events = []
    hidden = torch.arange(32, dtype=torch.float32).reshape(16, 2)
    buf = NS(
        has_captured_graph=lambda n: n == 1,
        max_seq_len_bounds={1: 1024},
        positions_buf=torch.empty(16, dtype=torch.int64),
        block_table_buf=torch.empty(1, 64, dtype=torch.int32),
    )
    request = NS(
        req_id="r",
        prompt_token_ids=list(range(computed + scheduled)),
        num_computed_tokens=computed,
        prompt_embeds=None,
        mm_features=[],
        lora_request=None,
        pooling_params=None,
    )
    output = NS(
        gr_stage_metadata={
            "r": NS(kind=GRStageKind.PREFILL, num_prompt_tokens=computed + scheduled)
        },
        scheduled_new_reqs=[request],
        scheduled_cached_reqs=NS(req_ids=[]),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_scheduled_tokens={"r": scheduled},
        total_num_scheduled_tokens=scheduled,
    )

    @contextmanager
    def fence():
        events.append("wait")
        yield
        events.append("record")

    def update(actual):
        assert actual is output
        events.append("update")
        return lambda: events.append("correct")

    def prepare(actual, counts):
        assert actual is output
        np.testing.assert_array_equal(counts, np.array([scheduled], dtype=np.int32))
        events.append("prepare")
        return torch.tensor([scheduled - 1], dtype=torch.int32), None

    def replay(bucket, ids, positions, slots, counts):
        assert bucket == 16 and counts == {"r": scheduled}
        assert events == ["wait", "update", "prepare", "record"]
        events.append("replay")
        return hidden

    def logits(h):
        events.append("logits")
        return h * 2

    runner = NS(
        _prefill_graph_runner=NS(
            match_bucket=lambda n: 16, buffers={16: buf}, execute_model_prefill=replay
        ),
        model_config=NS(enforce_eager=False),
        vllm_config=NS(ec_transfer_config=None),
        parallel_config=NS(use_ubatching=False),
        model=NS(compute_logits=logits),
        calculate_kv_scales=False,
        routed_experts_initialized=False,
        is_pooling_model=False,
        use_aux_hidden_state_outputs=False,
        broadcast_pp_output=False,
        uses_mrope=False,
        uses_xdrope_dim=0,
        enable_prompt_embeds=False,
        max_num_tokens=1024,
        positions=torch.empty(1024, dtype=torch.int64),
        input_ids=NS(gpu=torch.empty(1024, dtype=torch.int32)),
        query_start_loc=NS(gpu=torch.empty(2, dtype=torch.int32)),
        seq_lens=torch.empty(1, dtype=torch.int32),
        input_batch=NS(
            block_table=[
                NS(
                    slot_mapping=NS(gpu=torch.empty(1024, dtype=torch.int64)),
                    block_table=NS(gpu=torch.empty(1, 64, dtype=torch.int32)),
                )
            ]
        ),
        execute_model_state=None,
        kv_connector_output="old",
        synchronize_input_prep=fence,
        _update_states=update,
        _prepare_inputs=prepare,
    )
    return runner, output, events, hidden


@pytest.mark.parametrize("computed,scheduled", [(0, 16), (1008, 16), (1008, 1)])
def test_prefill_state_and_fence(computed, scheduled):
    runner, output, events, hidden = runtime(computed, scheduled)
    assert try_execute_gr_prefill(runner, output)
    state = runner.execute_model_state
    torch.testing.assert_close(state.logits, hidden[scheduled - 1 : scheduled] * 2)
    assert state.hidden_states is hidden
    assert state.scheduler_output is output
    assert state.slot_mappings is None and state.cudagraph_stats is None
    assert runner.kv_connector_output is None
    assert events == ["wait", "update", "prepare", "record", "replay", "logits", "correct"]
    with pytest.raises(RuntimeError, match="State error"):
        try_execute_gr_prefill(runner, output)
    runner.execute_model_state = None
    events.clear()
    assert try_execute_gr_prefill(runner, output)


@pytest.mark.parametrize(
    "path,value",
    [
        ("_prefill_graph_runner", None),
        ("model_config.enforce_eager", True),
        ("calculate_kv_scales", True),
        ("routed_experts_initialized", True),
        ("is_pooling_model", True),
        ("use_aux_hidden_state_outputs", True),
        ("broadcast_pp_output", True),
        ("uses_mrope", True),
        ("uses_xdrope_dim", 2),
        ("enable_prompt_embeds", True),
        ("vllm_config.ec_transfer_config", object()),
        ("parallel_config.use_ubatching", True),
        ("seq_lens", None),
        ("query_start_loc.gpu", None),
        ("positions", torch.empty(3, 1024)),
        ("input_ids.gpu", None),
        ("max_num_tokens", 8),
    ],
)
def test_runner_fallback_does_not_mutate(path, value):
    runner, output, events, _ = runtime()
    target = runner
    *parts, name = path.split(".")
    for part in parts:
        target = getattr(target, part)
    setattr(target, name, value)
    assert not try_execute_gr_prefill(runner, output)
    assert events == [] and runner.execute_model_state is None


@pytest.mark.parametrize(
    "case",
    [
        "no_capture",
        "kv_bound",
        "no_bound",
        "block_width",
        "no_block_table",
        "no_slots",
        "chunk",
        "no_new",
        "cached",
        "multiple",
        "decode",
        "wrong_prompt",
        "encoder",
        "spec",
        "embeds",
        "negative_computed",
        "no_tokens",
        "intermediate",
        "disabled",
    ],
)
def test_schedule_and_capture_fallback(case, monkeypatch):
    runner, output, events, _ = runtime()
    buf = runner._prefill_graph_runner.buffers[16]
    table = runner.input_batch.block_table[0]
    req = output.scheduled_new_reqs[0]
    intermediate = None
    if case == "no_capture":
        buf.has_captured_graph = lambda n: False
    elif case == "kv_bound":
        buf.max_seq_len_bounds[1] = 15
    elif case == "no_bound":
        buf.max_seq_len_bounds.clear()
    elif case == "block_width":
        buf.block_table_buf = torch.empty(1, 32)
    elif case == "no_block_table":
        table.block_table.gpu = None
    elif case == "no_slots":
        table.slot_mapping.gpu = None
    elif case == "chunk":
        req.prompt_token_ids.append(17)
    elif case == "no_new":
        output.scheduled_new_reqs.clear()
    elif case == "cached":
        output.scheduled_cached_reqs.req_ids.append("s")
    elif case == "multiple":
        output.num_scheduled_tokens["s"] = 1
    elif case == "decode":
        output.gr_stage_metadata["r"].kind = GRStageKind.DECODE
    elif case == "wrong_prompt":
        output.gr_stage_metadata["r"].num_prompt_tokens = 10
    elif case == "encoder":
        output.scheduled_encoder_inputs["r"] = [0]
    elif case == "spec":
        output.scheduled_spec_decode_tokens["r"] = [0]
    elif case == "embeds":
        req.prompt_embeds = torch.empty(16, 2)
    elif case == "negative_computed":
        req.num_computed_tokens = -1
    elif case == "no_tokens":
        output.total_num_scheduled_tokens = 0
    elif case == "intermediate":
        intermediate = object()
    elif case == "disabled":
        monkeypatch.setattr(
            "vllm_gr.v1.worker.gpu_prefill_execute.is_prefill_graph_enabled", lambda: False
        )
    assert not try_execute_gr_prefill(runner, output, intermediate)
    assert events == [] and runner.execute_model_state is None


def test_replay_failure_is_not_reexecuted():
    runner, output, events, _ = runtime()
    runner._prefill_graph_runner.execute_model_prefill = lambda *args: None
    with pytest.raises(RuntimeError, match="unexpectedly declined"):
        try_execute_gr_prefill(runner, output)
    assert events == ["wait", "update", "prepare", "record"]
    assert runner.execute_model_state is None


def test_direct_inputs_keep_record_before_replay(monkeypatch):
    from vllm_gr.v1.worker.gpu_prefill_graph_inputs import PreparedGRPrefillInputs

    runner, output, events, hidden = runtime(1008, 16)
    buf = runner._prefill_graph_runner.buffers[16]
    buf.input_ids_buf = torch.empty(16, dtype=torch.int32)
    buf.slot_mapping_buf = torch.empty(16, dtype=torch.int64)

    def prepare_direct(worker, scheduled, buffer, bucket):
        assert events == ["wait", "update"]
        events.append("direct")
        return PreparedGRPrefillInputs(buffer, bucket, "r", 16)

    def replay(bucket, ids, pos, slots, counts, *, prepared_inputs):
        assert events == ["wait", "update", "direct", "record"]
        prepared_inputs.consume(buf, bucket, counts)
        events.append("replay")
        return hidden

    monkeypatch.setattr(
        "vllm_gr.v1.worker.gpu_prefill_graph_inputs.try_prepare_gr_prefill_graph_inputs",
        prepare_direct,
    )
    runner._prefill_graph_runner.execute_model_prefill = replay
    assert try_execute_gr_prefill(runner, output)
    torch.testing.assert_close(runner.execute_model_state.logits, hidden[15:16] * 2)
    assert events == ["wait", "update", "direct", "record", "replay", "logits", "correct"]


@pytest.mark.parametrize("case", ["bucket", "buffer", "request", "count", "reuse"])
def test_prepared_inputs_ticket_rejects_mismatches(case):
    from vllm_gr.v1.worker.gpu_prefill_graph_inputs import PreparedGRPrefillInputs

    buffer = object()
    ticket = PreparedGRPrefillInputs(buffer, 16, "r", 16)
    counts = {"r": 16}
    bucket = 16
    if case == "bucket":
        bucket = 128
    elif case == "buffer":
        buffer = object()
    elif case == "request":
        counts = {"s": 16}
    elif case == "count":
        counts = {"r": 15}
    else:
        ticket.consume(buffer, bucket, counts)
    with pytest.raises(RuntimeError, match="stale"):
        ticket.consume(buffer, bucket, counts)
