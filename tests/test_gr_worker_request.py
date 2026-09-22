# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated GR state against native V1 update/retirement and fallback."""

import copy
import inspect
from types import MethodType
from types import SimpleNamespace as NS

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.worker import gpu_model_runner as native
from vllm.v1.worker.gpu.pool.late_interaction_runner import LateInteractionRunner
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_gr.v1.engine.beam_search_v1 import GRStageKind
from vllm_gr.v1.worker import gr_worker_request as gr
from vllm_gr.v1.worker.gpu_prefill_graph_inputs import PreparedGRPrefillInputs

NATIVE_UPDATE = inspect.unwrap(native.GPUModelRunner._update_states)


def runtime():
    batch = InputBatch(4, 128, 128, torch.device("cpu"), False, 128, [16], [16])
    owner = NS(closed=False, binding=NS(native_request_id="r", session_id="r"))
    worker = NS(
        input_batch=batch,
        requests={},
        num_prompt_logprobs={},
        encoder_cache={},
        device=torch.device("cpu"),
        speculative_config=None,
        is_pooling_model=False,
        uses_mrope=False,
        uses_xdrope_dim=0,
        use_async_spec_decode=False,
        use_async_scheduling=True,
        late_interaction_runner=LateInteractionRunner(),
        _may_reorder_batch=lambda _: None,
        _zero_block_ids=lambda _: None,
        _gr_gpu_stage_runner=owner,
    )
    owner.worker = worker
    worker._update_states = MethodType(gr.make_update_states_with_gr(NATIVE_UPDATE), worker)
    return worker


def new_output():
    output = SchedulerOutput.make_empty()
    params = SamplingParams(temperature=0, max_tokens=3, logprobs=8, detokenize=False)
    request = NewRequestData("r", list(range(32)), [], params, None, ([2, 3],), 0, None)
    output.scheduled_new_reqs = [request]
    output.num_scheduled_tokens = {"r": 32}
    output.total_num_scheduled_tokens = 32
    output.gr_stage_metadata = {
        "r": NS(kind=GRStageKind.PREFILL, owner_generation=9, num_prompt_tokens=32)
    }
    return output


def record_for(worker):
    request = new_output().scheduled_new_reqs[0]
    state = CachedRequestState(
        "r", request.prompt_token_ids, [], request.sampling_params, None, ([2, 3],), 0, []
    )
    owner = worker._gr_gpu_stage_runner
    worker.requests["r"] = state
    worker._gr_worker_request = gr.GRWorkerRequest(owner, owner.binding, 9, state)
    return state


def decode_output(computed=32, blocks=([7],), output_tokens=1):
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {"r": 8}
    output.total_num_scheduled_tokens = 8
    output.scheduled_cached_reqs = CachedRequestData(
        ["r"], set(), [], {}, [blocks], [computed], [output_tokens]
    )
    output.gr_stage_metadata = {"r": NS(kind=GRStageKind.DECODE, owner_generation=9)}
    return output


def assert_state(a, b):
    for name in (
        "req_id",
        "prompt_token_ids",
        "block_ids",
        "num_computed_tokens",
        "output_token_ids",
        "prev_num_draft_len",
        "num_prompt_tokens",
    ):
        assert getattr(a, name) == getattr(b, name), name


def assert_batch(a, b):
    assert a.req_ids == b.req_ids and a.req_id_to_index == b.req_id_to_index
    n = a.num_reqs
    for name in (
        "num_prompt_tokens",
        "num_computed_tokens_cpu",
        "num_tokens_no_spec",
        "temperature_cpu",
        "top_p_cpu",
        "top_k_cpu",
        "num_accepted_tokens_cpu",
    ):
        assert (getattr(a, name)[:n] == getattr(b, name)[:n]).all(), name
    for req_id in a.req_ids:
        row = a.req_id_to_index[req_id]
        end = a.num_tokens_no_spec[row]
        assert (a.token_ids_cpu[row, :end] == b.token_ids_cpu[row, :end]).all()
        columns = a.block_table[0].num_blocks_per_row[row]
        torch.testing.assert_close(
            a.block_table[0].block_table.cpu[row, :columns],
            b.block_table[0].block_table.cpu[row, :columns],
        )
    for name in (
        "all_greedy",
        "all_random",
        "no_penalties",
        "max_num_logprobs",
        "output_token_ids",
    ):
        assert getattr(a.sampling_metadata, name) == getattr(b.sampling_metadata, name), name
    assert a.sampling_metadata.spec_token_ids[:n] == b.sampling_metadata.spec_token_ids[:n]
    if a.sampling_metadata.no_penalties:
        return  # Native unused penalty buffers contain uninitialized storage.
    for name in ("frequency_penalties", "presence_penalties", "repetition_penalties"):
        torch.testing.assert_close(
            getattr(a.sampling_metadata, name), getattr(b.sampling_metadata, name)
        )


@pytest.fixture(autouse=True)
def single_rank(monkeypatch):
    monkeypatch.setattr(native, "get_pp_group", lambda: NS(is_last_rank=True))


def test_begin_keeps_native_batch_empty(monkeypatch):
    worker = runtime()
    output = new_output()
    buf = object()
    metadata = worker.input_batch.sampling_metadata
    monkeypatch.setattr(
        gr,
        "try_prepare_gr_prefill_graph_inputs",
        lambda *args, **kwargs: PreparedGRPrefillInputs(buf, 32, "r", 32, kv_len=32),
    )
    ticket = gr.try_begin_gr_worker_request(worker, output, buf, 32)
    assert ticket is not None
    assert worker.input_batch.num_reqs == 0
    assert worker.input_batch.sampling_metadata is metadata
    ref = runtime()
    NATIVE_UPDATE(ref, new_output())
    assert_state(worker.requests["r"], ref.requests["r"])
    assert worker._gr_worker_request.binding is worker._gr_gpu_stage_runner.binding


def test_update_states_preserves_native_keyword_arguments():
    worker = runtime()
    wrapped = gr.make_update_states_with_gr(NATIVE_UPDATE)
    wrapped(self=worker, scheduler_output=new_output())
    assert worker.input_batch.req_ids == ["r"]
    worker._update_states(scheduler_output=SchedulerOutput.make_empty())
    assert worker.input_batch.num_reqs == 0


@pytest.mark.parametrize(
    "case",
    [
        "ordinary",
        "pending",
        "custom_processor",
        "thinking",
        "active_record",
        "owner_closed",
        "owner_mismatch",
        "finished",
        "encoder",
        "zero_blocks",
        "spec",
        "seed",
        "prompt_logprobs",
        "mixed_embeds",
        "prepare_declines",
    ],
)
def test_begin_declines_without_mutation(monkeypatch, case):
    w = runtime()
    output = new_output()

    def prepare(*args, **kwargs):
        assert case == "prepare_declines", "ineligible request reached input writes"

    monkeypatch.setattr(gr, "try_prepare_gr_prefill_graph_inputs", prepare)
    if case == "ordinary":
        w.requests["ordinary"] = object()
    elif case == "pending":
        w.input_batch.batch_update_builder.batch_changed = True
    elif case == "custom_processor":
        w.input_batch.logitsprocs.argmax_invariant.append(object())
    elif case == "thinking":
        w.input_batch.thinking_budget_state_holder = object()
    elif case == "active_record":
        w._gr_worker_request = object()
    elif case == "owner_closed":
        w._gr_gpu_stage_runner.closed = True
    elif case == "owner_mismatch":
        w._gr_gpu_stage_runner.binding.native_request_id = "other"
    elif case == "finished":
        output.finished_req_ids.add("x")
    elif case == "encoder":
        output.free_encoder_mm_hashes.append("x")
    elif case == "zero_blocks":
        output.new_block_ids_to_zero = [1]
    elif case == "spec":
        w.speculative_config = object()
    elif case == "seed":
        output.scheduled_new_reqs[0].sampling_params = SamplingParams(temperature=1, seed=42)
    elif case == "prompt_logprobs":
        output.scheduled_new_reqs[0].sampling_params.prompt_logprobs = 1
    elif case == "mixed_embeds":
        output.scheduled_new_reqs[0].prompt_is_token_ids = [True] * 32
    before = dict(w.requests)
    metadata = w.input_batch.sampling_metadata
    assert gr.try_begin_gr_worker_request(w, output, object(), 32) is None
    assert w.requests == before and w.input_batch.num_reqs == 0
    assert w.input_batch.sampling_metadata is metadata


@pytest.mark.parametrize("blocks", [None, ([],), ([7, 9],)])
@pytest.mark.parametrize("output_tokens", [0, 1, 3])
def test_decode_matches_native(blocks, output_tokens):
    fast, ref = runtime(), runtime()
    state = record_for(fast)
    NATIVE_UPDATE(ref, new_output())
    state.output_token_ids.extend([71, 72, 73])
    ref.requests["r"].output_token_ids.extend([71, 72, 73])
    output = decode_output(33, blocks, output_tokens)
    NATIVE_UPDATE(ref, copy.deepcopy(output))
    metadata = fast.input_batch.sampling_metadata
    assert fast._update_states(copy.deepcopy(output)) is None
    assert_state(state, ref.requests["r"])
    assert fast.input_batch.num_reqs == 0
    assert fast.input_batch.sampling_metadata is metadata


@pytest.mark.parametrize("case", ["no_gr", "resumed", "zero_blocks", "finished_other", "mixed"])
def test_fallback_materializes_before_native_deltas(case):
    fast, ref = runtime(), runtime()
    record_for(fast)
    NATIVE_UPDATE(ref, new_output())
    output = decode_output()
    if case == "no_gr":
        output.gr_stage_metadata = {}
    elif case == "resumed":
        output.scheduled_cached_reqs.resumed_req_ids.add("r")
        output.scheduled_cached_reqs.all_token_ids["r"] = list(range(32)) + [71]
    elif case == "zero_blocks":
        output.new_block_ids_to_zero = [7]
    elif case == "finished_other":
        output.finished_req_ids.add("other")
    elif case == "mixed":
        new = new_output().scheduled_new_reqs[0]
        new.req_id = "ordinary"
        output.scheduled_new_reqs = [new]
        output.num_scheduled_tokens["ordinary"] = 32
        output.total_num_scheduled_tokens += 32
    NATIVE_UPDATE(ref, copy.deepcopy(output))
    fast._update_states(copy.deepcopy(output))
    assert fast._gr_worker_request is None
    assert_state(fast.requests["r"], ref.requests["r"])
    assert_batch(fast.input_batch, ref.input_batch)


def test_empty_tick_preserves_isolated_state_until_retirement(monkeypatch):
    fast, ref = runtime(), runtime()
    state = record_for(fast)
    NATIVE_UPDATE(ref, new_output())
    metadata = fast.input_batch.sampling_metadata

    def unexpected_add(*args, **kwargs):
        pytest.fail("empty tick must not materialize the GR request")

    monkeypatch.setattr(fast.input_batch, "add_request", unexpected_add)
    for _ in range(2):
        NATIVE_UPDATE(ref, SchedulerOutput.make_empty())
        fast._update_states(SchedulerOutput.make_empty())
        assert fast._gr_worker_request.request is state
        assert_state(state, ref.requests["r"])
        assert fast.input_batch.num_reqs == 0
        assert fast.input_batch.sampling_metadata is metadata
    # Native absent-row resume carries output IDs. The isolated request has
    # stayed resident in its GR lane, so no native resume materialization occurs.
    output = decode_output(output_tokens=0)
    NATIVE_UPDATE(ref, copy.deepcopy(output))
    fast._update_states(output)
    assert_state(state, ref.requests["r"])
    finished = SchedulerOutput.make_empty()
    finished.finished_req_ids.add("r")
    fast._gr_gpu_stage_runner.closed = True
    fast._update_states(finished)
    assert fast._gr_worker_request is None and not fast.requests
    assert fast.input_batch.sampling_metadata is metadata


def test_retirement_and_next_ordinary_request():
    fast, ref = runtime(), runtime()
    record_for(fast)
    NATIVE_UPDATE(ref, new_output())
    metadata = fast.input_batch.sampling_metadata
    output = SchedulerOutput.make_empty()
    output.finished_req_ids.add("r")
    fast._gr_gpu_stage_runner.closed = True
    NATIVE_UPDATE(ref, copy.deepcopy(output))
    fast._update_states(output)
    assert fast._gr_worker_request is None and not fast.requests
    assert fast.input_batch.num_reqs == 0
    assert fast.input_batch.sampling_metadata is metadata
    ordinary = new_output()
    ordinary.gr_stage_metadata = {}
    ordinary.scheduled_new_reqs[0].sampling_params = SamplingParams(
        temperature=0.7, top_k=7, presence_penalty=0.3
    )
    NATIVE_UPDATE(ref, copy.deepcopy(ordinary))
    fast._update_states(copy.deepcopy(ordinary))
    assert_batch(fast.input_batch, ref.input_batch)


@pytest.mark.parametrize("case", ["request", "owner", "binding", "generation"])
def test_owner_mismatch_fails_before_mutation(case):
    w = runtime()
    state = record_for(w)
    if case == "request":
        w.requests["r"] = object()
    elif case == "owner":
        w._gr_gpu_stage_runner = object()
    elif case == "binding":
        w._gr_gpu_stage_runner.binding = NS(native_request_id="other")
    output = decode_output()
    if case == "generation":
        output.gr_stage_metadata["r"].owner_generation = 10
    with pytest.raises(RuntimeError, match="execution owner"):
        w._update_states(output)
    assert state.num_computed_tokens == 0 and state.block_ids == ([2, 3],)
    assert w.input_batch.num_reqs == 0


def test_empty_native_batch_does_not_bypass_graph_kv_bound():
    from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner

    graph = object.__new__(GPUPrefillGraphRunner)
    calls = []
    tensor = torch.empty(16)
    buf = NS(
        has_captured_graph=lambda n: n == 1,
        max_seq_len_bounds={1: 16},
        input_ids_buf=tensor,
        positions_buf=tensor,
        slot_mapping_buf=tensor,
        captured_graphs={1: NS(replay=lambda: calls.append("replay"))},
        output_bufs={1: tensor},
    )
    graph.buffers = {16: buf}
    graph.model_runner = NS(input_batch=NS(req_id_to_index={}, num_computed_tokens_cpu=[]))
    ticket = PreparedGRPrefillInputs(buf, 16, "r", 16, kv_len=1024)
    assert (
        graph.execute_model_prefill(16, tensor, tensor, tensor, {"r": 16}, prepared_inputs=ticket)
        is None
    )
    assert not calls and not ticket.consumed
    assert graph.last_replay_reason == "kv_length_exceeds_bound"
    buf.max_seq_len_bounds[1] = 1024
    assert (
        graph.execute_model_prefill(16, tensor, tensor, tensor, {"r": 16}, prepared_inputs=ticket)
        is tensor
    )
    assert ticket.consumed and calls == ["replay"]
