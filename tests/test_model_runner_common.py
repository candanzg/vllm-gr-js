# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for device-neutral ModelRunner Beam helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.beam.context import BEAM_DATA_VAR
from vllm_gr.v1.worker import model_runner_common as common


def test_position_helpers_support_buffer_forms_and_partial_beam() -> None:
    device = torch.full((8,), -1, dtype=torch.long)
    host = torch.full((8,), -1, dtype=torch.long)

    buffered_runner = SimpleNamespace(positions=SimpleNamespace(gpu=device, cpu=host))
    assert common._get_positions_buffers(buffered_runner) == (device, host)

    plain_runner = SimpleNamespace(positions=device, _positions_cpu_buf=host)
    assert common._get_positions_buffers(plain_runner) == (device, host)

    logits_indices: list[int] = []
    result = common._process_beam_decode_request(
        {
            "prefix_len": 10,
            "beam_width": 2,
            "beam_decode_steps": 4,
            "cache_len": 12,
        },
        seq_len=3,
        token_offset=1,
        gpu_pos=device,
        cpu_pos=host,
        new_logits_indices=logits_indices,
    )

    assert result == (2, 1, 4, [0])
    assert logits_indices == [2]
    assert device.tolist() == [-1, 12, 13, 10, -1, -1, -1, -1]
    assert torch.equal(device, host)

    common._write_position_segments(0, [], [], device, host)
    assert torch.equal(device, host)


def test_incremental_beam_decode_fills_constant_position_ids() -> None:
    device = torch.full((6,), -1, dtype=torch.long)
    host = torch.full((6,), -1, dtype=torch.long)

    logits_indices: list[int] = []
    result = common._process_beam_decode_request(
        {
            "prefix_len": 10,
            "beam_width": 3,
            "beam_decode_steps": 4,
            "cache_len": 12,
            "decode_step": 2,
        },
        seq_len=3,
        token_offset=1,
        gpu_pos=device,
        cpu_pos=host,
        new_logits_indices=logits_indices,
        is_incremental=True,
    )

    assert result == (3, 3, 4, [0, 1, 2])
    assert logits_indices == [1, 2, 3]
    # Every beam consumes one token at position prefix_len + decode_step.
    assert device.tolist() == [-1, 12, 12, 12, -1, -1]
    assert torch.equal(device, host)


def test_mixed_batch_remap_installs_functional_proxy() -> None:
    sampling_metadata = SimpleNamespace(
        temperature=torch.tensor([10, 20, 30, 40]),
        top_p=None,
        top_k=None,
        min_p=None,
        frequency_penalties=None,
        presence_penalties=None,
        repetition_penalties=None,
    )
    input_batch = SimpleNamespace(
        req_ids=["ordinary", "beam", None, "prefill"],
        logprob_token_ids={"stale": [1]},
        num_computed_tokens_cpu=[5, 0, 0, 2],
        num_prompt_tokens=[5, 0, 0, 4],
        num_tokens_no_spec=[10, 20, 30, 40],
        num_tokens=[11, 21, 31, 41],
        token_ids_cpu=torch.arange(16).view(4, 4),
        sampling_metadata=sampling_metadata,
        passthrough="original",
    )
    positions = SimpleNamespace(
        gpu=torch.full((12,), -1, dtype=torch.long),
        cpu=torch.full((12,), -1, dtype=torch.long),
    )
    runner = SimpleNamespace(input_batch=input_batch, positions=positions)
    scheduler_output = SimpleNamespace(
        beam_data={
            "beam": {
                "is_beam_decode": True,
                "prefix_len": 4,
                "beam_width": 2,
                "beam_decode_steps": 2,
                "cache_len": 4,
            }
        }
    )

    logits_indices = common._beam_remap_inputs(runner, scheduler_output, [2, 4, 1, 3])

    assert logits_indices == [3, 5, 9]
    assert positions.gpu[2:6].tolist() == [4, 5, 4, 5]
    assert torch.equal(positions.gpu, positions.cpu)
    assert scheduler_output._gr_request_row_configs == [
        ("ordinary", 0, 0, []),
        ("beam", 1, 2, [0, 1]),
        ("prefill", 4, 1, []),
    ]

    proxy = runner.input_batch
    assert isinstance(proxy, common.InputBatchProxy)
    assert proxy.req_ids.copy() == ["ordinary", "beam", "beam", None, "prefill"]
    assert proxy.sampling_metadata.temperature.tolist() == [20, 20, 40]
    assert proxy.num_tokens_no_spec[2] == 20
    proxy.num_tokens[2] = 99
    assert input_batch.num_tokens[1] == 99
    proxy.is_token_ids[4, 0] = 77
    assert input_batch.token_ids_cpu[3, 0].item() == 77
    proxy.passthrough = "updated"
    assert input_batch.passthrough == "updated"
    assert input_batch.logprob_token_ids == {}


def test_sampling_metadata_remap_is_cached_and_warmed_at_remap() -> None:
    """The remap runs once at proxy install; repeated reads hit the cache."""
    sampling_metadata = SimpleNamespace(
        temperature=torch.tensor([20]),
        top_p=None,
        top_k=None,
        min_p=None,
        frequency_penalties=None,
        presence_penalties=None,
        repetition_penalties=None,
    )
    input_batch = SimpleNamespace(
        req_ids=["beam"],
        logprob_token_ids={},
        sampling_metadata=sampling_metadata,
    )
    runner = SimpleNamespace(
        input_batch=input_batch,
        positions=SimpleNamespace(
            gpu=torch.zeros(8, dtype=torch.long),
            cpu=torch.zeros(8, dtype=torch.long),
        ),
    )
    scheduler_output = SimpleNamespace(
        beam_data={
            "beam": {
                "is_beam_decode": True,
                "prefix_len": 4,
                "beam_width": 2,
                "beam_decode_steps": 2,
                "cache_len": 4,
            }
        }
    )

    common._beam_remap_inputs(runner, scheduler_output, [4])
    proxy = runner.input_batch

    # Eager warm: the metadata was remapped during _beam_remap_inputs
    # (prepare phase), not deferred to the first _sample access.  One beam
    # request with beam_width=2 → the 1-row temperature expands to 2 rows.
    assert sampling_metadata.temperature.tolist() == [20, 20]
    cache = object.__getattribute__(proxy, "_sampling_metadata_cache")
    assert cache is sampling_metadata
    idx_cache = object.__getattribute__(proxy, "_logits_idx_tensor_cache")
    assert idx_cache is not None and idx_cache[1].tolist() == [0, 0]

    # Cache hit: a fresh request-length tensor planted after the first read
    # must NOT be re-expanded (the remap loop is skipped entirely on the
    # second access).
    sampling_metadata.temperature = torch.tensor([30])
    assert proxy.sampling_metadata is sampling_metadata
    assert proxy.sampling_metadata.temperature.tolist() == [30]


def test_sampling_metadata_cache_invalidated_by_rebuild_and_assignment() -> None:
    """A metadata swap (refresh_metadata / setattr) forces a fresh remap."""
    sampling_metadata = SimpleNamespace(
        temperature=torch.tensor([10]),
        top_p=None,
        top_k=None,
        min_p=None,
        frequency_penalties=None,
        presence_penalties=None,
        repetition_penalties=None,
    )
    input_batch = SimpleNamespace(req_ids=["beam"])
    proxy = common.InputBatchProxy(input_batch, ["beam", "beam"], [0, 0], [0, 0])
    input_batch.sampling_metadata = sampling_metadata

    assert proxy.sampling_metadata is sampling_metadata
    assert sampling_metadata.temperature.tolist() == [10, 10]

    # Mid-step rebuild (refresh_metadata): a new metadata object with fresh
    # per-request tensors is picked up and remapped despite the cache.
    rebuilt = SimpleNamespace(
        temperature=torch.tensor([7]),
        top_p=None,
        top_k=None,
        min_p=None,
        frequency_penalties=None,
        presence_penalties=None,
        repetition_penalties=None,
    )
    input_batch.sampling_metadata = rebuilt
    assert proxy.sampling_metadata is rebuilt
    assert rebuilt.temperature.tolist() == [7, 7]

    # Assignment through the proxy forwards to the underlying batch AND
    # invalidates the cache, so the next read remaps the new object.
    replaced = SimpleNamespace(
        temperature=torch.tensor([5]),
        top_p=None,
        top_k=None,
        min_p=None,
        frequency_penalties=None,
        presence_penalties=None,
        repetition_penalties=None,
    )
    proxy.sampling_metadata = replaced
    assert input_batch.sampling_metadata is replaced
    assert proxy.sampling_metadata is replaced
    assert replaced.temperature.tolist() == [5, 5]


def test_logprob_remux_preserves_flattening_and_placeholder_rows() -> None:
    st_ids = torch.tensor([[10], [20], [30]])
    logprobs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    token_ids = torch.tensor([[1, 2], [3, 4], [5, 6]])
    ranks = torch.tensor([7, 8, 9])

    final_st, final_lps, final_ids, final_ranks = common._remux_logprobs_tensors(
        [
            ("ordinary", 0, 1, []),
            ("beam", 1, 2, [0, 1]),
            ("prefill", 3, 0, []),
        ],
        max_chained_dim=4,
        target_K=2,
        orig_lps=logprobs,
        orig_lp_ids=token_ids,
        orig_ranks=ranks,
        st_ids=st_ids,
    )

    assert final_st.tolist() == [[10], [20], [-1]]
    assert final_lps[:2].tolist() == [
        [1.0, 2.0, float("-inf"), float("-inf")],
        [3.0, 4.0, 5.0, 6.0],
    ]
    assert torch.isneginf(final_lps[2]).all()
    assert final_ids.tolist() == [[1, 2, 0, 0], [3, 4, 5, 6], [0, 0, 0, 0]]
    assert final_ranks.tolist() == [[7, 0, 0, 0], [8, 9, 0, 0], [0, 0, 0, 0]]


def test_beam_collapse_accumulates_partial_rows_and_clears_cache() -> None:
    runner = SimpleNamespace()
    beam_data = {"beam": {"is_beam_decode": True, "beam_width": 2}}

    def sampler(values: list[float], token_ids: list[int], rank: int) -> SimpleNamespace:
        return SimpleNamespace(
            sampled_token_ids=torch.tensor([[token_ids[0]]]),
            logprobs_tensors=SimpleNamespace(
                logprobs=torch.tensor([values]),
                logprob_token_ids=torch.tensor([token_ids]),
                selected_token_ranks=torch.tensor([rank]),
            ),
        )

    first = sampler([0.1, 0.2], [1, 2], 3)
    common._beam_collapse_bookkeeping(
        runner,
        object(),
        [("beam", 0, 1, [0])],
        beam_data,
        first,
    )
    assert runner._beam_logprobs_cache["beam"]["completed_beams"] == {0}
    assert first.logprobs_tensors.logprobs.shape == (1, 4)

    second = sampler([0.3, 0.4], [3, 4], 5)
    collapsed = common._beam_collapse_bookkeeping(
        runner,
        object(),
        [("beam", 0, 1, [1])],
        beam_data,
        second,
    )

    assert collapsed == (["beam"], {"beam": 0}, 1)
    assert torch.allclose(
        second.logprobs_tensors.logprobs,
        torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
    )
    assert second.logprobs_tensors.logprob_token_ids.tolist() == [[1, 2, 3, 4]]
    assert "beam" not in runner._beam_logprobs_cache


def test_execute_wrapper_is_idempotent_and_restores_outer_context() -> None:
    class ExpectedError(RuntimeError):
        pass

    scheduler_output = SimpleNamespace(beam_data={"beam": {"is_beam_decode": True}})
    runner = SimpleNamespace()
    should_raise = False

    def execute_model(self: Any, output: Any) -> str:
        assert BEAM_DATA_VAR.get() is self._current_beam_info
        assert self._current_beam_info == {
            "beam_data": output.beam_data,
            "req_ids": [],
        }
        # The Constraint Tables _sample hook reads the scheduler output to find
        # _gr_request_row_configs; it must be published for the whole run.
        assert self._current_scheduler_output is output
        if should_raise:
            raise ExpectedError
        return "result"

    wrapped = common.make_patched_execute_model(execute_model)
    assert common.make_patched_execute_model(wrapped) is wrapped

    outer = {"level": "outer"}
    token = BEAM_DATA_VAR.set(outer)
    try:
        assert wrapped(runner, scheduler_output) == "result"
        assert BEAM_DATA_VAR.get() is outer

        should_raise = True
        with pytest.raises(ExpectedError):
            wrapped(runner, scheduler_output)
        assert BEAM_DATA_VAR.get() is outer
        should_raise = False
    finally:
        BEAM_DATA_VAR.reset(token)

    # The beam context deliberately OUTLIVES execute_model: with async
    # scheduling, sample_tokens (constraint hook + worker beam decision)
    # runs AFTER execute_model returns and still needs the published
    # context — including on the exception path (only the ContextVar is
    # reset in the finally block).
    assert runner._current_beam_data is scheduler_output.beam_data
    assert runner._current_scheduler_output is scheduler_output
    assert runner._current_beam_info == {
        "beam_data": scheduler_output.beam_data,
        "req_ids": [],
    }

    # A subsequent non-beam step overwrites the stale context at entry.
    plain_output = SimpleNamespace(beam_data={})
    assert wrapped(runner, plain_output) == "result"
    assert runner._current_beam_data == {}
    assert runner._current_scheduler_output is plain_output
    assert runner._current_beam_info == {"beam_data": {}, "req_ids": []}


@pytest.mark.parametrize("model_dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
def test_beam_context_pool_follows_model_activation_dtype(
    model_dtype: torch.dtype,
) -> None:
    parallel_config = object()
    model_config = SimpleNamespace(
        dtype=model_dtype,
        get_num_kv_heads=lambda config: 2,
        get_head_size=lambda: 8,
        get_num_layers=lambda config: 2,
    )
    runner = SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        device=torch.device("cpu"),
        # A quantized paged prefix cache must not determine the suffix pool dtype.
        cache_config=SimpleNamespace(cache_dtype="fp8"),
        # Provide a GRConfig with beam_max_width set: beam decode requests
        # without a configured pool raise in _get_or_create_beam_session.
        _vllm_gr_config=SimpleNamespace(beam_max_width=128, beam_max_decode_steps=3),
    )
    beam_data = {
        "beam": {
            "is_beam_decode": True,
            "decode_step": 0,
            "beam_decode_steps": 1,
            "beam_width": 2,
            "prefix_len": 4,
            "max_decode_steps": 3,
        }
    }

    session = common._get_or_create_beam_session(
        runner,
        beam_data,
        SimpleNamespace(beam_finished_session_ids=[]),
    )

    assert session is not None
    assert runner._beam_context.dtype == model_dtype
    assert runner._beam_context.suffix_k_buf.dtype == model_dtype
    assert runner._beam_context.suffix_v_buf.dtype == model_dtype
    assert all(pool.dtype == model_dtype for pool in runner._beam_context.pools)


def test_beam_session_raises_when_beam_max_width_unconfigured() -> None:
    """Beam decode without a configured pool must fail loudly.

    The scheduler freezes the paged cache at the prefix for beam decode
    requests, so running them without the beam KV pool would silently
    compute wrong attention; a RuntimeError pointing at --beam-max-width
    is the only safe outcome.
    """
    runner = SimpleNamespace(
        _vllm_gr_config=SimpleNamespace(beam_max_width=None, beam_max_decode_steps=3),
    )
    beam_data = {
        "beam": {
            "is_beam_decode": True,
            "decode_step": 0,
            "beam_decode_steps": 1,
            "beam_width": 2,
            "prefix_len": 4,
            "max_decode_steps": 3,
        }
    }

    with pytest.raises(RuntimeError, match="beam_max_width"):
        common._get_or_create_beam_session(
            runner,
            beam_data,
            SimpleNamespace(beam_finished_session_ids=[]),
        )


def test_beam_session_rejects_concurrent_incremental_requests() -> None:
    """Two concurrent incremental beam-decode requests must be rejected loudly.

    The shared beam context is created with max_batch=1 (pool slots, suffix
    buffers and captured FULL graphs all bake a single session).  Without
    this guard the second request's registry allocate would fail mid-batch
    with the generic "No free beam session slots" error — and the stale-
    session retry cannot help because BOTH ids are active.  An explicit
    concurrency error is the only safe outcome.
    """
    runner = SimpleNamespace(
        _vllm_gr_config=SimpleNamespace(beam_max_width=128, beam_max_decode_steps=3),
        model_config=SimpleNamespace(
            dtype=torch.float16,
            get_num_kv_heads=lambda config: 2,
            get_head_size=lambda: 8,
            get_num_layers=lambda config: 1,
        ),
        parallel_config=object(),
        device=torch.device("cpu"),
    )
    beam_data = {
        "beam-a": {
            "is_beam_decode": True,
            "decode_step": 0,
            "beam_decode_steps": 1,
            "beam_width": 2,
            "prefix_len": 4,
            "max_decode_steps": 3,
        },
        "beam-b": {
            "is_beam_decode": True,
            "decode_step": 0,
            "beam_decode_steps": 1,
            "beam_width": 2,
            "prefix_len": 4,
            "max_decode_steps": 3,
        },
    }

    with pytest.raises(RuntimeError, match="concurrent incremental beam decode"):
        common._get_or_create_beam_session(
            runner,
            beam_data,
            SimpleNamespace(beam_finished_session_ids=[]),
        )
