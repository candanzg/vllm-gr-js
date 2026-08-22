# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core contracts for NPU ModelRunner wrappers."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.worker import npu_model_runner_patch as npu_patch
from vllm_gr.v1.worker.model_runner_common import InputBatchProxy


def test_npu_prepare_unproxies_and_preserves_four_item_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_batch = SimpleNamespace(req_ids=["request"])
    proxy = InputBatchProxy(input_batch, ["request"], [0], [0])
    spec_metadata = object()
    compressed = object()
    runner = SimpleNamespace(
        input_batch=proxy,
        _current_beam_info={"req_ids": []},
        device=torch.device("cpu"),
    )

    def original(
        self: Any,
        scheduler_output: Any,
        num_scheduled_tokens: Any,
    ) -> tuple[object, object, int, object]:
        assert self.input_batch is input_batch
        assert self._current_beam_info["req_ids"] is input_batch.req_ids
        return object(), spec_metadata, 7, compressed

    monkeypatch.setattr(npu_patch, "_beam_remap_inputs", lambda *args: [3])
    wrapped = npu_patch.make_patched_prepare_inputs(original)
    assert npu_patch.make_patched_prepare_inputs(wrapped) is wrapped

    ordinary = wrapped(runner, SimpleNamespace(beam_data={}), [1])
    assert ordinary[1:] == (spec_metadata, 7, compressed)

    runner.input_batch = proxy
    runner._current_beam_info["req_ids"] = []
    logits, returned_spec, total, returned_compressed = wrapped(
        runner,
        SimpleNamespace(beam_data={"request": {}}),
        [1],
    )
    assert logits.dtype is torch.int32
    assert logits.tolist() == [3]
    assert returned_spec is spec_metadata
    assert total == 7
    assert returned_compressed is compressed


def test_npu_bookkeeping_preserves_six_items_and_restores_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_batch = SimpleNamespace()
    proxy = SimpleNamespace(_obj=original_batch)
    runner = SimpleNamespace(
        input_batch=original_batch,
        _beam_logprobs_cache={"finished": {}, "active": {}},
    )
    ordinary_result = (["lp"], ["valid"], object(), ["request"], {"request": 0}, [3])
    sampler_output = object()

    def original(
        self: Any,
        scheduler_output: Any,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        if getattr(scheduler_output, "fail", False):
            raise RuntimeError("boom")
        return ordinary_result

    def collapse(
        self: Any,
        orig_batch: Any,
        configs: Any,
        beam_data: Any,
        sampler: Any,
    ) -> tuple[list[str], dict[str, int], int]:
        assert orig_batch is original_batch
        assert sampler is sampler_output
        return ["beam"], {"beam": 0}, 1

    monkeypatch.setattr(npu_patch, "_beam_collapse_bookkeeping", collapse)
    wrapped = npu_patch.make_patched_bookkeeping_sync(original)

    assert wrapped(runner, SimpleNamespace(beam_data={}), sampler_output) is ordinary_result

    beam_output = SimpleNamespace(
        beam_data={"beam": {}},
        _gr_request_row_configs=[("beam", 0, 1, [0])],
        finished_req_ids={"finished"},
    )
    runner.input_batch = proxy
    result = wrapped(runner, beam_output, sampler_output)
    assert len(result) == 6
    assert result[:2] == ([[]], [[]])
    assert result[3:5] == (["beam"], {"beam": 0})
    assert runner.input_batch is original_batch
    assert "finished" not in runner._beam_logprobs_cache

    runner.input_batch = proxy
    with pytest.raises(RuntimeError, match="boom"):
        wrapped(
            runner,
            SimpleNamespace(
                beam_data={"beam": {}},
                _gr_request_row_configs=[],
                fail=True,
            ),
            sampler_output,
        )
    assert runner.input_batch is original_batch


def test_npu_sample_patch_is_signature_agnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NPU _sample wrapper tolerates a different metadata argument."""
    from vllm_gr.v1.worker.model_runner_common import _WORKER_SAMPLE_PATCH_MARKER

    events: list[str] = []
    sample_output = object()

    def original(self: Any, logits: Any, extra: Any) -> Any:
        assert extra == "extra"
        events.append("sample")
        return sample_output

    def override(self: Any, logits: Any, sampler_output: Any) -> None:
        assert self is runner
        assert logits == "logits"
        assert sampler_output is sample_output
        events.append("override")

    monkeypatch.setattr(npu_patch, "apply_constrained_sample_override", override)
    wrapped = npu_patch.make_patched_sample_npu(original)
    assert getattr(wrapped, _WORKER_SAMPLE_PATCH_MARKER, False)
    assert npu_patch.make_patched_sample_npu(wrapped) is wrapped

    runner = SimpleNamespace()
    assert wrapped(runner, "logits", "extra") is sample_output
    assert events == ["sample", "override"]


# ---------------------------------------------------------------------------
# patched_determine — FULL grant / guards (mirrors GPU-side contracts)
# ---------------------------------------------------------------------------


def _make_determine_wrapper(native_result: Any) -> Any:
    """Build a patched NPU determine with a stubbed native _orig_determine."""

    def _orig_determine(self: Any, *args: Any, **kwargs: Any) -> Any:
        return native_result

    return npu_patch.make_patched_determine_batch_execution_and_padding_npu(_orig_determine)


def _make_beam_info(session: Any, req_ids: list[str], beam_keys: list[str]) -> dict[str, Any]:
    beam_data = {
        req_id: ({"is_beam_decode": True, "decode_step": 0} if req_id in beam_keys else {})
        for req_id in req_ids
    }
    return {
        "beam_data": beam_data,
        "req_ids": req_ids,
        "beam_session": session,
    }


def test_npu_determine_no_beam_info_passthrough() -> None:
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    native = (CUDAGraphMode.NONE, BatchDescriptor(num_tokens=16))
    runner = SimpleNamespace(_current_beam_info=None)
    wrapped = _make_determine_wrapper(native)
    result = wrapped(runner, num_tokens=16)
    assert result[0] == CUDAGraphMode.NONE
    assert result[1].num_tokens == 16


def test_npu_determine_session_missing_downgrades_full() -> None:
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    beam_info: dict[str, Any] = {
        "beam_data": {"req-1": {"is_beam_decode": True, "decode_step": 0}},
        "req_ids": ["req-1"],
        # No "beam_session" key
    }
    native = (CUDAGraphMode.FULL, BatchDescriptor(num_tokens=4, uniform=True))
    runner = SimpleNamespace(_current_beam_info=beam_info)

    orig_xllm = npu_patch._npu_xllm_decode_mode_active
    npu_patch._npu_xllm_decode_mode_active = lambda: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        npu_patch._npu_xllm_decode_mode_active = orig_xllm


def test_npu_determine_mixed_batch_downgrades_full() -> None:
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(is_active=True)
    beam_info = _make_beam_info(session, ["beam-1", "std-1"], ["beam-1"])
    native = (CUDAGraphMode.FULL, BatchDescriptor(num_tokens=8, uniform=True))
    runner = SimpleNamespace(_current_beam_info=beam_info)

    orig_xllm = npu_patch._npu_xllm_decode_mode_active
    npu_patch._npu_xllm_decode_mode_active = lambda: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=8)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        npu_patch._npu_xllm_decode_mode_active = orig_xllm


def test_npu_determine_nonzero_slot_downgrades_full() -> None:
    """The captured graph bakes pool slot 0; other slots must not replay it."""
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(is_active=True, slot=1)
    beam_info = _make_beam_info(session, ["beam-1"], ["beam-1"])
    native = (CUDAGraphMode.FULL, BatchDescriptor(num_tokens=4, uniform=True))
    runner = SimpleNamespace(_current_beam_info=beam_info)

    orig_xllm = npu_patch._npu_xllm_decode_mode_active
    npu_patch._npu_xllm_decode_mode_active = lambda: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        npu_patch._npu_xllm_decode_mode_active = orig_xllm


def test_npu_determine_cascade_mode_downgrades_full() -> None:
    """Only the xllm execution mode is graph-safe; cascade must go eager."""
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(is_active=True, slot=0)
    beam_info = _make_beam_info(session, ["beam-1"], ["beam-1"])
    native = (CUDAGraphMode.FULL, BatchDescriptor(num_tokens=4, uniform=True))
    runner = SimpleNamespace(
        _current_beam_info=beam_info,
        _beam_graph_keys={4},
    )

    orig_xllm = npu_patch._npu_xllm_decode_mode_active
    orig_enabled = npu_patch._beam_graph_enabled_for_run
    npu_patch._npu_xllm_decode_mode_active = lambda: False
    npu_patch._beam_graph_enabled_for_run = lambda r: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        npu_patch._npu_xllm_decode_mode_active = orig_xllm
        npu_patch._beam_graph_enabled_for_run = orig_enabled


def test_npu_determine_pure_beam_batch_grants_full() -> None:
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(is_active=True, slot=0)
    beam_info = _make_beam_info(session, ["beam-1"], ["beam-1"])
    native = (CUDAGraphMode.NONE, BatchDescriptor(num_tokens=4))
    runner = SimpleNamespace(
        _current_beam_info=beam_info,
        _beam_graph_keys={4},
    )

    orig_xllm = npu_patch._npu_xllm_decode_mode_active
    orig_enabled = npu_patch._beam_graph_enabled_for_run
    npu_patch._npu_xllm_decode_mode_active = lambda: True
    npu_patch._beam_graph_enabled_for_run = lambda r: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.FULL
        assert result[1].num_tokens == 4
        assert result[1].uniform is True
    finally:
        npu_patch._npu_xllm_decode_mode_active = orig_xllm
        npu_patch._beam_graph_enabled_for_run = orig_enabled


def test_npu_determine_graph_disabled_no_full() -> None:
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(is_active=True, slot=0)
    beam_info = _make_beam_info(session, ["beam-1"], ["beam-1"])
    native = (CUDAGraphMode.NONE, BatchDescriptor(num_tokens=4))
    runner = SimpleNamespace(
        _current_beam_info=beam_info,
        _beam_graph_keys={4},
    )

    orig_xllm = npu_patch._npu_xllm_decode_mode_active
    orig_enabled = npu_patch._beam_graph_enabled_for_run
    npu_patch._npu_xllm_decode_mode_active = lambda: True
    npu_patch._beam_graph_enabled_for_run = lambda r: False
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        npu_patch._npu_xllm_decode_mode_active = orig_xllm
        npu_patch._beam_graph_enabled_for_run = orig_enabled


def test_npu_determine_width_mismatch_no_full() -> None:
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(is_active=True, slot=0)
    beam_info = _make_beam_info(session, ["beam-1"], ["beam-1"])
    native = (CUDAGraphMode.NONE, BatchDescriptor(num_tokens=8))
    runner = SimpleNamespace(
        _current_beam_info=beam_info,
        _beam_graph_keys={4},  # captured at 4, batch is 8 tokens
    )

    orig_xllm = npu_patch._npu_xllm_decode_mode_active
    orig_enabled = npu_patch._beam_graph_enabled_for_run
    npu_patch._npu_xllm_decode_mode_active = lambda: True
    npu_patch._beam_graph_enabled_for_run = lambda r: True
    try:
        wrapped = _make_determine_wrapper(native)
        result = wrapped(runner, num_tokens=8)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        npu_patch._npu_xllm_decode_mode_active = orig_xllm
        npu_patch._beam_graph_enabled_for_run = orig_enabled


# ---------------------------------------------------------------------------
# capture_beam_decode_graph — entry guards
# ---------------------------------------------------------------------------


def _stub_gr_config(monkeypatch: pytest.MonkeyPatch, gr_cfg: Any) -> None:
    import vllm_gr.v1.worker.model_runner_common as common

    monkeypatch.setattr(common, "_get_gr_config", lambda r: gr_cfg)


def test_npu_capture_disabled_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_gr_config(monkeypatch, SimpleNamespace(beam_graph_active=False))
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False)
    )
    assert npu_patch.capture_beam_decode_graph(runner) is False


def test_npu_capture_cascade_mode_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the graph-safe xllm execution mode, capture must refuse."""
    _stub_gr_config(
        monkeypatch,
        SimpleNamespace(beam_graph_active=True, beam_max_width=4, beam_max_decode_steps=3),
    )
    monkeypatch.setattr(npu_patch, "_npu_xllm_decode_mode_active", lambda: False)
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False)
    )
    assert npu_patch.capture_beam_decode_graph(runner) is False


def test_npu_capture_tp_gt1_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_gr_config(
        monkeypatch,
        SimpleNamespace(beam_graph_active=True, beam_max_width=4, beam_max_decode_steps=3),
    )
    monkeypatch.setattr(npu_patch, "_npu_xllm_decode_mode_active", lambda: True)
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=2, use_ubatching=False),
    )
    assert npu_patch.capture_beam_decode_graph(runner) is False


def test_npu_capture_ubatching_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_gr_config(
        monkeypatch,
        SimpleNamespace(beam_graph_active=True, beam_max_width=4, beam_max_decode_steps=3),
    )
    monkeypatch.setattr(npu_patch, "_npu_xllm_decode_mode_active", lambda: True)
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=True),
    )
    assert npu_patch.capture_beam_decode_graph(runner) is False


def test_npu_capture_reentry_returns_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a graph for the width is already recorded, capture short-circuits True."""
    _stub_gr_config(
        monkeypatch,
        SimpleNamespace(beam_graph_active=True, beam_max_width=4, beam_max_decode_steps=3),
    )
    monkeypatch.setattr(npu_patch, "_npu_xllm_decode_mode_active", lambda: True)
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False),
        _beam_graph_keys={4},
    )
    assert npu_patch.capture_beam_decode_graph(runner) is True


def test_npu_capture_forwards_run_under_inference_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warmup + capture forwards must run inside torch.inference_mode().

    The ACL graph pool holds inference tensors created by the native
    ``_dummy_run`` (decorated with ``@torch.inference_mode()``).  Running the
    capture forward outside inference mode makes ``capture_begin``'s in-place
    pool updates raise "Inplace update to inference tensor outside
    InferenceMode is not allowed".
    """
    import contextlib

    import vllm.compilation.monitor as monitor

    _stub_gr_config(
        monkeypatch,
        SimpleNamespace(beam_graph_active=True, beam_max_width=4, beam_max_decode_steps=3),
    )
    monkeypatch.setattr(npu_patch, "_npu_xllm_decode_mode_active", lambda: True)
    monkeypatch.setattr(monitor, "set_cudagraph_capturing_enabled", lambda *_: None)

    side_stream = SimpleNamespace(wait_stream=lambda *_: None)
    fake_npu = SimpleNamespace(
        Stream=lambda: side_stream,
        current_stream=lambda: SimpleNamespace(wait_stream=lambda *_: None),
        stream=lambda _side: contextlib.nullcontext(),
        synchronize=lambda: None,
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    session = SimpleNamespace(end_session=lambda: None)
    monkeypatch.setattr(
        npu_patch,
        "_capture_create_dummy_context_npu",
        lambda *args, **kwargs: (object(), session, object(), object()),
    )

    inference_flags: list[bool] = []

    def fake_forward(*args: Any, **kwargs: Any) -> None:
        inference_flags.append(torch.is_inference_mode_enabled())

    monkeypatch.setattr(npu_patch, "_dummy_beam_forward_npu", fake_forward)
    monkeypatch.setattr(npu_patch, "_verify_and_record_captured_graph_npu", lambda r, width: True)

    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False),
        _beam_sessions={},
        _current_beam_info=None,
    )
    assert npu_patch.capture_beam_decode_graph(runner) is True
    assert len(inference_flags) == npu_patch._BEAM_CAPTURE_NUM_WARMUPS + 1
    assert all(inference_flags), f"capture forwards ran outside inference_mode: {inference_flags}"


# ---------------------------------------------------------------------------
# _verify_and_record_captured_graph_npu — ACL entry verification
# ---------------------------------------------------------------------------


def test_npu_verify_and_record_records_on_match(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    mock_entry = MagicMock()
    mock_entry.aclgraph = MagicMock()  # not None → captured

    mock_wrapper = MagicMock()
    mock_wrapper.runtime_mode = CUDAGraphMode.FULL
    mock_wrapper.concrete_aclgraph_entries = {
        BatchDescriptor(num_tokens=4, uniform=True): mock_entry
    }

    runner = SimpleNamespace()
    monkeypatch.setattr(
        "vllm_ascend.compilation.acl_graph.ACLGraphWrapper._all_instances",
        [mock_wrapper],
    )
    assert npu_patch._verify_and_record_captured_graph_npu(runner, width=4) is True
    assert 4 in runner._beam_graph_keys


def test_npu_verify_and_record_no_match(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    mock_wrapper = MagicMock()
    mock_wrapper.runtime_mode = MagicMock()  # not FULL
    mock_wrapper.concrete_aclgraph_entries = {}

    runner = SimpleNamespace()
    monkeypatch.setattr(
        "vllm_ascend.compilation.acl_graph.ACLGraphWrapper._all_instances",
        [mock_wrapper],
    )
    assert npu_patch._verify_and_record_captured_graph_npu(runner, width=4) is False
    assert not hasattr(runner, "_beam_graph_keys")


def test_npu_verify_and_record_appends_to_existing_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    mock_entry = MagicMock()
    mock_entry.aclgraph = MagicMock()

    mock_wrapper = MagicMock()
    mock_wrapper.runtime_mode = CUDAGraphMode.FULL
    mock_wrapper.concrete_aclgraph_entries = {
        BatchDescriptor(num_tokens=8, uniform=True): mock_entry
    }

    runner = SimpleNamespace(_beam_graph_keys={4})
    monkeypatch.setattr(
        "vllm_ascend.compilation.acl_graph.ACLGraphWrapper._all_instances",
        [mock_wrapper],
    )
    assert npu_patch._verify_and_record_captured_graph_npu(runner, width=8) is True
    assert runner._beam_graph_keys == {4, 8}
