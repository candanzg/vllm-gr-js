# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The prewarmed reset graph must match full reset and preserve other rows."""

import pytest
import torch

from tests.kernels.test_gpu_beam_stage_runner import admission_worker, dispatch, metadata, runtime
from vllm_gr.v1.worker.gpu_beam_stage_runner import get_stage_runner

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
# mypy: disable-error-code="no-untyped-def,no-untyped-call"


def release(owner, m):
    owner.previous = m
    done = torch.Event()
    done.record()
    owner.release(owner.binding, consumer_done=done)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_reset_graph_matches_reference_and_preserves_other_rows(dtype):
    owner = runtime(dtype=dtype, n=3, stop_token_ids=(9,))
    beam = owner.beam
    state = beam.state
    tensors = {k: v for k, v in vars(state).items() if isinstance(v, torch.Tensor)}
    config = {"max_new_tokens_tensor", "stop_token_ids", "num_stop_tokens", "ignore_stop_tokens"}
    for name, tensor in tensors.items():
        if name not in config:
            tensor.fill_(7)
    state.decision.group_token_num.fill_(11)
    state.reset(beam.state_index)
    state.decision.reset(beam.state_index)
    expected = {name: tensor.clone() for name, tensor in tensors.items()}
    expected_groups = state.decision.group_token_num.clone()
    with pytest.raises(RuntimeError, match="released"):
        beam.capture_session_reset()
    release(owner, metadata())
    beam.capture_session_reset()
    assert beam._session_reset_graph is not None
    graph = beam._session_reset_graph
    beam.capture_session_reset()
    assert beam._session_reset_graph is graph
    for _ in range(3):
        for name, tensor in tensors.items():
            if name not in config:
                tensor[beam.state_index].fill_(33)
        state.decision.group_token_num[beam.state_index].fill_(44)
        beam.reset_for_session()
        for name, tensor in tensors.items():
            torch.testing.assert_close(tensor, expected[name])
        torch.testing.assert_close(state.decision.group_token_num, expected_groups)


@torch.inference_mode()
def test_get_stage_reuses_reset_graph_but_new_configuration_falls_back():
    worker = admission_worker()
    m = metadata(n=3)
    owner = get_stage_runner(worker, dispatch(m))
    old = owner.beam
    release(owner, m)
    old.capture_session_reset()
    get_stage_runner(worker, dispatch(m))
    assert owner.beam is old
    assert not old.state.num_generated_tokens.any().item()
    with pytest.raises(RuntimeError, match="released"):
        old.reset_for_session()
    release(owner, m)
    m = metadata(n=2)
    get_stage_runner(worker, dispatch(m))
    assert owner.beam is not old
    assert owner.beam._session_reset_graph is None
    release(owner, m)


@pytest.mark.parametrize("mutation", ["replace", "set", "offset"])
@torch.inference_mode()
def test_changed_storage_is_rejected_before_reset_graph_replay(mutation):
    worker = admission_worker()
    m = metadata()
    owner = get_stage_runner(worker, dispatch(m))
    beam = owner.beam
    release(owner, m)
    owner.graph = object()
    beam.capture_session_reset()
    beam.state.sequence.fill_(321)
    tensor = beam.workspace.selected_token_ids
    if mutation == "replace":
        beam.workspace.selected_token_ids = tensor.clone()
    elif mutation == "set":
        tensor.set_(tensor.clone())
    else:
        replacement = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)
        tensor.set_(replacement[1:].view_as(tensor))
        assert tensor.storage_offset() == 1
    with pytest.raises(ValueError, match="resource addresses or layouts changed"):
        get_stage_runner(worker, dispatch(m))
    assert (beam.state.sequence == 321).all().item()
    assert owner.closed and not worker._beam_sessions
