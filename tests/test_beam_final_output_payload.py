# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact output ABI, optional fields and independence from reused storage."""

from itertools import product

import numpy as np
import numpy.typing as npt
import pytest
import torch

from vllm_gr.ops.cuda.beam_state import FINISH_LENGTH, FINISH_STOP
from vllm_gr.v1.beam.batch_contracts import BeamOutputOptions
from vllm_gr.v1.worker.beam_final_output import _build_result, _PayloadLayout


@pytest.mark.parametrize("steps", [1, 2, 3])  # type: ignore[misc]
@pytest.mark.parametrize("as_tensor", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("flags", list(product([False, True], repeat=3)))  # type: ignore[misc]
def test_compact_payload_fields(steps: int, as_tensor: bool, flags: tuple[bool, ...]) -> None:
    include_stop, include_score, include_parents = flags
    layout = _PayloadLayout.create(4, steps)
    values: npt.NDArray[np.int32] = np.zeros(layout.size, dtype=np.int32)
    values[3] = 2  # Ignore the unused two rows.
    rows = values[4:].reshape(4, layout.row_stride)
    rows[0, :steps] = np.arange(10, 10 + steps)
    rows[1, :steps] = np.arange(20, 20 + steps)
    rows[:, steps : 2 * steps] = np.arange(steps)
    rows.view(np.float32)[:2, 2 * steps] = [-1.25, -2.5]
    rows[:2, 2 * steps + 1] = [steps, 1]
    rows[:2, 2 * steps + 2] = [FINISH_LENGTH, FINISH_STOP]
    rows[:2, 2 * steps + 3] = [-1, 20]
    payload = torch.from_numpy(values) if as_tensor else values
    result = _build_result(
        payload,
        layout,
        session_id="session",
        request_id="native",
        item_index=7,
        options=BeamOutputOptions(2, include_stop, include_score, include_parents),
    )
    values.fill(0)  # Reusing D2H storage cannot mutate an already-published result.
    assert result.failure is None
    assert result.batch_id == "session"
    assert result.output is not None
    item = result.output.items[0]
    assert (item.item_index, item.request_id) == (7, "native")
    length, stop = item.sequences
    assert length.token_ids == tuple(range(10, 10 + steps))
    assert stop.token_ids == ((20,) if include_stop else ())
    assert length.cumulative_logprob == (-1.25 if include_score else None)
    assert stop.cumulative_logprob == (-2.5 if include_score else None)
    assert length.parent_beam_indices == (tuple(range(steps)) if include_parents else ())
    assert stop.parent_beam_indices == ((0,) if include_parents else ())
    assert (length.finish_reason, length.stop_token_id) == ("length", None)
    assert (stop.finish_reason, stop.stop_token_id) == ("stop", 20)


def test_empty_compact_payload_failure() -> None:
    layout = _PayloadLayout.create(4, 3)
    result = _build_result(
        np.zeros(layout.size, dtype=np.int32),
        layout,
        session_id="session",
        request_id="native",
        item_index=7,
        options=BeamOutputOptions(4, True, True, True),
    )
    assert result.output is None
    assert result.failure is not None
    assert result.failure.code == "no_valid_beams"
    assert result.failure.item_index == 7
