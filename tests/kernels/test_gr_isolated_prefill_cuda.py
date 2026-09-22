# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inputs prepared from NewRequestData match native-row GR preparation."""

from types import SimpleNamespace as NS

import pytest
import torch

from tests.kernels.test_gr_prefill_graph_inputs_cuda import FIELDS, make_buffer
from tests.kernels.test_gr_prefill_inputs_cuda import make_runtime
from vllm_gr.v1.worker.gpu_prefill_graph_inputs import try_prepare_gr_prefill_graph_inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.mark.parametrize(
    "computed,scheduled,bucket",
    [
        (0, 1024, 1024),
        (1008, 16, 16),
        (1023, 1, 1),
        (0, 1, 128),
        (1000, 24, 128),
        (7, 249, 256),
        (31, 2017, 2048),
    ],
)
@pytest.mark.parametrize("block_size", [16, 32])
def test_isolated_matches_native_row_inputs(computed, scheduled, bucket, block_size):
    native, output, _ = make_runtime(
        computed, scheduled, computed + scheduled, block_sizes=(block_size,)
    )
    isolated, standalone_output, _ = make_runtime(
        computed, scheduled, computed + scheduled, block_sizes=(block_size,)
    )
    output.num_scheduled_tokens = standalone_output.num_scheduled_tokens = {"r": scheduled}
    isolated.input_batch.num_reqs = 0
    isolated.input_batch.req_ids.clear()
    expected, actual = make_buffer(native, bucket, 7), make_buffer(isolated, bucket, 7)
    for iteration in range(2):
        count = (computed + scheduled + block_size - 1) // block_size
        blocks = list(reversed(range(9 + iteration * 100, 9 + iteration * 100 + count)))
        prompt = [x + iteration * 99 for x in range(computed + scheduled)]
        native.input_batch.block_table[0].add_row(blocks, 0)
        native.input_batch.token_ids_cpu[0, : len(prompt)] = prompt
        request = NS(
            req_id="r", prompt_token_ids=prompt, num_computed_tokens=computed, block_ids=(blocks,)
        )
        native_ticket = try_prepare_gr_prefill_graph_inputs(native, output, expected, bucket)
        ticket = try_prepare_gr_prefill_graph_inputs(
            isolated, standalone_output, actual, bucket, new_request=request
        )
        assert native_ticket is not None and ticket is not None
        assert ticket.kv_len == computed + scheduled
        torch.accelerator.synchronize()
        for name in FIELDS:
            torch.testing.assert_close(
                getattr(actual, name), getattr(expected, name), rtol=0, atol=0, msg=name
            )
        assert isolated.input_batch.num_reqs == 0 and not isolated.input_batch.req_ids


@pytest.mark.parametrize(
    "case",
    [
        "active_batch",
        "missing_blocks",
        "extra_group",
        "short_buffer",
        "unpin",
        "wrong_count",
        "overflow_blocks",
    ],
)
def test_isolated_decline_before_staging(case):
    runner, output, _ = make_runtime()
    output.num_scheduled_tokens = {"r": 1024}
    runner.input_batch.num_reqs = 0
    runner.input_batch.req_ids.clear()
    request = NS(
        req_id="r",
        prompt_token_ids=list(range(1024)),
        num_computed_tokens=0,
        block_ids=(list(range(64)),),
    )
    buf = make_buffer(runner, 1024)
    if case == "active_batch":
        runner.input_batch.num_reqs = 1
    elif case == "missing_blocks":
        request.block_ids = ([1],)
    elif case == "extra_group":
        request.block_ids = (list(range(64)), list(range(64)))
    elif case == "short_buffer":
        buf.positions_buf = buf.positions_buf[:16]
    elif case == "unpin":
        runner.input_batch.block_table[0].block_table.cpu = torch.empty(4, 256, dtype=torch.int32)
    elif case == "wrong_count":
        output.num_scheduled_tokens = {"other": 1024}
    elif case == "overflow_blocks":
        request.block_ids = (list(range(300)),)
    before = {name: getattr(buf, name).clone() for name in FIELDS}
    ids = runner.input_ids.cpu.clone()
    table = runner.input_batch.block_table[0].block_table.cpu.clone()
    assert (
        try_prepare_gr_prefill_graph_inputs(runner, output, buf, 1024, new_request=request) is None
    )
    torch.accelerator.synchronize()
    for name in FIELDS:
        torch.testing.assert_close(getattr(buf, name), before[name])
    torch.testing.assert_close(runner.input_ids.cpu, ids)
    torch.testing.assert_close(
        runner.input_batch.block_table[0].block_table.cpu, table, equal_nan=True
    )
