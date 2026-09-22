# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct graph preparation versus native input preparation and staging."""

import numpy as np
import pytest
import torch

from tests.kernels.test_gr_prefill_inputs_cuda import NATIVE_PREPARE, make_runtime
from vllm_gr.v1.worker.gpu_prefill_graph_inputs import try_prepare_gr_prefill_graph_inputs
from vllm_gr.v1.worker.prefill_graph_common import PrefillBucketBuffers

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
FIELDS = (
    "input_ids_buf",
    "positions_buf",
    "slot_mapping_buf",
    "query_start_loc_buf",
    "seq_lens_buf",
    "block_table_buf",
)


def make_buffer(runner, bucket, extra=0):
    columns = runner.input_batch.block_table[0].block_table.cpu.shape[1]
    buf = PrefillBucketBuffers(bucket, 8, runner.device, 17, columns + extra)
    buf.positions_buf = torch.empty(bucket, dtype=torch.int64, device=runner.device)
    for name in FIELDS:
        getattr(buf, name).fill_(37)
    return buf


def native_copy(runner, buf, scheduled, bucket):
    buf.input_ids_buf[:scheduled].copy_(runner.input_ids.gpu[:scheduled])
    buf.input_ids_buf[scheduled:bucket].fill_(buf.pad_token_id)
    buf.positions_buf[:scheduled].copy_(runner.positions[:scheduled])
    buf.positions_buf[scheduled:bucket].zero_()
    table = runner.input_batch.block_table[0]
    buf.slot_mapping_buf[:scheduled].copy_(table.slot_mapping.gpu[:scheduled])
    buf.slot_mapping_buf[scheduled:bucket].fill_(-1)
    buf.query_start_loc_buf[:2].copy_(runner.query_start_loc.gpu[:2])
    buf.seq_lens_buf[:1].copy_(runner.seq_lens[:1])
    buf.block_table_buf[:1, : table.block_table.gpu.shape[1]].copy_(table.block_table.gpu[:1])


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
def test_direct_matches_native_and_next_ordinary_inputs(computed, scheduled, bucket, block_size):
    native, output, counts = make_runtime(
        computed, scheduled, computed + scheduled, block_sizes=(block_size,)
    )
    direct, direct_output, _ = make_runtime(
        computed, scheduled, computed + scheduled, block_sizes=(block_size,)
    )
    direct_output.num_scheduled_tokens = {"r": scheduled}
    expected, actual = make_buffer(native, bucket, 7), make_buffer(direct, bucket, 7)
    # Verify that every request reads current block IDs and token values.
    for iteration in range(2):
        for runner in (native, direct):
            table = runner.input_batch.block_table[0]
            n = (computed + scheduled + block_size - 1) // block_size
            table.add_row(list(reversed(range(9 + iteration * 100, 9 + iteration * 100 + n))), 0)
            runner.input_batch.token_ids_cpu[0, : computed + scheduled] += iteration * 99
        NATIVE_PREPARE(native, output, counts)
        native_copy(native, expected, scheduled, bucket)
        ticket = try_prepare_gr_prefill_graph_inputs(direct, direct_output, actual, bucket)
        assert ticket is not None and not ticket.consumed
        torch.accelerator.synchronize()
        for name in FIELDS:
            torch.testing.assert_close(
                getattr(actual, name), getattr(expected, name), rtol=0, atol=0, msg=name
            )

    # A subsequent ordinary request must rebuild native inputs correctly,
    # even though direct GR preparation deliberately left them untouched.
    next_count = 513
    for runner in (native, direct):
        runner.input_batch.num_computed_tokens_cpu[0] = 0
        runner.input_batch.num_prompt_tokens[0] = next_count
        runner.requests["r"].num_tokens = next_count
        runner.input_batch.block_table[0].add_row(list(range(50)), 0)
    output.total_num_scheduled_tokens = next_count
    output.gr_stage_metadata = {}
    counts = np.array([next_count], dtype=np.int32)
    a, _ = NATIVE_PREPARE(native, output, counts)
    b, _ = NATIVE_PREPARE(direct, output, counts)
    torch.testing.assert_close(a, b)
    for name in ("input_ids", "query_pos", "req_indices"):
        torch.testing.assert_close(
            getattr(native, name).gpu[:next_count], getattr(direct, name).gpu[:next_count]
        )
    torch.testing.assert_close(native.positions[:next_count], direct.positions[:next_count])
    torch.testing.assert_close(native.seq_lens, direct.seq_lens)
    torch.testing.assert_close(native.query_start_loc.gpu, direct.query_start_loc.gpu)
    torch.testing.assert_close(
        native.input_batch.block_table[0].slot_mapping.gpu,
        direct.input_batch.block_table[0].slot_mapping.gpu,
    )


@pytest.mark.parametrize(
    "case",
    [
        "chunk",
        "mrope",
        "hybrid",
        "cp",
        "missing_blocks",
        "unpin",
        "short_buffer",
        "noncontiguous",
        "wrong_dtype",
        "wrong_request",
        "multi_group",
        "previous_samples",
    ],
)
def test_decline_before_writing(case):
    sizes = (16, 32) if case == "multi_group" else (16,)
    runner, output, _ = make_runtime(block_sizes=sizes)
    output.num_scheduled_tokens = {"r": 1024}
    buf = make_buffer(runner, 1024)
    table = runner.input_batch.block_table[0]
    if case == "chunk":
        runner.input_batch.num_prompt_tokens[0] = 2048
    elif case == "mrope":
        runner.uses_mrope = True
    elif case == "hybrid":
        table.use_hybrid_blocks = True
    elif case == "cp":
        table.dcp_world_size = 2
    elif case == "missing_blocks":
        table.num_blocks_per_row[0] = 2
    elif case == "unpin":
        table.block_table.cpu = torch.empty(table.block_table.cpu.shape, dtype=torch.int32)
    elif case == "short_buffer":
        buf.positions_buf = buf.positions_buf[:16]
    elif case == "noncontiguous":
        buf.positions_buf = torch.full((2048,), 37, device="cuda", dtype=torch.int64)[::2]
    elif case == "wrong_dtype":
        buf.positions_buf = buf.positions_buf.int()
    elif case == "wrong_request":
        output.num_scheduled_tokens = {"other": 1024}
    elif case == "previous_samples":
        runner.input_batch.prev_sampled_token_ids = torch.zeros(1, device="cuda")
    before = {name: getattr(buf, name).clone() for name in FIELDS}
    cpu_before = runner.input_ids.cpu.clone()
    assert try_prepare_gr_prefill_graph_inputs(runner, output, buf, 1024) is None
    torch.accelerator.synchronize()
    for name in FIELDS:
        torch.testing.assert_close(getattr(buf, name), before[name])
    torch.testing.assert_close(runner.input_ids.cpu, cpu_before)
