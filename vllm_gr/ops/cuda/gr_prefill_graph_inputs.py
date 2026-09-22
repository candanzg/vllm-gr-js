# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate one-request Prefill metadata at the captured graph addresses."""

import triton
import triton.language as tl

# mypy: disable-error-code="misc,no-untyped-def"


@triton.jit(do_not_specialize=["computed", "scheduled", "pad_token"])
def prepare_gr_prefill_graph_metadata(
    input_ids,
    positions,
    slots,
    query_start,
    seq_lens,
    block_table,
    computed,
    scheduled,
    pad_token,
    BUCKET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    position = computed + i
    physical_block = tl.load(block_table + position // BLOCK_SIZE, i < scheduled, other=0)
    slot = physical_block.to(tl.int64) * BLOCK_SIZE + position % BLOCK_SIZE
    tl.store(input_ids + i, pad_token, (i >= scheduled) & (i < BUCKET))
    tl.store(positions + i, tl.where(i < scheduled, position, 0), i < BUCKET)
    tl.store(slots + i, tl.where(i < scheduled, slot, -1), i < BUCKET)
    tl.store(query_start + i, tl.where(i == 0, 0, scheduled), i < 2)
    tl.store(seq_lens + i, computed + scheduled, i == 0)
