# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-request Prefill metadata, written to native ModelRunner buffers."""

import triton
import triton.language as tl

# Triton pointer parameters and constexpr annotations are handled by its JIT.
# mypy: disable-error-code="misc,no-untyped-def"


@triton.jit(do_not_specialize=["computed", "scheduled", "num_tokens"])
def prepare_gr_prefill_metadata(
    positions,
    query_start_loc,
    seq_lens,
    num_computed_tokens,
    req_indices,
    query_pos,
    num_scheduled_tokens,
    discard_request_mask,
    num_accepted_tokens,
    logits_indices,
    computed,
    scheduled,
    num_tokens,
    MAX_REQS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(positions + index, computed + index.to(tl.int64), index < scheduled)
    tl.store(req_indices + index, 0, index < scheduled)
    tl.store(query_pos + index, index, index < scheduled)
    # Native attention requires a nondecreasing padded query_start_loc.
    tl.store(query_start_loc + index, tl.where(index == 0, 0, scheduled), index <= MAX_REQS)
    tl.store(seq_lens + index, tl.where(index == 0, computed + scheduled, 0), index < MAX_REQS)
    tl.store(num_accepted_tokens + index, 1, index < MAX_REQS)
    tl.store(num_computed_tokens + index, computed, index == 0)
    tl.store(num_scheduled_tokens + index, scheduled, index == 0)
    tl.store(discard_request_mask + index, computed + scheduled < num_tokens, index == 0)
    tl.store(logits_indices + index, scheduled - 1, index == 0)
