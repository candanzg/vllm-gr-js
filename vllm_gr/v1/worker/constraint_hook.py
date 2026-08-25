# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-neutral Constraint Tables raw-logits hook (issue #246).

Constraint Tables generate candidates in the Worker at the raw-logits boundary,
not in the frontend: the sampler's ``_sample(logits, ...)`` is wrapped so that,
after the original sampling runs, every *constrained* row of the sampler output
is overwritten in place with the ``rec_constrained_topk`` operator's fixed-width
``W`` result.  ``_beam_collapse_bookkeeping`` / ``_remux_logprobs_tensors``
(model_runner_common) then need zero changes: they derive ``actual_K`` from the
logprob tensor width, which this hook shrinks to ``W`` (so ``W*W`` per session
row, matching the transmission contract P2).

Wire contract (P2):

- decode: one session flat ``[1, W*W]`` — ``W`` beams × ``W`` candidates,
  stride ``W`` per parent block (each block is exactly one operator row).
- prefill: one wide ``[1, W*W]`` segment — ``W`` valid candidates followed by
  ``-inf`` remux padding.
- padding slots in operator rows carry the finite ``INVALID_LOGIT`` sentinel;
  the frontend filters them by threshold, never by ``np.isfinite``.

Guards:

- ``constraint_step`` must be in ``(0, 1, 2)`` — a Constraint Table request
  never degrades to unconstrained sampling (P6).
- A constrained step must not share the sampler output with unconstrained
  logprobs rows, and all constrained rows must share one ``beam_width``
  (mixed-batch / mixed-width routing is P8 scope).
- The operator is only invoked on the last prefill chunk (``W_logits == 1``);
  intermediate chunks carry no logits rows and keep the normal dummy path.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

# Fixed 3-level FSM: constrained prefill (0) plus two constrained decode steps.
_VALID_CONSTRAINT_STEPS = (0, 1, 2)

_DEFAULT_TEMPERATURE = 1.0


def validate_constraint_step(step: Any) -> int:
    """Return ``step`` normalized to a valid Constraint Table FSM step.

    ``constraint_step`` outside ``(0, 1, 2)`` is a hard error (never a silent
    fall back to unconstrained sampling).  Accepts any int-like value
    (``operator.index``), rejecting bools and floats.
    """
    if isinstance(step, bool):
        raise RuntimeError(
            f"constraint_step must be one of {_VALID_CONSTRAINT_STEPS}, got bool {step!r}"
        )
    try:
        normalized = operator.index(step)
    except TypeError:
        normalized = -1
    if normalized not in _VALID_CONSTRAINT_STEPS:
        raise RuntimeError(
            f"constraint_step must be one of {_VALID_CONSTRAINT_STEPS}, "
            f"got {step!r}; Constraint Tables never degrade to unconstrained "
            "sampling"
        )
    return normalized


def collect_constrained_rows(
    request_row_configs: Sequence[Any],
    beam_data: dict[str, Any],
) -> list[tuple[Any, int, int, dict[str, Any]]]:
    """Locate constrained requests and their sampler/logits row ranges.

    Walks ``request_row_configs`` with the same ``gpu_row_cursor`` traversal as
    ``_remux_logprobs_tensors`` (model_runner_common), so the returned
    ``row_start`` values index directly into the sampler-output tensors and the
    ``logits`` argument of the patched ``_sample``.

    Returns ``(req_id, row_start, W_logits, mdata)`` tuples in sampler-row
    order.  Requests with ``W_logits == 0`` (intermediate chunked-prefill
    chunks) produce no logits rows and never run the operator.  Every
    ``is_constrained`` request's ``constraint_step`` is validated, including
    chunk-only visits, so an out-of-range step fails fast.
    """
    collected: list[tuple[Any, int, int, dict[str, Any]]] = []
    gpu_row_cursor = 0
    for req_id, _, W_logits, *_ in request_row_configs:
        mdata = beam_data.get(req_id) if req_id is not None else None
        if mdata is not None and mdata.get("is_constrained", False):
            # ``constraint_step`` is mandatory for every constrained request
            # (the scheduler_metadata builder guarantees it).  Missing is a
            # protocol error: defaulting to 0 would silently treat a decode
            # request as prefill and build the wrong sequence_group.
            constraint_step = mdata.get("constraint_step")
            if constraint_step is None:
                raise RuntimeError(
                    f"constrained request {req_id!r} is missing constraint_step "
                    "in beam metadata; refusing to guess prefill — the "
                    "scheduler metadata chain must provide it"
                )
            validate_constraint_step(constraint_step)
            if W_logits > 0:
                collected.append((req_id, gpu_row_cursor, W_logits, mdata))
        gpu_row_cursor += W_logits
    return collected


def build_sequence_group_for_step(
    rows: Sequence[tuple[Any, int, int, dict[str, Any]]],
    device: Any,
    step: int,
) -> torch.Tensor:
    """Build the operator ``sequence_group`` tensor for constrained rows.

    - prefill (step 0): ``[B, 1, 1]`` zeros — the operator ignores content.
    - decode (steps 1/2): ``[B, W, max_step]`` (int32), one request per ``B``;
      each beam writes its last ``step`` beam tokens into ``[b, :step]``
      (step 1 reads position 0 = ``s_a``; step 2 reads positions 0..1 =
      ``s_a, s_b``).  ``max_step`` is ``max(step, max_decode_steps) + 1``, so
      the operator's ``max_step > current_step`` check always passes.

    Cross-checks ``len(beam_suffix[b]) >= step``: the per-request
    ``constraint_step`` is authoritative and must not disagree with the suffix
    the request actually carries (a global counter inference would hide this).
    """
    step = validate_constraint_step(step)
    if step == 0:
        return torch.zeros([len(rows), 1, 1], dtype=torch.int32, device=device)

    widths = {mdata["beam_width"] for _, _, _, mdata in rows}
    if len(widths) != 1:
        raise RuntimeError(
            f"constrained step mixes beam_widths {sorted(widths)}; "
            "mixed-width Constraint Table batches are not supported yet (P8)"
        )
    width = next(iter(widths))

    max_step = step + 1
    for _, _, _, mdata in rows:
        request_steps = mdata.get("max_decode_steps") or 0
        if request_steps + 1 > max_step:
            max_step = request_steps + 1

    flat: list[int] = []
    for _, _, _, mdata in rows:
        suffix = mdata.get("beam_suffix") or []
        for b in range(width):
            seq = list(suffix[b][-step:]) if b < len(suffix) else []
            if len(seq) != step:
                raise RuntimeError(
                    f"constrained decode step {step} needs the last {step} "
                    f"token(s) per beam, got {len(seq)} from "
                    f"beam_suffix[{b}]={suffix[b]!r}; beam_suffix length is "
                    "inconsistent with constraint_step"
                )
            flat.extend(seq)
            flat.extend([0] * (max_step - step))
    return torch.tensor(flat, dtype=torch.int32, device=device).reshape(len(rows), width, max_step)


def _temperatures_for_rows(runner: Any, rows: Sequence[int], device: Any) -> torch.Tensor:
    """Return a ``[rows]`` / ``[1]`` temperature tensor for constrained rows.

    Prefers the proxy-mapped per-logits-row temperature on the runner's
    sampling metadata (``InputBatchProxy`` maps it to one entry per logits row);
    falls back to a scalar ``1.0`` default.  Always float32 on ``device``.
    """
    s_meta = getattr(getattr(runner, "input_batch", None), "sampling_metadata", None)
    temps = getattr(s_meta, "temperature", None) if s_meta is not None else None
    if isinstance(temps, torch.Tensor) and temps.numel() > 0:
        temps = temps.detach().float().to(device)
        if temps.numel() == 1:
            return temps
        if temps.numel() >= max(rows) + 1:
            return temps.index_select(0, torch.tensor(rows, dtype=torch.long, device=temps.device))
    return torch.tensor([_DEFAULT_TEMPERATURE], dtype=torch.float32, device=device)


def run_constrained_partition(
    runner: Any,
    logits: torch.Tensor,
    rows: Sequence[tuple[Any, int, int, dict[str, Any]]],
    width: int,
    adapter: Any,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Run the constrained operator on every constrained row group.

    Requests are grouped by their ``constraint_step`` and each step group is
    submitted as a *single* batch ``rec_constrained_topk`` call (the operator
    contract is already ``[B*W, vocab]`` logits + ``[B, W, max_step]``
    sequence_group), replacing the previous one call per request.  Requests at
    different FSM steps still run in separate batches because the operator's
    candidate table selection depends on ``current_step``.  The operator
    output rows are returned in the order of the global sampler/logits row
    indices collected in ``global_rows``.
    """
    device = getattr(runner, "device", logits.device)
    top_tokens_parts: list[torch.Tensor] = []
    top_logprobs_parts: list[torch.Tensor] = []
    global_rows: list[int] = []

    by_step: dict[int, list[tuple[Any, int, int, dict[str, Any]]]] = {}
    for item in rows:
        step = validate_constraint_step(item[3].get("constraint_step", 0))
        by_step.setdefault(step, []).append(item)

    for step in sorted(by_step):
        step_rows = by_step[step]
        sequence_group = build_sequence_group_for_step(step_rows, device, step)
        group_rows: list[int] = []
        for _, row_start, W_logits, _ in step_rows:
            group_rows.extend(range(row_start, row_start + W_logits))
        result = adapter.run(
            logits[group_rows],
            sequence_group,
            _temperatures_for_rows(runner, group_rows, device),
            step,
            width,
        )
        top_tokens_parts.append(result.top_tokens)
        top_logprobs_parts.append(result.top_logprobs)
        global_rows.extend(group_rows)

    return (
        torch.cat(top_tokens_parts, dim=0),
        torch.cat(top_logprobs_parts, dim=0),
        global_rows,
    )


def patch_sampler_output_for_constrained(
    sampler_output: Any,
    global_rows: Sequence[int],
    top_tokens: torch.Tensor,
    top_logprobs: torch.Tensor,
    width: int,
) -> None:
    """Overwrite constrained sampler-output rows with operator results.

    Writes the operator ``[rows, width]`` result in place over the sampled
    token / logprob tensors for ``global_rows``, then shrinks the logprob
    tensors to width ``width`` so ``_beam_collapse_bookkeeping`` derives
    ``actual_K == width`` (it reads the tensor width, not a request field).
    The explicit ``.to(...)`` calls protect against dtype/device drift between
    the operator (int32/float32 on the runner device) and the sampler tensors.
    """
    lp_tensors = sampler_output.logprobs_tensors
    if lp_tensors is None or lp_tensors.logprobs is None:
        raise RuntimeError(
            "constrained rows require logprobs in the sampler output, but "
            "logprobs_tensors.logprobs is None; check sampling_params.logprobs"
        )
    if lp_tensors.logprobs.shape[1] < width:
        raise RuntimeError(
            f"sampler logprob width {lp_tensors.logprobs.shape[1]} is smaller "
            f"than constrained beam_width {width}; sampling_params.logprobs "
            "must be at least beam_width"
        )

    rows_idx = torch.tensor(global_rows, dtype=torch.long, device=top_tokens.device)

    # Shrink the logprob tensors to width W BEFORE the in-place scatter.  A
    # ``resize_`` reinterprets the flat storage, so doing it after a scatter
    # would misalign the rows; done first, the first ``num_rows * W`` storage
    # elements are exactly the row-aligned ``[:, :W]`` slice, which is what
    # ``_beam_collapse_bookkeeping`` needs (it derives ``actual_K`` from the
    # tensor width).
    num_rows = lp_tensors.logprobs.shape[0]
    lp_tensors.logprobs.resize_(num_rows, width)
    if lp_tensors.logprob_token_ids is not None:
        lp_tensors.logprob_token_ids.resize_(num_rows, width)

    st_ids = sampler_output.sampled_token_ids

    # Contiguous row ranges (the common pure-beam-decode case: global_rows
    # covers [0, W)) use slice assignment.  Advanced indexing (``[rows_idx]``)
    # dispatches the IndexPut AI_CPU kernel, which profiling (bw128_1k_t2)
    # showed as the single largest device-side cost — 10.3ms per step, 15.7%
    # of total kernel time, with the matching IndexPutImplGetWorkspaceSize
    # host call reaching 47ms — for what is a whole-tensor overwrite when the
    # rows are contiguous.  Slices stay on the vector-core fast path.
    contiguous = global_rows == list(range(global_rows[0], global_rows[-1] + 1))
    n_rows = len(global_rows)
    if contiguous:
        r0 = global_rows[0]

        if st_ids is not None:
            best_tokens = top_tokens[:, 0].to(st_ids.dtype)
            if st_ids.dim() == 2:
                st_ids[r0 : r0 + n_rows, 0] = best_tokens
            else:
                st_ids[r0 : r0 + n_rows] = best_tokens

        # ``logprob_token_ids`` is nullable in the sampler-output contract;
        # see the non-contiguous branch below for the full note.
        if lp_tensors.logprob_token_ids is not None:
            lp_tensors.logprob_token_ids[r0 : r0 + n_rows, :width] = top_tokens.to(
                lp_tensors.logprob_token_ids.dtype
            )
        lp_tensors.logprobs[r0 : r0 + n_rows, :width] = top_logprobs.to(lp_tensors.logprobs.dtype)

        ranks = getattr(lp_tensors, "selected_token_ranks", None)
        if ranks is not None:
            ranks[r0 : r0 + n_rows] = torch.zeros(n_rows, dtype=ranks.dtype, device=ranks.device)
        return

    if st_ids is not None:
        best_tokens = top_tokens[:, 0].to(st_ids.dtype)
        if st_ids.dim() == 2:
            st_ids[rows_idx, 0] = best_tokens
        else:
            st_ids[rows_idx] = best_tokens

    # ``logprob_token_ids`` is nullable in the sampler-output contract (the
    # beam-collapse remux in model_runner_common guards it the same way); only
    # the logprobs tensor is mandatory.  When the sampler omits it, the
    # constrained candidates simply have no token-id side-channel.
    if lp_tensors.logprob_token_ids is not None:
        lp_tensors.logprob_token_ids[rows_idx, :width] = top_tokens.to(
            lp_tensors.logprob_token_ids.dtype
        )
    lp_tensors.logprobs[rows_idx, :width] = top_logprobs.to(lp_tensors.logprobs.dtype)

    ranks = getattr(lp_tensors, "selected_token_ranks", None)
    if ranks is not None:
        ranks[rows_idx] = torch.zeros(len(global_rows), dtype=ranks.dtype, device=ranks.device)


def get_or_create_execution_adapter(runner: Any) -> Any:
    """Lazily create the constrained execution adapter and load tables.

    Fails explicitly when the platform adapter is missing/unavailable (the
    factory already raises), when constraint tables were not injected into this
    worker, or when the loaded digest differs from the injected one
    (``load_tables`` rejects digest swaps).  Never falls back to Trie.
    """
    adapter = getattr(runner, "_gr_constraint_execution", None)
    if adapter is not None:
        return adapter

    from vllm_gr.constraints.worker_inject import get_worker_manager

    manager = get_worker_manager()
    digest = manager.current_digest
    if digest is None:
        raise RuntimeError(
            "constraint tables not injected into this worker; the driver must "
            "inject them via EngineCore.collective_rpc before a constrained "
            "request runs"
        )

    from vllm_gr.constraints.startup import create_constraint_execution

    adapter = create_constraint_execution(device=getattr(runner, "device", None))
    adapter.load_tables(manager.require(digest))
    runner._gr_constraint_execution = adapter
    return adapter


def _validate_homogeneous_step(
    request_row_configs: Sequence[Any],
    beam_data: dict[str, Any],
) -> None:
    """Reject a constrained step that shares rows with unconstrained logprobs.

    The whole logprob tensor is resized to the constrained ``W``, which would
    corrupt unconstrained rows; a constrained request next to any
    non-constrained logprobs row is a hard error (mixed-batch routing is P8).
    Rows without logits (``W_logits == 0``, e.g. regular completion decodes)
    keep the existing dummy path and are not part of this guard.
    """
    for req_id, _, W_logits, *_ in request_row_configs:
        if W_logits == 0:
            continue
        mdata = beam_data.get(req_id) if req_id is not None else None
        if mdata is None or not mdata.get("is_constrained", False):
            raise RuntimeError(
                "a constrained step cannot share the sampler output with "
                f"unconstrained logprobs rows (req_id={req_id!r}); mixed-batch "
                "Constraint Table routing is not supported yet (P8)"
            )


def apply_constrained_sample_override(
    runner: Any,
    logits: torch.Tensor,
    sampler_output: Any,
) -> None:
    """Override constrained sampler-output rows with operator results.

    Short-circuits for pure-completion and Trie-beam steps (no constrained
    rows).  Requires the runner to carry ``_current_beam_data`` (from
    ``make_patched_execute_model``) and ``_current_scheduler_output`` (its
    ``_gr_request_row_configs``), both published before sampling runs.
    """
    beam_data = getattr(runner, "_current_beam_data", None)
    if not beam_data:
        return
    scheduler_output = getattr(runner, "_current_scheduler_output", None)
    request_row_configs = (
        getattr(scheduler_output, "_gr_request_row_configs", None)
        if scheduler_output is not None
        else None
    )
    if not request_row_configs:
        return

    rows = collect_constrained_rows(request_row_configs, beam_data)
    if not rows:
        return

    _validate_homogeneous_step(request_row_configs, beam_data)

    widths = {mdata["beam_width"] for _, _, _, mdata in rows}
    if len(widths) != 1:
        raise RuntimeError(
            f"constrained step mixes beam_widths {sorted(widths)}; "
            "mixed-width Constraint Table batches are not supported yet (P8)"
        )
    width = next(iter(widths))

    adapter = get_or_create_execution_adapter(runner)
    top_tokens, top_logprobs, global_rows = run_constrained_partition(
        runner, logits, rows, width, adapter
    )
    patch_sampler_output_for_constrained(
        sampler_output, global_rows, top_tokens, top_logprobs, width
    )


def collect_constrained_override_rows(runner: Any) -> list[tuple[Any, int, int, dict[str, Any]]]:
    """Return the constrained rows for the runner's current step, if any.

    Thin pre-check for the sampler short-circuit path: same collection walk
    as ``apply_constrained_sample_override`` (rows + homogeneous validation)
    without running the operator, so ``patched_sample`` can decide whether
    the original sampler is needed before paying for it.
    """
    beam_data = getattr(runner, "_current_beam_data", None)
    if not beam_data:
        return []
    scheduler_output = getattr(runner, "_current_scheduler_output", None)
    request_row_configs = (
        getattr(scheduler_output, "_gr_request_row_configs", None)
        if scheduler_output is not None
        else None
    )
    if not request_row_configs:
        return []

    rows = collect_constrained_rows(request_row_configs, beam_data)
    if not rows:
        return []

    _validate_homogeneous_step(request_row_configs, beam_data)
    widths = {mdata["beam_width"] for _, _, _, mdata in rows}
    if len(widths) != 1:
        raise RuntimeError(
            f"constrained step mixes beam_widths {sorted(widths)}; "
            "mixed-width Constraint Table batches are not supported yet (P8)"
        )
    return rows


def build_constrained_sampler_output(
    runner: Any,
    logits: torch.Tensor,
    rows: Sequence[tuple[Any, int, int, dict[str, Any]]],
) -> Any:
    """Run the constrained operator and build a SamplerOutput directly.

    ACS-aligned sampler short-circuit (``model_runner_v1`` decode path):
    when every sampling row is constrained, the sampler's log_softmax +
    topk + gather_logprobs is dead work — the operator already produces the
    fixed-width ``W`` candidates the beam decision consumes.  This builds
    the final sampler output from the operator result in one pass.

    Shape contract (matches ``patch_sampler_output_for_constrained``):
    ``logprob_token_ids``/``logprobs`` are ``[rows, W]`` with the top-1
    candidate in column 0 (``_select_candidates`` takes the ``n_cols == W``
    branch); ``sampled_token_ids`` is ``[rows, 1]`` int32; ranks are all 1
    (the top-1 candidate is rank 1 by definition).
    """
    from vllm.v1.outputs import LogprobsTensors, SamplerOutput

    width = next(iter({mdata["beam_width"] for _, _, _, mdata in rows}))
    adapter = get_or_create_execution_adapter(runner)
    top_tokens, top_logprobs, global_rows = run_constrained_partition(
        runner, logits, rows, width, adapter
    )

    num_rows = top_tokens.shape[0]
    token_ranks = torch.ones(num_rows, dtype=torch.int32, device=top_tokens.device)
    return SamplerOutput(
        sampled_token_ids=top_tokens[:, :1].to(torch.int32),
        logprobs_tensors=LogprobsTensors(
            top_tokens.to(torch.int32),
            top_logprobs.to(torch.float32),
            token_ranks,
        ),
    )
