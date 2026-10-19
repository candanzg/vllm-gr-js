# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_gr.v1.engine.types import BeamRequestStepUpdate

logger = init_logger(__name__)

if TYPE_CHECKING:
    pass

# Bounded buffering for beam-session outputs that arrive while the session is
# between step collectors (EngineCore engine-driven chaining runs the next
# step ahead of the frontend's register_beam_output).  Upstream drops outputs for
# request ids without state (output_processor "already-aborted" branch),
# which would hang the beam coroutine's un-timeouted q.get() forever.
# Bootstrap per-session cap; register_beam_output_fn raises it to
# beam_max_decode_steps + 2 once the config is reachable (a live session can
# legitimately buffer one output per engine-led decode step, i.e. up to
# beam_max_decode_steps; +2 covers the terminal output and a margin).
# The cap is an admission/backpressure signal, not a drop threshold: a
# tracked live session must never lose a finished output (that hangs q.get
# until the 120s collect timeout). Overflows are counted and logged.
_GR_BEAM_PENDING_PER_ID = 4
_GR_BEAM_PENDING_MAX_SESSIONS = 256
_PROCESS_OUTPUTS_PATCH_MARKER = "_vllm_gr_beam_pending_patch"

# Terminal admission rejections, keyed by the request id the *engine* used,
# parked between the output that carries one and the serving layer that reports
# it. A request the engine refuses ends on a bare error output whose reason
# lives in ``kv_transfer_params``; the serving loop only ever sees the finish
# reason, so the message needs somewhere to wait in between.
#
# The engine's name for a request is not the name its caller is answered under.
# The frontend resolves the caller's own name into the id it reports and then
# extends it before submitting it, and the engine extends it again, so an
# ordinary request's refusal is filed under a name no serving loop ever sees.
# ``_CLIENT_REQUEST_IDS_ATTR`` records which engine request each caller-owned
# name became, and a claim is an exact lookup under one name or the other --
# never a prefix, since the caller supplies its own name and one may extend
# another's. Pooled rather than assumed empty because a refused V1 session is
# reported by the beam client off the output itself and claims nothing.
_ADMISSION_REJECTIONS_ATTR = "_gr_admission_rejections"
_CLIENT_REQUEST_IDS_ATTR = "_gr_client_request_ids"
_ADMISSION_REJECTION_MAX = 256

# The name the caller of the request being handled in this task will be told.
# Both of the OpenAI frontends resolve a caller-supplied id through
# ``OpenAIServing._base_request_id`` and extend it before submitting it to the
# engine, so the two names differ for every ordinary request and only the
# frontend holds both. It publishes the caller's name here as it resolves it
# (see ``patch_client_request_id``), and both the submit path and the report
# path read it back; outside an OpenAI request it is unset, which keeps
# in-process callers and beam sessions on the engine's own names.
_CLIENT_REQUEST_ID: ContextVar[str | None] = ContextVar("vllm_gr_client_request_id", default=None)


def note_admission_rejections(processor, engine_core_outputs) -> None:
    """Park the reason of every refused output in ``engine_core_outputs``."""
    from vllm_gr.v1.beam.batch_contracts import admission_rejection

    for ec_output in engine_core_outputs:
        message = admission_rejection(ec_output)
        if message is None:
            # A finished output that is not itself a refusal says its id now
            # belongs to a request this pool was never told about. Whatever is
            # parked under that id described an earlier one and must not answer
            # for this one: the caller picks its own id and may reuse it.
            if getattr(ec_output, "finish_reason", None) is not None:
                forget_admission_rejection(processor, getattr(ec_output, "request_id", ""))
            continue
        rejections = getattr(processor, _ADMISSION_REJECTIONS_ATTR, None)
        if rejections is None:
            rejections = {}
            setattr(processor, _ADMISSION_REJECTIONS_ATTR, rejections)
        while len(rejections) >= _ADMISSION_REJECTION_MAX:
            rejections.pop(next(iter(rejections)))
        rejections[getattr(ec_output, "request_id", "")] = message


def forget_admission_rejection(processor, request_id: str) -> None:
    """Drop whatever is parked for ``request_id``, if anything is."""
    rejections = getattr(processor, _ADMISSION_REJECTIONS_ATTR, None)
    if rejections:
        rejections.pop(request_id, None)


def remember_client_request_id(processor, client_request_id: str, engine_request_id: str) -> None:
    """Record which engine request a caller-owned name was submitted as.

    The frontend resolves the caller's name into the id it reports and extends
    it once more before the engine is handed the request, so this pairing of
    the two exists nowhere else. Only the latest engine request per name is
    kept: a name the caller reuses has to find the request it was last
    submitted for, and not the one before it.
    """
    ids = getattr(processor, _CLIENT_REQUEST_IDS_ATTR, None)
    if ids is None:
        ids = {}
        setattr(processor, _CLIENT_REQUEST_IDS_ATTR, ids)
    while len(ids) >= _ADMISSION_REJECTION_MAX:
        ids.pop(next(iter(ids)))
    ids[client_request_id] = engine_request_id


def engine_request_id_for(processor, client_request_id: str) -> str | None:
    """The engine request a caller-owned name was last submitted as, if any."""
    ids = getattr(processor, _CLIENT_REQUEST_IDS_ATTR, None)
    return (ids or {}).get(client_request_id)


def take_admission_rejection(
    engine_client, request_id: str, client_request_id: str | None = None
) -> str | None:
    """Consume the parked reason for the request this caller is asking about.

    Two names can stand for the request that was refused: ``request_id``, the
    one the serving layer reports, and ``client_request_id``, the caller's own,
    which the frontend resolved into the engine's when it submitted the request
    (see ``remember_client_request_id``). Which of the two the reason was filed
    under depends on how the refusal reached the serving layer, so both are
    tried -- and both are looked up exactly, and nothing looser. That is what
    keeps the answer honest: the caller supplies its own name -- an
    ``X-Request-Id`` header, or the body's ``request_id`` -- so a match by
    shared prefix would let one caller's name decide another request's refusal
    and report a real engine fault as a client mistake.
    """
    processor = getattr(engine_client, "output_processor", None)
    rejections = getattr(processor, _ADMISSION_REJECTIONS_ATTR, None)
    if not rejections:
        return None
    message = rejections.pop(request_id, None)
    if message is not None:
        return message
    if client_request_id is None:
        return None
    engine_request_id = engine_request_id_for(processor, client_request_id)
    if engine_request_id is None:
        return None
    return rejections.pop(engine_request_id, None)


def adopt_admission_rejection(engine_client, engine_request_id: str, request_id: str) -> bool:
    """Re-file a reason parked under ``engine_request_id`` as ``request_id``.

    The caller that extended the id calls this once the engine reports back on
    it, holding both names at once, so the reason can be claimed under the name
    the serving layer looks up. The engine's own key is dropped as it moves:
    a reason is never left under two names, and so is still claimed once.
    """
    processor = getattr(engine_client, "output_processor", None)
    rejections = getattr(processor, _ADMISSION_REJECTIONS_ATTR, None)
    if not rejections:
        return False
    message = rejections.pop(engine_request_id, None)
    if message is None:
        return False
    rejections[request_id] = message
    return True


def _ensure_beam_session_ids(processor) -> set:
    session_ids = getattr(processor, "_gr_beam_session_ids", None)
    if session_ids is None:
        session_ids = set()
        processor._gr_beam_session_ids = session_ids
    return session_ids


def _stamp_pending_cap(processor, owner=None) -> int:
    """Raise the per-session buffer cap from config when reachable."""
    cap = getattr(processor, "_gr_beam_pending_cap", None)
    if cap is not None:
        return cap
    cap = _GR_BEAM_PENDING_PER_ID
    try:
        from vllm_gr.config import GRConfig

        vllm_config = getattr(processor, "vllm_config", None)
        if vllm_config is None and owner is not None:
            vllm_config = getattr(owner, "vllm_config", None)
        cap = GRConfig.from_vllm_config(vllm_config).beam_max_decode_steps + 2
    except Exception:  # pragma: no cover - config read is best-effort
        pass
    cap = max(cap, _GR_BEAM_PENDING_PER_ID)
    processor._gr_beam_pending_cap = cap
    return cap


def track_beam_session_fn(self, request_id: str) -> None:
    """Mark ``request_id`` as a live beam session before EngineCore dispatch.

    Must run before SUBMIT_SESSION / ADD_BATCH: engine-driven chaining can
    emit the first decode output while the frontend is still awaiting
    prefill, and process_outputs only buffers ids in this set.
    """
    processor = getattr(self, "output_processor", None)
    if processor is None or not request_id:
        return
    _ensure_beam_session_ids(processor).add(request_id)
    _stamp_pending_cap(processor, owner=self)


def make_patched_process_outputs(original_process_outputs):
    """Wrap OutputProcessor.process_outputs to buffer beam-session outputs.

    Outputs for a known beam session id with no request state are held in a
    per-session buffer instead of being dropped; the next
    register_beam_output for that session replays them.  Non-beam outputs
    keep upstream behavior exactly.  Tracked live sessions never drop:
    exceeding the cap logs and counts an overflow, then keeps the output.

    A refusal the engine attached comes off the same batch on the way past, so
    the serving layer can report the caller's mistake as one.
    """
    if getattr(original_process_outputs, _PROCESS_OUTPUTS_PATCH_MARKER, False):
        return original_process_outputs

    @wraps(original_process_outputs)
    def patched_process_outputs(self, engine_core_outputs, *args, **kwargs):
        # Runs before the early return below: a refused ordinary request gets
        # its error output precisely while a beam session is holding the
        # engine, but the two are independent and neither may mask the other.
        note_admission_rejections(self, engine_core_outputs)
        session_ids = getattr(self, "_gr_beam_session_ids", None)
        if not session_ids:
            return original_process_outputs(self, engine_core_outputs, *args, **kwargs)
        pending = getattr(self, "_gr_beam_pending", None)
        if pending is None:
            pending = {}
            self._gr_beam_pending = pending
        remaining = []
        per_id_cap = getattr(self, "_gr_beam_pending_cap", _GR_BEAM_PENDING_PER_ID)
        overflow = getattr(self, "_gr_beam_pending_overflows", 0)
        for ec_output in engine_core_outputs:
            req_id = getattr(ec_output, "request_id", None)
            if req_id in session_ids and req_id not in self.request_states:
                if req_id not in pending and len(pending) >= _GR_BEAM_PENDING_MAX_SESSIONS:
                    overflow += 1
                    logger.error(
                        "vllm-gr: beam pending session admission overflow "
                        "(sessions=%d, cap=%d); keeping output for live session %s",
                        len(pending),
                        _GR_BEAM_PENDING_MAX_SESSIONS,
                        req_id,
                    )
                buffer = pending.setdefault(req_id, [])
                if len(buffer) >= per_id_cap:
                    overflow += 1
                    logger.error(
                        "vllm-gr: beam pending buffer overflow for live session %s "
                        "(buffered=%d, per_id_cap=%d); keeping the output "
                        "(dropping it would hang the collector)",
                        req_id,
                        len(buffer),
                        per_id_cap,
                    )
                buffer.append(ec_output)
                continue
            remaining.append(ec_output)
        self._gr_beam_pending_overflows = overflow
        if remaining:
            return original_process_outputs(self, remaining, *args, **kwargs)
        return original_process_outputs(self, [], *args, **kwargs)

    setattr(patched_process_outputs, _PROCESS_OUTPUTS_PATCH_MARKER, True)
    return patched_process_outputs


# ---------------------------------------------------------------------------
# AsyncLLM methods for ADD_BATCH / BEAM_REQUEST_STEP_UPDATE
# ---------------------------------------------------------------------------


async def prepare_request_fn(
    self,
    request_id,
    prompt,
    params,
    *,
    lora_request=None,
    tokenization_kwargs=None,
    trace_headers=None,
    priority=0,
    data_parallel_rank=None,
    prompt_text=None,
    arrival_time=None,
):
    """Process input and register output queue, but DON'T send to
    EngineCore. Returns (output_queue, engine_core_request) for later
    batch send via _add_requests_batch().

    The 0.22 input processor validates requests against the asynchronously
    discovered supported-task set, so request preparation is asynchronous."""
    from vllm.pooling_params import PoolingParams
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.output_processor import RequestOutputCollector

    if self.errored:
        from vllm.v1.engine.exceptions import EngineDeadError

        raise EngineDeadError()

    is_pooling = isinstance(params, PoolingParams)

    if (
        self.vllm_config.cache_config.kv_sharing_fast_prefill
        and not is_pooling
        and params.prompt_logprobs
    ):
        raise ValueError(
            "--kv-sharing-fast-prefill produces incorrect logprobs for "
            "prompt tokens, please disable it when the requests need "
            "prompt logprobs"
        )

    # Convert Input --> Request.
    if isinstance(prompt, EngineCoreRequest):
        request = prompt
    else:
        if prompt_text is not None:
            raise ValueError("should only provide prompt_text with EngineCoreRequest")
        request = self.input_processor.process_inputs(
            request_id,
            prompt,
            params,
            supported_tasks=await self.get_supported_tasks(),
            arrival_time=arrival_time,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
        )
        from vllm.renderers.inputs.preprocess import extract_prompt_components

        prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)

    self.input_processor.assign_request_id(request)

    self._run_output_handler()

    # Create a new output collector for the request.
    params = request.params
    queue = RequestOutputCollector(params.output_kind, request.request_id)

    # Register the request in OutputProcessor (this process) but
    # do NOT send to EngineCore yet.
    self.output_processor.add_request(request, prompt_text, None, 0, queue)

    if self.log_requests:
        logger.info("Prepared request %s.", request.request_id)

    return queue, request


async def _add_requests_batch_fn(
    self,
    requests,
    use_batch_message: bool = False,
) -> None:
    """Send multiple already-prepared requests to EngineCore.

    Args:
        use_batch_message: If True, always send as ADD_BATCH (needed for
            beam cache). If False, send individual ADD messages.
    """
    await self.engine_core.add_requests_async(requests, force_batch=use_batch_message)


def register_beam_output_fn(
    self,
    request_id: str,
    prompt_token_ids: list[int],
    sampling_params,
    eos_token_id=None,
    lora_request=None,
    trace_headers=None,
    priority=0,
    data_parallel_rank: int | None = None,
):
    """Register output queue for a beam fork child (no ZMQ send).

    The actual request will be created by EngineCore's beam_request_step_update handler.
    We just need a queue in OutputProcessor to receive outputs."""
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.output_processor import RequestOutputCollector

    if eos_token_id is not None:
        sampling_params.update_from_generation_config({}, eos_token_id)

    ec_request = EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=lora_request,
        cache_salt=None,
        data_parallel_rank=data_parallel_rank,
        trace_headers=trace_headers,
        priority=priority,
    )
    # Set external_req_id directly (skip assign_request_id which
    # would mangle the ID that EngineCore already knows about).
    ec_request.external_req_id = request_id

    self._run_output_handler()
    queue = RequestOutputCollector(sampling_params.output_kind, request_id)
    self.output_processor.add_request(ec_request, None, None, 0, queue)

    # Engine-driven hardening: track the session and replay any outputs that
    # arrived while the session had no collector (the chained step can run
    # ahead of this registration; without the replay they would have been
    # dropped upstream and the beam coroutine would hang on q.get()).
    processor = self.output_processor
    _ensure_beam_session_ids(processor).add(request_id)
    _stamp_pending_cap(processor, owner=self)
    pending = getattr(processor, "_gr_beam_pending", None)
    if pending:
        buffered = pending.get(request_id)
        if buffered:
            # R1: release exactly the OLDEST buffered output. Every beam
            # step Request runs with max_tokens=1, so each EngineCoreOutput
            # arrives finished=True and _collect_beam_result consumes exactly
            # one output per beam_search loop iteration: register:consume is
            # 1:1, so a FIFO one-at-a-time release never starves or stalls.
            # Replaying the whole list in ONE process_outputs call would let
            # the first finished output pop the request state and every
            # remaining one would hit upstream's req_state-is-None drop -
            # the next loop iteration would then await a collector that
            # never fires (bounded only by the 120s collect timeout).
            oldest = buffered.pop(0)
            logger.debug(
                "vllm-gr: replaying oldest of %d buffered beam output(s) for session %s",
                len(buffered) + 1,
                request_id,
            )
            processor.process_outputs([oldest])
            if not buffered:
                pending.pop(request_id, None)
    return queue


async def beam_request_step_update_fn(self, update: BeamRequestStepUpdate) -> None:
    """Send BEAM_REQUEST_STEP_UPDATE to EngineCore to grouped processing."""
    if update.beam_width == 0:
        # Session teardown (the beam_search cleanup update): drop any outputs
        # still buffered for this session (early-EOS break / abort leftovers;
        # see the FIFO release in register_beam_output_fn) and stop tracking
        # the id so late outputs revert to upstream's normal dropping.
        processor = getattr(self, "output_processor", None)
        pending = getattr(processor, "_gr_beam_pending", None)
        if pending is not None:
            pending.pop(update.session_id, None)
        session_ids = getattr(processor, "_gr_beam_session_ids", None)
        if session_ids is not None:
            session_ids.discard(update.session_id)
    await self.engine_core.beam_request_step_update_async(update)


async def submit_session_fn(
    self,
    requests,
    template: BeamRequestStepUpdate,
) -> None:
    """Submit a beam session as one SUBMIT_SESSION message.

    Replaces the ADD_BATCH + BEAM_SESSION_TEMPLATE pair: the prefill batch
    and the session template reach the EngineCore atomically in a single
    wire message.  Must be called before awaiting the prefill outputs (the
    EngineCore consumes the template in the same busy-loop iteration that
    processes the prefill output).  InprocClient keeps its direct-call
    path; all supported clients must expose a merged submission method.

    Tracks the session id before dispatch so a chained step-0 output that
    arrives during prefill collection is buffered instead of dropped.
    """
    session_id = getattr(template, "session_id", None) or (
        getattr(requests[0], "request_id", None) if requests else None
    )
    if session_id:
        track_beam_session_fn(self, session_id)
    submit = getattr(self.engine_core, "submit_session_async", None)
    if submit is not None:
        await submit(requests, template)
        return
    # InprocClient (no wire): direct engine-core handler call.
    inproc_submit = getattr(self.engine_core, "submit_session", None)
    if inproc_submit is not None:
        inproc_submit(requests, template)
        return
    raise RuntimeError("EngineCore client is missing the required submit_session interface")
