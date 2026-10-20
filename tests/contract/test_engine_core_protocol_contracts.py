# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
"""Production V1 transport inventory; retired message bytes stay reserved."""

from types import SimpleNamespace

import msgspec
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
from vllm.v1.serial_utils import MsgpackDecoder

from vllm_gr.v1.engine.wire import register_engine_core_request_types


UPSTREAM_REQUEST_WIRE_VALUES = {
    "ADD": b"\x00",
    "ABORT": b"\x01",
    "START_DP_WAVE": b"\x02",
    "UTILITY": b"\x03",
    "EXECUTOR_FAILED": b"\x04",
    "WAKEUP": b"\x05",
}


ENGINE_CORE_REQUEST_FIELDS = (
    "request_id",
    "prompt_token_ids",
    "mm_features",
    "sampling_params",
    "pooling_params",
    "arrival_time",
    "lora_request",
    "cache_salt",
    "data_parallel_rank",
    "prompt_embeds",
    "prompt_is_token_ids",
    "client_index",
    "current_wave",
    "priority",
    "trace_headers",
    "resumable",
    "external_req_id",
    "reasoning_ended",
    "reasoning_parser_kwargs",
    "abort_immediately",
)


def _sampling_params() -> SamplingParams:
    return SamplingParams(max_tokens=4, temperature=1.0)


def test_native_request_payload_schema() -> None:
    # Guard against upstream vLLM EngineCoreRequest field schema changes.
    assert EngineCoreRequest.__struct_fields__ == ENGINE_CORE_REQUEST_FIELDS

    request = EngineCoreRequest(
        request_id="request-0",
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=_sampling_params(),
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    encoded = msgspec.msgpack.encode([request])

    raw_payload = msgspec.msgpack.decode(encoded)
    assert isinstance(raw_payload, list)
    assert len(raw_payload) == 1
    assert isinstance(raw_payload[0], list)
    assert raw_payload[0][0] == "request-0"

    decoded = MsgpackDecoder(list[EngineCoreRequest]).decode(encoded)
    assert len(decoded) == 1
    assert isinstance(decoded[0], EngineCoreRequest)
    assert decoded[0].request_id == "request-0"


def test_process_input_sockets_adapter_preserves_upstream_handshake() -> None:
    """GR socket adapter is a vLLM-version wrapper: upstream handshake stays."""
    import inspect

    import vllm_gr.v1.engine.engine_core_patch as ecp

    ecp.apply_engine_core_request_patches()
    assert ecp.UPSTREAM_PROCESS_INPUT_SOCKETS is not None
    src = inspect.getsource(ecp.UPSTREAM_PROCESS_INPUT_SOCKETS)
    for token in (
        "READY",
        "ABORT",
        "put_nowait",
        "DEALER",
        "EngineCoreReadyResponse",
        "preprocess_add_request",
    ):
        assert token in src, token


def test_only_v1_extends_native_wire_types() -> None:
    register_engine_core_request_types()
    register_engine_core_request_types()
    actual = {name: member.value for name, member in EngineCoreRequestType.__members__.items()}
    expected = UPSTREAM_REQUEST_WIRE_VALUES | {"SUBMIT_GR_SESSION": b"\x0a"}
    assert actual == expected
    for name, value in expected.items():
        assert EngineCoreRequestType(value) is getattr(EngineCoreRequestType, name)


@pytest.mark.parametrize("value", [b"\x06", b"\x07", b"\x08", b"\x09"])
def test_retired_wire_bytes_rejected(value: bytes) -> None:
    register_engine_core_request_types()
    with pytest.raises(ValueError):
        EngineCoreRequestType(value)


def test_gr_dispatch_only_admits_v1(monkeypatch) -> None:
    from vllm_gr.v1.engine import core

    register_engine_core_request_types()
    handled = []
    engine = SimpleNamespace()
    batch = object()
    monkeypatch.setattr(core, "_admit_gr_session", lambda *args: handled.append(args))
    for kind in UPSTREAM_REQUEST_WIRE_VALUES:
        assert not core.dispatch_gr_engine_core_input(
            engine, getattr(EngineCoreRequestType, kind), batch
        )
    assert core.dispatch_gr_engine_core_input(
        engine, EngineCoreRequestType.SUBMIT_GR_SESSION, batch
    )
    assert handled == [(engine, batch)]
