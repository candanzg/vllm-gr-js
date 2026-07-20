# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lightweight performance logging helpers.

The helpers in this module are intentionally small and dependency-free so they
can be imported from runtime patch paths without pulling in heavy extras.
They emit structured JSON log lines that are easy to aggregate into CSV files
like ``performance_examle.csv``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from vllm_gr.logger import init_logger

logger = init_logger("vllm_gr.performance")
_PROBE_SEEN: set[tuple[str, str]] = set()


def perf_logging_enabled() -> bool:
    """Return True when runtime performance logging is enabled."""
    raw_value = os.getenv("VLLM_GR_PERF_LOG", "").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


def _clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def emit_probe_once(module: str, func: str, **fields: Any) -> None:
    """Emit a one-time lightweight probe line for path verification."""
    if not perf_logging_enabled():
        return
    key = (module, func)
    if key in _PROBE_SEEN:
        return
    _PROBE_SEEN.add(key)
    payload = {
        "event": "probe",
        "module": module,
        "func": func,
    }
    payload.update(fields)
    logger.info(json.dumps(_clean_payload(payload), ensure_ascii=False, sort_keys=True))


def log_perf_event(
    module: str,
    func: str,
    elapsed_s: float,
    *,
    request_id: str | None = None,
    phase: str | None = None,
    event: str = "function",
    **fields: Any,
) -> None:
    """Emit a structured performance log line."""
    if not perf_logging_enabled():
        return

    payload: dict[str, Any] = {
        "event": event,
        "module": module,
        "func": func,
        "duration_ms": round(elapsed_s * 1000.0, 3),
    }
    if request_id is not None:
        payload["request_id"] = request_id
    if phase is not None:
        payload["phase"] = phase
    payload.update(fields)
    logger.info(json.dumps(_clean_payload(payload), ensure_ascii=False, sort_keys=True))


@contextmanager
def perf_timer(
    module: str,
    func: str,
    *,
    request_id: str | None = None,
    phase: str | None = None,
    event: str = "function",
    **fields: Any,
) -> Iterator[None]:
    """Context manager that logs elapsed time on exit."""
    start = time.perf_counter()
    try:
        yield
    finally:
        log_perf_event(
            module,
            func,
            time.perf_counter() - start,
            request_id=request_id,
            phase=phase,
            event=event,
            **fields,
        )


def wrap_timed_method(
    method: Callable[..., Any],
    *,
    module: str,
    func: str,
    request_id_getter: Callable[..., str | None] | None = None,
    extra_getter: Callable[..., dict[str, Any]] | None = None,
) -> Callable[..., Any]:
    """Return a wrapper that logs method duration if perf logging is enabled."""

    from functools import wraps

    @wraps(method)
    def wrapped(*args: Any, **kwargs: Any):
        if not perf_logging_enabled():
            return method(*args, **kwargs)

        emit_probe_once(module, func)
        request_id = request_id_getter(*args, **kwargs) if request_id_getter else None
        extra = extra_getter(*args, **kwargs) if extra_getter else {}
        start = time.perf_counter()
        try:
            return method(*args, **kwargs)
        finally:
            log_perf_event(
                module,
                func,
                time.perf_counter() - start,
                request_id=request_id,
                **extra,
            )

    return wrapped


@dataclass
class RequestPerfTrace:
    """Collect per-request phase timings and emit a summary log."""

    request_id: str
    module: str
    fields: dict[str, Any] = field(default_factory=dict)
    phases: list[tuple[str, float]] = field(default_factory=list)
    _phase_start: float | None = field(default=None, init=False, repr=False)
    _phase_name: str | None = field(default=None, init=False, repr=False)
    _finished: bool = field(default=False, init=False, repr=False)

    def __enter__(self) -> "RequestPerfTrace":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish(error=None if exc is None else exc.__class__.__name__)

    @contextmanager
    def phase(self, phase_name: str, **extra_fields: Any) -> Iterator[None]:
        start = time.perf_counter()
        previous_name = self._phase_name
        previous_start = self._phase_start
        self._phase_name = phase_name
        self._phase_start = start
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.phases.append((phase_name, elapsed))
            payload_fields = _clean_payload({**self.fields, **extra_fields})
            log_perf_event(
                self.module,
                phase_name,
                elapsed,
                request_id=self.request_id,
                phase=phase_name,
                event="request_phase",
                **payload_fields,
            )
            self._phase_name = previous_name
            self._phase_start = previous_start

    def finish(self, **summary_fields: Any) -> None:
        if self._finished or not perf_logging_enabled():
            return
        self._finished = True
        payload: dict[str, Any] = {
            "event": "request_summary",
            "module": self.module,
            "request_id": self.request_id,
            "phases_ms": {
                phase_name: round(elapsed * 1000.0, 3) for phase_name, elapsed in self.phases
            },
        }
        payload.update(_clean_payload({**self.fields, **summary_fields}))
        logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True))
