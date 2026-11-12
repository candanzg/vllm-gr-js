# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Process-local observations of real installer calls and their postconditions."""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import ParamSpec, TypeVar

from vllm_gr.compat.runtime_v0_30 import (
    SYMBOLS,
    RuntimeContractError,
    check_structure,
    resolve_symbol,
)
from vllm_gr.patch_registry import (
    BINDINGS,
    BY_NAME,
    INSTALLATIONS,
    Installation,
    for_installer,
)

P = ParamSpec("P")
T = TypeVar("T")
_PHASE: ContextVar[str | None] = ContextVar("gr_patch_phase", default=None)
_PID = os.getpid()
_REPORTS: dict[str, dict[str, dict[str, object]]] = {}
logger = logging.getLogger(__name__)


def _reset_after_fork() -> None:
    global _PID
    if _PID != os.getpid():
        _REPORTS.clear()
        _PID = os.getpid()
        _PHASE.set(None)


def runtime_platform() -> str:
    module = sys.modules.get("vllm.platforms")
    return str(getattr(getattr(module, "current_platform", None), "device_type", "unknown"))


def applicable(item: Installation) -> bool:
    if item.condition == "always":
        return True
    if "vllm.platforms" not in sys.modules:
        importlib.import_module("vllm.platforms")
    if item.condition in {"cuda", "npu"}:
        return runtime_platform() == item.condition
    if item.condition == "npu_weight_update_api":
        if runtime_platform() != "npu":
            return False
        worker = resolve_symbol("NPUWorker")
        return all(
            callable(getattr(worker, name, None))
            for name in ("start_weight_update", "update_weights", "finish_weight_update")
        )
    if item.condition == "supported_npu_kv":
        if runtime_platform() != "npu":
            return False
        from vllm_gr.v1.npu_kv_layout import (
            _SUPPORTED_ASCEND_VERSION,
            _compatible_policy_pair,
        )

        if version("vllm-ascend") != _SUPPORTED_ASCEND_VERSION:
            return False
        utils = resolve_symbol("ascend_attention_utils")
        runner = resolve_symbol("ascend_runner_module")
        first = getattr(utils, "requires_contiguous_pa_kv_cache", None)
        second = getattr(runner, "requires_contiguous_pa_kv_cache", None)
        return bool(_compatible_policy_pair(first, second))
    raise RuntimeContractError(f"Unknown activation condition {item.condition}")


def selected_symbols(item: Installation) -> tuple[str, ...]:
    names = item.symbols
    if item.name == "cuda_model_runner":
        from vllm_gr.v1.worker.model_runner_patch import _use_v2_model_runner

        runner = "GPUModelRunnerV2" if _use_v2_model_runner() else "GPUModelRunner"
        return (runner,) + tuple(name for name in names if not name.startswith("GPUModelRunner"))
    return names


def verify_installation(item: Installation) -> str | None:
    if not item.validator:
        return None
    module, name = item.validator.rsplit(".", 1)
    check: Callable[[], str | None] = getattr(importlib.import_module(module), name)
    error = check()
    if error:
        return error
    for target in target_details(item):
        if not target["present"]:
            return f"Missing installed target {target['path']}"
    return None


def target_details(item: Installation) -> list[dict[str, object]]:
    selected = selected_symbols(item)
    result: list[dict[str, object]] = []
    for binding in BINDINGS[item.name]:
        symbol, _, attribute = binding.partition(".")
        if symbol not in selected:
            continue
        owner = resolve_symbol(symbol, loaded_only=True)
        result.append(
            {
                "symbol": symbol,
                "path": SYMBOLS[symbol].path + "." + attribute,
                "present": getattr(owner, attribute, None) is not None,
            }
        )
    return result


def binding_snapshot(item: Installation) -> tuple[object, ...]:
    selected = selected_symbols(item)
    values: list[object] = []
    for binding in BINDINGS[item.name]:
        symbol, _, attribute = binding.partition(".")
        if symbol in selected:
            owner = resolve_symbol(symbol, loaded_only=True)
            target = getattr(owner, attribute, None)
            # Bound classmethods create a fresh method object on each access.
            if attribute == "model_fields" and isinstance(target, dict):
                values.append(tuple((key, repr(field.annotation)) for key, field in target.items()))
            else:
                values.append(id(getattr(target, "__func__", target)))
    return tuple(values)


def _record(item: Installation, status: str, detail: str | None = None) -> None:
    phase = _PHASE.get()
    if phase is None:
        return
    targets: list[dict[str, object]] = []
    if status in {"applied", "already_applied"}:
        targets = target_details(item)
    _REPORTS.setdefault(phase, {})[item.name] = {
        "patch_id": item.patch_id,
        "name": item.name,
        "phase": phase,
        "process_role": {"engine_core": "engine_core", "worker_init": "worker"}.get(
            phase, "unknown"
        ),
        "platform": runtime_platform(),
        "runner": ("v2" if "GPUModelRunnerV2" in selected_symbols(item) else "v1")
        if item.name in {"cuda_model_runner", "npu_model_runner"}
        and status in {"applied", "already_applied"}
        else None,
        "status": status,
        "detail": detail,
        "targets": targets,
    }


def observe_existing(names: tuple[str, ...]) -> None:
    """Re-probe skipped installers; cached initialization flags are not evidence."""
    for name in names:
        item = BY_NAME[name]
        try:
            if not applicable(item):
                _record(item, "not_applicable", f"Condition {item.condition} is false")
                continue
            check_structure(item.name)
            error = verify_installation(item)
            if error:
                raise RuntimeContractError(f"{item.patch_id}/{item.name}: {error}")
            _record(item, "already_applied")
        except Exception as exc:
            _record(item, "failed", f"{type(exc).__name__}: {exc}")
            raise


def observed_installation(function: Callable[P, T]) -> Callable[P, T]:
    """Observe installed targets inside startup scopes without changing return values."""
    path = function.__module__ + "." + function.__name__
    declarations = for_installer(path)
    if not declarations:
        # File-based contract tests may load an installer under an isolated name.
        source = Path(function.__code__.co_filename)
        try:
            relative = source.relative_to(Path(__file__).parent).with_suffix("")
        except ValueError:
            pass
        else:
            path = "vllm_gr." + ".".join(relative.parts) + "." + function.__name__
            declarations = for_installer(path)
    if not declarations:
        raise ValueError(f"Installer has no patch ownership: {path}")

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        _reset_after_fork()
        if _PHASE.get() is None:
            return function(*args, **kwargs)
        active: list[Installation] = []
        before: dict[str, tuple[object, ...]] = {}
        try:
            for item in declarations:
                if applicable(item):
                    active.append(item)
                    check_structure(item.name)
                    before[item.name] = binding_snapshot(item)
                else:
                    _record(item, "not_applicable", f"Condition {item.condition} is false")
            result = function(*args, **kwargs)
            for item in active:
                error = verify_installation(item)
                if error:
                    raise RuntimeContractError(f"{item.patch_id}/{item.name}: {error}")
                after = binding_snapshot(item)
                changed = after != before[item.name]
                _record(
                    item, "applied" if changed or item.kind == "lifecycle" else "already_applied"
                )
            return result
        except Exception as exc:
            for item in active or declarations:
                _record(item, "failed", f"{type(exc).__name__}: {exc}")
            raise

    return wrapped


@contextmanager
def patch_phase(phase: str, *, expected: tuple[str, ...] | None = None) -> Iterator[None]:
    _reset_after_fork()
    parent = _PHASE.get()
    names = (
        expected
        if expected is not None
        else tuple(item.name for item in INSTALLATIONS if phase in item.phases)
    )
    if parent != phase:
        _REPORTS[phase] = {}
    token = _PHASE.set(phase)
    try:
        yield
        for name in names:
            existing = _REPORTS[phase].get(name)
            if existing and existing["status"] in {"applied", "already_applied"}:
                try:
                    error = verify_installation(BY_NAME[name])
                    if error:
                        raise RuntimeContractError(f"{BY_NAME[name].patch_id}/{name}: {error}")
                except Exception as exc:
                    _record(BY_NAME[name], "failed", f"{type(exc).__name__}: {exc}")
                    raise
            elif existing and existing["status"] in {"failed", "blocked"}:
                raise RuntimeContractError(f"{BY_NAME[name].patch_id}/{name}: {existing['detail']}")
            if name not in _REPORTS[phase]:
                item = BY_NAME[name]
                if item.kind == "lifecycle":
                    _record(item, "deferred", "Lifecycle entrypoint was not invoked in this scope")
                else:
                    observe_existing((name,))
        if phase == "runtime":
            for item in INSTALLATIONS:
                if item.phases == ("worker_init",) and item.name not in _REPORTS[phase]:
                    status = "deferred" if runtime_platform() == "npu" else "not_applicable"
                    _record(item, status, "Requires NPU worker device initialization")
    except Exception as exc:
        for name in names:
            if name not in _REPORTS[phase]:
                item = BY_NAME[name]
                _record(item, "blocked", f"Initialization stopped: {type(exc).__name__}: {exc}")
        raise
    finally:
        _PHASE.reset(token)
        if parent != phase:
            if parent is not None:
                for name, result in _REPORTS[phase].items():
                    _REPORTS[parent][name] = dict(result, phase=parent)
            logger.info(
                "GR_PATCH_REPORT %s", json.dumps(get_runtime_patch_report(), sort_keys=True)
            )


def reported_phase(
    phase: str, *, expected: tuple[str, ...] | None = None
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Attach reporting to a startup boundary, preserving its public signature."""

    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            with patch_phase(phase, expected=expected):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def get_runtime_patch_report() -> dict[str, object]:
    """Return a JSON-safe snapshot; inherited parent-process results are discarded."""
    _reset_after_fork()
    versions = {}
    for name in ("vllm", "vllm-ascend"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    return {
        "schema_version": 1,
        "pid": _PID,
        "runtime_versions": versions,
        "phases": json.loads(json.dumps(_REPORTS)),
    }
