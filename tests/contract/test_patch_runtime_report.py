# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
import yaml

from vllm_gr import patch_report as reports
from vllm_gr.compat.runtime_v0_30 import (
    RuntimeContractError,
    check_method,
    resolve_symbol,
)
from vllm_gr.patch_registry import BY_NAME

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)  # type: ignore[misc]
def fresh_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reports, "_REPORTS", {})
    monkeypatch.setattr(reports, "_PID", os.getpid())


def results(phase: str = "frontend") -> Any:
    return cast(dict[str, Any], reports.get_runtime_patch_report()["phases"])[phase]


def test_real_install_reentry_and_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.logprobs import append_fast
    from vllm_gr.logprobs_patch import patch_flat_logprobs

    owner = resolve_symbol("FlatLogprobs")
    monkeypatch.setattr(owner, "append_fast", lambda self: None)
    with reports.patch_phase("frontend", expected=("flat_logprobs",)):
        patch_flat_logprobs()
    assert owner.append_fast is append_fast
    item = results()["flat_logprobs"]
    assert item["patch_id"] == "GR-PATCH-003"
    assert item["status"] == "applied"
    assert item["targets"][0]["path"].endswith("FlatLogprobs.append_fast")
    assert item["targets"][0]["present"] is True
    with reports.patch_phase("frontend", expected=("flat_logprobs",)):
        patch_flat_logprobs()
    assert results()["flat_logprobs"]["status"] == "already_applied"
    monkeypatch.setattr(owner, "append_fast", lambda self: None)
    with reports.patch_phase("frontend", expected=("flat_logprobs",)):
        patch_flat_logprobs()
    assert results()["flat_logprobs"]["status"] == "applied"


def test_failed_and_blocked_are_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.logprobs_patch import patch_flat_logprobs

    monkeypatch.delattr(resolve_symbol("FlatLogprobs"), "append_fast")
    with pytest.raises(RuntimeContractError, match="append_fast"):
        with reports.patch_phase("frontend", expected=("flat_logprobs", "engine_args")):
            patch_flat_logprobs()
    assert results()["flat_logprobs"]["status"] == "failed"
    assert results()["engine_args"]["status"] == "blocked"


def test_postcondition_failure_cannot_report_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr import logprobs_patch

    monkeypatch.setattr(logprobs_patch, "_append_fast", lambda self: None)
    with pytest.raises(RuntimeContractError, match="GR-PATCH-003"):
        with reports.patch_phase("frontend", expected=("flat_logprobs",)):
            logprobs_patch.patch_flat_logprobs()
    assert results()["flat_logprobs"]["status"] == "failed"


def test_phase_finalization_detects_later_overwrite(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.logprobs_patch import patch_flat_logprobs

    owner = resolve_symbol("FlatLogprobs")
    monkeypatch.setattr(owner, "append_fast", lambda self: None)
    with pytest.raises(RuntimeContractError, match="GR-PATCH-003"):
        with reports.patch_phase("frontend", expected=("flat_logprobs",)):
            patch_flat_logprobs()
            owner.append_fast = lambda self: None
    assert results()["flat_logprobs"]["status"] == "failed"


def test_nested_phases_keep_actual_applied_status(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.logprobs_patch import patch_flat_logprobs

    monkeypatch.setattr(resolve_symbol("FlatLogprobs"), "append_fast", lambda self: None)
    with reports.patch_phase("runtime", expected=("flat_logprobs",)):
        with reports.patch_phase("frontend", expected=("flat_logprobs",)):
            patch_flat_logprobs()
    assert results()["flat_logprobs"]["status"] == "applied"
    assert results("runtime")["flat_logprobs"]["status"] == "applied"


@pytest.mark.parametrize("platform", ["cpu", "cuda", "npu"])  # type: ignore[misc]
def test_unselected_platform_is_not_imported(
    monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    module = ModuleType("vllm.platforms")
    setattr(module, "current_platform", SimpleNamespace(device_type=platform))
    monkeypatch.setitem(sys.modules, "vllm.platforms", module)
    excluded = "cuda_model_runner" if platform == "npu" else "npu_model_runner"
    assert not reports.applicable(BY_NAME[excluded])
    with reports.patch_phase("engine_core", expected=(excluded,)):
        pass
    assert results("engine_core")[excluded]["status"] == "not_applicable"
    assert results("engine_core")[excluded]["targets"] == []


def test_deferred_worker_hooks_are_not_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reports, "runtime_platform", lambda: "npu")
    with reports.patch_phase("runtime", expected=()):
        pass
    assert results("runtime")["penalties_warmup"]["status"] == "deferred"
    assert results("runtime")["npu_kv_layout"]["status"] == "deferred"


def test_required_npu_symbol_failure_is_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reports, "runtime_platform", lambda: "npu")
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.worker", None)
    with pytest.raises(ImportError):
        with reports.patch_phase("worker_init", expected=("npu_worker_init",)):
            pass
    assert results("worker_init")["npu_worker_init"]["status"] == "failed"


def test_descriptor_and_signature_rejected_before_install() -> None:
    class Args:
        def add_cli_args(self, wrong: object) -> None:
            pass

    with pytest.raises(RuntimeContractError, match="staticmethod"):
        check_method(Args, "add_cli_args", descriptor=staticmethod)
    with pytest.raises(RuntimeContractError, match="parameters"):
        check_method(Args, "add_cli_args", parameters=("parser",))


def test_spawn_alias_drift_is_repaired(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.npu_spawn_patch import apply_npu_spawn_patch

    system = resolve_symbol("system_utils")
    consumer = resolve_symbol("multiproc_executor")
    monkeypatch.setattr(system, "_maybe_force_spawn", lambda: None)
    monkeypatch.setattr(consumer, "_maybe_force_spawn", lambda: None)
    with reports.patch_phase("runtime", expected=("npu_spawn",)):
        assert apply_npu_spawn_patch()
    wrapper = system._maybe_force_spawn
    monkeypatch.setattr(consumer, "_maybe_force_spawn", lambda: None)
    with reports.patch_phase("runtime", expected=("npu_spawn",)):
        assert apply_npu_spawn_patch()
    assert system._maybe_force_spawn is consumer._maybe_force_spawn is wrapper
    assert results("runtime")["npu_spawn"]["status"] == "applied"


def test_fork_discards_parent_reports() -> None:
    if not hasattr(os, "fork"):
        pytest.skip("fork is unavailable")
    code = """
import json, os
from vllm_gr import patch_report as reports
reports._REPORTS['parent'] = {}
read_fd, write_fd = os.pipe()
child = os.fork()
if child == 0:
    try:
        os.close(read_fd)
        os.write(write_fd, json.dumps(reports.get_runtime_patch_report()).encode())
    finally:
        os._exit(0)
os.close(write_fd)
with os.fdopen(read_fd) as stream:
    data = json.load(stream)
_, status = os.waitpid(child, 0)
assert status == 0
assert data['pid'] == child
assert data['phases'] == {}
assert 'parent' in reports.get_runtime_patch_report()['phases']
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=30
    )


def test_fresh_process_imports_no_device_runtime() -> None:
    code = """
import json, sys
from vllm_gr.patch_report import get_runtime_patch_report
assert not any(name == 'torch' or name.startswith(('vllm.', 'vllm_ascend.')) for name in sys.modules)
print(json.dumps(get_runtime_patch_report()))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=30
    )
    data = json.loads(completed.stdout)
    assert data["pid"] != os.getpid()
    assert data["phases"] == {}


def test_inventory_contract_and_negative_drift(tmp_path: Path) -> None:
    import shutil

    checker = runpy.run_path(str(ROOT / "tools/check_patch_inventory.py"))["check"]
    checker(ROOT)
    for name in ("docs", "tests", "vllm_gr"):
        (tmp_path / name).symlink_to(ROOT / name, target_is_directory=True)
    (tmp_path / "docs").unlink()
    (tmp_path / "docs").mkdir()
    shutil.copy(ROOT / "docs/runtime_patch_inventory.md", tmp_path / "docs")
    inventory = yaml.safe_load((ROOT / "docs/patch_inventory.yaml").read_text())
    inventory["patches"][0]["installations"][0]["installer"] = "missing.installer"
    (tmp_path / "docs/patch_inventory.yaml").write_text(yaml.safe_dump(inventory))
    with pytest.raises(AssertionError, match="contract drift"):
        checker(tmp_path)


def test_built_wheel_report_without_source_tree(tmp_path: Path) -> None:
    import shutil
    import zipfile

    source = tmp_path / "source"
    source.mkdir()
    for name in ("setup.py", "pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / name, source / name)
    shutil.copytree(ROOT / "requirements", source / "requirements")
    shutil.copytree(
        ROOT / "vllm_gr",
        source / "vllm_gr",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so"),
    )
    env = dict(os.environ, VLLM_GR_TARGET_DEVICE="cpu", VLLM_GR_VERSION_OVERRIDE="0.30.0+cpu")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            "dist",
        ],
        cwd=source,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    wheel = next((source / "dist").glob("*.whl"))
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(installed)
    shutil.rmtree(source)
    code = """
import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
import vllm_gr
from vllm_gr.patch_report import get_runtime_patch_report, patch_phase
assert pathlib.Path(vllm_gr.__file__).is_relative_to(sys.argv[1])
assert not pathlib.Path('docs/patch_inventory.yaml').exists()
with patch_phase('runtime', expected=()):
    pass
assert not any(name == 'torch' or name.startswith(('vllm.', 'vllm_ascend.')) for name in sys.modules)
print(json.dumps(get_runtime_patch_report()))
"""
    run = subprocess.run(
        [sys.executable, "-I", "-c", code, str(installed)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "runtime" in json.loads(run.stdout)["phases"]


def test_scheduler_subtarget_drift_is_rejected_and_repaired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.v1.engine.engine_core_patch import apply_scheduler_patch

    apply_scheduler_patch()
    owner = resolve_symbol("Scheduler")
    monkeypatch.setattr(owner, "add_request", lambda self, request: None)
    with pytest.raises(RuntimeContractError, match="Scheduler.add_request"):
        with reports.patch_phase("engine_core", expected=("scheduler",)):
            pass
    with reports.patch_phase("engine_core", expected=("scheduler",)):
        apply_scheduler_patch()
    assert results("engine_core")["scheduler"]["status"] == "applied"


def test_memo_consumer_binding_drift_is_repaired(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.engine.engine_core_patch import apply_kv_hash_memo_patch

    apply_kv_hash_memo_patch()
    utils = resolve_symbol("kv_cache_utils")
    core = resolve_symbol("engine_core_module")
    original = utils.get_request_block_hasher
    monkeypatch.setattr(core, "get_request_block_hasher", lambda: None)
    with reports.patch_phase("engine_core", expected=("kv_hash_memo",)):
        apply_kv_hash_memo_patch()
    assert core.get_request_block_hasher is original
    assert results("engine_core")["kv_hash_memo"]["status"] == "applied"


def test_missing_target_during_finalization_is_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.logprobs_patch import patch_flat_logprobs

    owner = resolve_symbol("FlatLogprobs")
    with pytest.raises(AttributeError):
        with reports.patch_phase("frontend", expected=("flat_logprobs",)):
            patch_flat_logprobs()
            monkeypatch.delattr(owner, "append_fast")
    assert results()["flat_logprobs"]["status"] == "failed"


def test_inventory_rejects_unowned_native_mutation() -> None:
    import ast

    namespace = runpy.run_path(str(ROOT / "tools/check_patch_inventory.py"))
    tree = ast.parse("EngineArgs.untracked_method = replacement")
    with pytest.raises(AssertionError, match="Unowned native mutation"):
        namespace["check_mutations"](tree, Path("example.py"))


def test_beam_dispatch_validator_checks_v1_implementation(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.entrypoints.openai.beam_search_patch import patch_beam_search
    from vllm_gr.patch import _validate_beam_search_dispatch_patch

    patch_beam_search()
    monkeypatch.delattr(resolve_symbol("GenerateBaseServing"), "beam_search_v1")
    assert _validate_beam_search_dispatch_patch() is not None


def test_npu_validation_ignores_cuda_scoring_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.ops.sliced_logits_patch import apply_sliced_logits_patch
    from vllm_gr.patch_verification import verify_sliced_logits

    apply_sliced_logits_patch()
    module = ModuleType("vllm.v1.worker.gpu.sample.logprob")
    setattr(module, "compute_topk_scores", lambda: None)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(reports, "runtime_platform", lambda: "npu")
    assert verify_sliced_logits() is None
    monkeypatch.setattr(reports, "runtime_platform", lambda: "cuda")
    assert "compute_topk_scores is not clamped" in (verify_sliced_logits() or "")
