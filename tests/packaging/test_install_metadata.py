# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke tests for platform-aware install metadata routing."""

from __future__ import annotations

import ast
import importlib.util
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_FILE = REPO_ROOT / "requirements" / "common.txt"
PYPROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


def _load_setup_module() -> ModuleType:
    setup_path = REPO_ROOT / "setup.py"
    spec = importlib.util.spec_from_file_location("vllm_gr_setup", setup_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["vllm_gr_setup"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(  # type: ignore[misc]
    ("device", "expected_subset", "expected_packages", "excluded_packages"),
    [
        (
            "cpu",
            {"msgspec", "regex", "safetensors>=0.6.2", "PyYAML>=6.0"},
            {"msgspec", "regex", "safetensors", "pyyaml"},
            {"vllm"},
        ),
        (
            "cuda",
            {
                "msgspec",
                "regex",
                "safetensors>=0.6.2",
                "PyYAML>=6.0",
                "vllm==0.30.0",
            },
            {"msgspec", "regex", "safetensors", "pyyaml", "vllm"},
            set(),
        ),
        (
            "npu",
            {"msgspec", "regex", "safetensors>=0.6.2", "PyYAML>=6.0"},
            {"msgspec", "regex", "safetensors", "pyyaml"},
            {"vllm"},
        ),
    ],
)
def test_get_install_requires_by_device(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    expected_subset: set[str],
    expected_packages: set[str],
    excluded_packages: set[str],
) -> None:
    gr_setup = _load_setup_module()
    # Reset the module-level cache so the monkeypatched function is used.
    monkeypatch.setattr(gr_setup, "_cached_device", None)
    monkeypatch.setattr(gr_setup, "detect_target_device", lambda: device)
    monkeypatch.setattr(gr_setup, "warn_deprecated_extra_install", lambda: None)

    requirements = set(gr_setup.get_install_requires())
    package_names = {
        Requirement(requirement).name.lower().replace("_", "-") for requirement in requirements
    }

    assert expected_subset.issubset(requirements)
    assert expected_packages.issubset(package_names)
    assert package_names.isdisjoint(excluded_packages)


def test_common_requirements_exclude_vllm_and_dev_tools() -> None:
    content = COMMON_FILE.read_text()
    lines = {
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert "pytest" not in lines
    assert "ruff" not in lines


def test_release_version_by_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_versions = {
        "cuda": "0.30.0",
        "npu": "0.30.0+npu",
        "cpu": "0.30.0+cpu",
    }

    for device, expected in expected_versions.items():
        gr_setup = _load_setup_module()
        monkeypatch.delenv("VLLM_GR_VERSION_OVERRIDE", raising=False)
        monkeypatch.setattr(gr_setup, "detect_target_device", lambda device=device: device)
        write_version = MagicMock()
        monkeypatch.setattr(gr_setup, "write_version_file", write_version)

        assert gr_setup.get_vllm_gr_version() == expected
        assert gr_setup.VLLM_GR_BASE_VERSION == "0.30.0"
        write_version.assert_called_once_with(expected)


def test_release_version_override(monkeypatch: pytest.MonkeyPatch) -> None:
    gr_setup = _load_setup_module()
    monkeypatch.setenv("VLLM_GR_VERSION_OVERRIDE", "0.30.0.dev1")
    write_version = MagicMock()
    monkeypatch.setattr(gr_setup, "write_version_file", write_version)

    assert gr_setup.get_vllm_gr_version() == "0.30.0.dev1"
    write_version.assert_called_once_with("0.30.0.dev1")


@pytest.mark.parametrize(  # type: ignore[misc]
    ("version", "expected_tuple"),
    [
        ("0.30.0", (0, 30, 0)),
        ("0.30.0+npu", (0, 30, 0, "npu")),
        ("0.30.0+cpu", (0, 30, 0, "cpu")),
        ("0.30.0.dev1", (0, 30, 0, "dev1")),
    ],
)
def test_version_file_matches_runtime_contract(
    tmp_path: Path, version: str, expected_tuple: tuple[int | str, ...]
) -> None:
    gr_setup = _load_setup_module()
    path = tmp_path / "_version.py"
    gr_setup.write_version_file(version, path)

    namespace: dict[str, object] = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    # vllm_gr/version.py and tools/run_wheel_smoke_test.sh read these names.
    assert namespace["__version__"] == namespace["version"] == version
    assert namespace["__version_tuple__"] == namespace["version_tuple"] == expected_tuple
    assert namespace["__commit_id__"] is None


def test_setup_is_the_only_version_mechanism() -> None:
    build_requires = {
        Requirement(value).name.lower() for value in PYPROJECT["build-system"]["requires"]
    }
    assert "setuptools-scm" not in build_requires
    assert "setuptools_scm" not in PYPROJECT.get("tool", {})
    assert "setuptools_scm" not in (REPO_ROOT / "setup.py").read_text()
    assert PYPROJECT["project"]["dynamic"] == ["version", "dependencies"]


def test_missing_requirements_file_fails_build(tmp_path: Path) -> None:
    gr_setup = _load_setup_module()
    with pytest.raises(FileNotFoundError, match="missing.txt"):
        gr_setup.load_requirements(tmp_path / "missing.txt")

    nested = tmp_path / "cuda.txt"
    nested.write_text("-r common.txt\nvllm==0.30.0\n")
    with pytest.raises(FileNotFoundError, match="common.txt"):
        gr_setup.load_requirements(nested)


def test_package_data_declares_non_python_package_files() -> None:
    """Every non-Python file in the package must ship without a .git directory."""
    declared = {
        pattern
        for patterns in PYPROJECT["tool"]["setuptools"]["package-data"].values()
        for pattern in patterns
    }
    for pattern in ("_version.py", "proto/*.proto", "generated/README.md"):
        assert pattern in declared

    for path in (REPO_ROOT / "vllm_gr").rglob("*"):
        relative = path.relative_to(REPO_ROOT / "vllm_gr")
        if path.suffix in {".py", ".pyc"} or not path.is_file():
            continue
        assert any(relative.match(pattern) for pattern in declared), (
            f"vllm_gr/{relative} is not declared in [tool.setuptools.package-data]"
        )


def test_requirement_pin_policy() -> None:
    """Only the patched runtime and ABI-coupled packages may use exact pins."""
    allowed_exact = {"vllm", "numpy"}
    for path in sorted((REPO_ROOT / "requirements").glob("*.txt")):
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith("-r "):
                continue
            requirement = Requirement(line)
            if any(spec.operator == "==" for spec in requirement.specifier):
                assert requirement.name.lower() in allowed_exact, f"{path.name}: {line}"


def test_dev_extra_uses_ruff_formatter_only() -> None:
    dev = {
        Requirement(value).name.lower()
        for value in PYPROJECT["project"]["optional-dependencies"]["dev"]
    }
    assert "black" not in dev
    assert "ruff" in dev
    assert "black" not in PYPROJECT.get("tool", {})


def _literal(path: Path, name: str) -> str:
    for node in ast.parse(path.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
            and isinstance(node.value, ast.Constant)
        ):
            return str(node.value.value)
    raise AssertionError(f"{path}: no literal {name}")


def test_readme_states_runtime_version_contract() -> None:
    readme = (REPO_ROOT / "README.md").read_text()
    contract = readme.split("## Runtime version contract", 1)[1].split("\n## ", 1)[0]
    env_check = REPO_ROOT / "vllm_gr" / "_env_check.py"
    base = _literal(REPO_ROOT / "setup.py", "VLLM_GR_BASE_VERSION")
    for value in (
        f"vllm-gr **{base}**",
        f"vLLM `{_literal(env_check, '_SUPPORTED_VLLM_VERSION')}`",
        f"vLLM-Ascend `{_literal(env_check, '_SUPPORTED_VLLM_ASCEND_VERSION')}`",
        f"Triton-Ascend `{_literal(env_check, '_SUPPORTED_TRITON_ASCEND_VERSION')}`",
        f"`{base}+npu`",
    ):
        assert value in contract, value
    # The first screen of the README, before installation details.
    assert readme.index("## Runtime version contract") < readme.index("## Installation")


def test_warn_deprecated_extra_install_emits_warning() -> None:
    import warnings

    gr_setup = _load_setup_module()
    original_argv = sys.argv
    try:
        sys.argv = ["pip", "install", "-e", ".[gpu]"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gr_setup.warn_deprecated_extra_install()
        assert any(isinstance(w.message, DeprecationWarning) for w in caught)
    finally:
        sys.argv = original_argv
