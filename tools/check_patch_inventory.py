# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Check installation declarations against source, tests, YAML, and Markdown."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vllm_gr.compat.runtime_v0_30 import SYMBOLS  # noqa: E402
from vllm_gr.patch_registry import (  # noqa: E402
    BINDINGS,
    INSTALLATIONS,
    RETIRED_PATCH_IDS,
)

START = "<!-- patch-contract-table:start -->"
END = "<!-- patch-contract-table:end -->"


def overview(patches: list[dict[str, Any]]) -> str:
    lines = [
        START,
        "| ID | Installation responsibilities | Phases | Condition |",
        "| --- | --- | --- | --- |",
    ]
    for patch in patches:
        actions = patch["installations"]
        names = ", ".join(f"`{item['name']}`" for item in actions)
        phases = ", ".join(patch["phases"])
        conditions = ", ".join(patch["required_when"])
        lines.append(f"| {patch['id']} | {names} | {phases} | {conditions} |")
    return "\n".join([*lines, END])


def check_mutations(tree: ast.Module, path: Path) -> None:
    """Check direct writes to named native owners independently of declarations."""
    bindings = {binding for group in BINDINGS.values() for binding in group}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Attribute) or not isinstance(target.value, ast.Name):
                continue
            if target.value.id not in SYMBOLS:
                continue
            # Legacy diagnostic flags are deliberately not installation evidence.
            if target.attr == "_patched_for_beam_requests":
                continue
            binding = f"{target.value.id}.{target.attr}"
            assert binding in bindings, f"Unowned native mutation {binding} in {path}"


def check(root: Path = ROOT) -> None:
    inventory = yaml.safe_load((root / "docs/patch_inventory.yaml").read_text())
    assert inventory["schema_version"] == 3, "Inventory schema mismatch"
    patches = inventory["patches"]
    ids = [patch["id"] for patch in patches]
    assert len(ids) == len(set(ids)), "Duplicate patch ID"
    retired = {patch["id"] for patch in inventory["retired_patches"]}
    assert retired == set(RETIRED_PATCH_IDS), "Retired IDs changed"
    assert not set(ids) & retired, "Retired ID reused"
    assert set(ids) == {item.patch_id for item in INSTALLATIONS}, "Patch ownership mismatch"
    declared = {item.name: item.contract() for item in INSTALLATIONS}
    documented = [item for patch in patches for item in patch["installations"]]
    assert len(documented) == len(declared), "Missing or duplicate installation"
    assert {item["name"]: item for item in documented} == declared, "Installation contract drift"
    assert set(BINDINGS) == set(declared), "Missing target ownership"
    for patch in patches:
        items = [item for item in INSTALLATIONS if item.patch_id == patch["id"]]
        for key, values in {
            "installer": [item.installer for item in items],
            "validators": [item.validator for item in items if item.validator],
            "targets": [binding for item in items for binding in BINDINGS[item.name]],
            "phases": [phase for item in items for phase in item.phases],
            "required_when": [item.condition for item in items],
            "dependencies": [name for item in items for name in item.dependencies],
        }.items():
            assert patch[key] == list(dict.fromkeys(values)), f"{patch['id']} {key} drift"
        assert patch["tests"], f"{patch['id']} has no test coverage reference"
        for reference in patch["tests"]:
            file, _, name = reference.partition("::")
            path = root / file
            assert path.is_file(), f"Missing test file {file}"
            if name:
                tree = ast.parse(path.read_text())
                assert any(
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
                    for node in ast.walk(tree)
                ), f"Missing test {reference}"
    for item in INSTALLATIONS:
        assert set(item.dependencies) <= declared.keys(), f"Unknown dependency for {item.name}"
        for binding in BINDINGS[item.name]:
            assert binding.split(".", 1)[0] in SYMBOLS, f"Unknown target {binding}"
        module, name = item.installer.rsplit(".", 1)
        path = root / (module.replace(".", "/") + ".py")
        tree = ast.parse(path.read_text())
        if item.kind == "registration":
            assert any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == name
                for node in ast.walk(tree)
            ), f"Missing registration {item.name}"
            assert '"attention_registration"' in path.read_text(), "Registration is not observed"
        else:
            function = next(
                (
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name
                ),
                None,
            )
            assert function is not None, f"Missing installer {item.installer}"
            assert any(
                isinstance(node, ast.Name) and node.id == "observed_installation"
                for node in function.decorator_list
            ), f"Unobserved installer {item.installer}"
        if item.validator:
            module, name = item.validator.rsplit(".", 1)
            tree = ast.parse((root / (module.replace(".", "/") + ".py")).read_text())
            assert any(
                isinstance(node, ast.FunctionDef) and node.name == name for node in tree.body
            ), f"Missing validator {item.validator}"
    # Reverse lookup catches newly observed installers without an inventory entry.
    owned = {item.installer for item in INSTALLATIONS}
    for path in (root / "vllm_gr").rglob("*.py"):
        if "generated" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        check_mutations(tree, path)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and any(
                isinstance(d, ast.Name) and d.id == "observed_installation"
                for d in node.decorator_list
            ):
                module = ".".join(path.relative_to(root).with_suffix("").parts)
                assert f"{module}.{node.name}" in owned, f"Unowned installer {module}.{node.name}"
    markdown = (root / "docs/runtime_patch_inventory.md").read_text()
    assert overview(patches) in markdown, "Markdown overview drift"
    headings = [
        line.split()[1] for line in markdown.splitlines() if line.startswith("### GR-PATCH-")
    ]
    assert headings == ids, "Markdown patch ID drift"


if __name__ == "__main__":
    check()
    print("Patch inventory contracts passed")
