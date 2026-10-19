# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared benchmark utilities; device/model imports occur only when building artifacts."""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
from pathlib import Path
from typing import Literal


def _ordered_samples(values: list[float]) -> list[float]:
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("latency samples must be finite")
    return ordered


def _percentile(ordered: list[float], quantile: float, method: Literal["linear", "lower"]) -> float:
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    if method == "lower":
        return ordered[lower]
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def percentile(
    values: list[float], quantile: float, *, method: Literal["linear", "lower"] = "linear"
) -> float | None:
    """Preserve the explicit historical percentile policy of each benchmark."""
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    if method not in ("linear", "lower"):
        raise ValueError(f"unknown percentile method: {method}")
    if not values:
        return None
    return _percentile(_ordered_samples(values), quantile, method)


def distribution_ms(
    values: list[float], *, method: Literal["linear", "lower"] = "linear"
) -> dict[str, float | str]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    if method not in ("linear", "lower"):
        raise ValueError(f"unknown percentile method: {method}")
    ordered = _ordered_samples(values)
    return {
        "mean": statistics.fmean(ordered),
        "std": statistics.pstdev(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        **{f"p{p}": _percentile(ordered, p / 100, method) for p in (50, 90, 95, 99)},
        "unit": "ms",
    }


def benchmark_provenance(repo: Path) -> dict[str, str | None]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "commit": commit,
        "process_mode": (
            "multi" if int(os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1")) else "single"
        ),
    }


def load_triples(path: Path) -> list[list[str]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("triples") if isinstance(document, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} is not a nonempty Constraint Table triples document")
    triples = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not all(key in row for key in ("a", "b", "c")):
            raise ValueError(f"invalid triples row {index}: {row!r}")
        triples.append([str(row["a"]), str(row["b"]), str(row["c"])])
    return sorted(triples)


def build_artifact(
    model: Path,
    catalog: Path,
    artifact_path: Path,
) -> tuple[str, str]:
    from transformers import AutoConfig, AutoTokenizer

    from vllm_gr.constraints.artifact import save_constraint_table_artifact
    from vllm_gr.constraints.artifact_compiler import ConstraintTableArtifactCompiler
    from vllm_gr.constraints.json_compiler import compile_relations_json
    from vllm_gr.constraints.tokenizer_digest import compute_tokenizer_digest

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    tokenizer_digest = compute_tokenizer_digest(tokenizer)
    table = ConstraintTableArtifactCompiler(
        tokenizer,
        int(model_config.vocab_size),
        tokenizer_digest=tokenizer_digest,
    ).compile(compile_relations_json(load_triples(catalog)))
    artifact_digest = save_constraint_table_artifact(table, artifact_path)
    return artifact_digest, tokenizer_digest
