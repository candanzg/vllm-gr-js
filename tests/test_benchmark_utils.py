# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import subprocess
import sys
import weakref
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Literal, Self, cast

import pytest

from benchmarks.qwen import benchmark_v1
from tools.benchmark_utils import distribution_ms, load_triples, percentile


def test_percentile_policies_preserve_historical_results() -> None:
    samples = [4.0, 1.0, 3.0, 2.0]
    assert percentile(samples, 0.5) == 2.5
    assert percentile(samples, 0.9) == pytest.approx(3.7)
    assert percentile(samples, 0.5, method="lower") == 2
    assert percentile(samples, 0.9, method="lower") == 3
    assert benchmark_v1._percentile(samples, 0.9) == 3
    assert samples == [4.0, 1.0, 3.0, 2.0]
    result = distribution_ms(samples)
    assert result["mean"] == 2.5
    assert result["min"] == 1 and result["max"] == 4
    assert result["p99"] == pytest.approx(3.97)
    assert distribution_ms(samples, method="lower")["p99"] == 3


def test_empty_single_and_invalid_statistics() -> None:
    assert percentile([], 0.5) is None
    with pytest.raises(ValueError, match="empty"):
        distribution_ms([])
    result = distribution_ms([7.5])
    assert all(result[key] == 7.5 for key in ("mean", "min", "max", "p50", "p90", "p99"))
    assert result["std"] == 0
    for invalid in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="finite"):
            distribution_ms([1, invalid])
        with pytest.raises(ValueError, match="finite"):
            percentile([invalid], 0.5)
    for quantile in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="quantile"):
            percentile([1], quantile)
    bad_method = cast(Literal["linear", "lower"], "unknown")
    with pytest.raises(ValueError, match="method"):
        distribution_ms([1], method=bad_method)


def test_triples_loader_preserves_order_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "triples.json"
    path.write_text(
        json.dumps({"triples": [{"a": "z", "b": "b", "c": "c"}, {"a": "a", "b": "b", "c": "c"}]})
    )
    assert load_triples(path) == [["a", "b", "c"], ["z", "b", "c"]]
    path.write_text('{"triples": [{"a": "a"}]}')
    with pytest.raises(ValueError, match="invalid triples row"):
        load_triples(path)


def test_qwen_performance_mode_releases_every_output() -> None:
    class Output:
        sequences = (None, None)

    previous: weakref.ReferenceType[Output] | None = None

    class LLM:
        def beam_search_v1(self, *args: Any, **kwargs: Any) -> list[Output]:
            nonlocal previous
            assert previous is None or previous() is None
            output = Output()
            previous = weakref.ref(output)
            return [output]

    result = benchmark_v1._run_once(
        LLM(),
        [{"input_ids": [1, 2]}] * 10_000,
        object(),
        beam_width=2,
        beam_steps=3,
        record_outputs=False,
    )
    assert result["outputs"] == []
    assert len(result["request_latencies_ms"]) == 10_000
    assert previous is not None and previous() is None
    assert result["outputs_recorded"] is False
    assert all(
        key in result["request_latency_ms"] for key in ("mean", "max", "min", "p50", "p90", "p99")
    )


def test_qwen_cli_preserves_accuracy_default() -> None:
    parser = benchmark_v1.build_parser()
    required = [
        "--model",
        "model",
        "--output-json",
        "result.json",
        "--workload-jsonl",
        "prompts.jsonl",
        "--token-id-start",
        "0",
        "--token-id-end",
        "100",
    ]
    assert parser.parse_args(required).record_outputs is True
    assert parser.parse_args([*required, "--record-outputs"]).record_outputs is True
    assert parser.parse_args([*required, "--no-record-outputs"]).record_outputs is False


def test_offline_example_keeps_only_final_output(tmp_path: Path) -> None:
    from examples.offline_inference.beam_search import offline_beam_search_v1 as example

    class Output:
        sequences = (None, None)

    previous: weakref.ReferenceType[Output] | None = None

    class LLM:
        def beam_search_v1(self, *args: Any, **kwargs: Any) -> list[Output]:
            nonlocal previous
            assert previous is None or previous() is None
            output = Output()
            previous = weakref.ref(output)
            return [output]

    args = Namespace(
        warmup_runs=0,
        repeat=100,
        model=tmp_path,
        catalog=tmp_path,
        prompt=tmp_path,
        constraint_backend="cuda",
        execution="graph",
        batch_size=1,
        beam_width=2,
        input_len=1024,
        generated_steps=3,
        length_penalty=1.0,
        max_model_len=8192,
        gpu_memory_utilization=0.5,
    )
    outputs, report = example.benchmark_beam_search(
        LLM(), args, [{"prompt_token_ids": [1, 2], "cache_salt": "test"}], object()
    )
    assert previous is not None and previous() is outputs[0]
    assert len(report["runs"]) == 100
    assert all(
        key in report["batch_latency_ms"] for key in ("mean", "max", "min", "p50", "p90", "p99")
    )


def test_onerec_output_modes_release_raw_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from benchmarks.open_one_rec import one_rec_v1_benchmark as benchmark

    class Output:
        def __init__(self) -> None:
            self.sequences = [
                SimpleNamespace(text="sid", tokens=list(range(7)), cum_logprob=-1.0)
                for _ in range(2)
            ]

    output_refs: list[weakref.ReferenceType[Output]] = []

    class LLM:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def reset_prefix_cache(self) -> bool:
            return True

        def beam_search_v1(self, *args: Any, **kwargs: Any) -> list[Output]:
            assert all(reference() is None for reference in output_refs)
            output = Output()
            output_refs.append(weakref.ref(output))
            return [output]

    engine_module = ModuleType("vllm_gr.entrypoints.gr")
    engine_module.__dict__["GRLLM"] = LLM
    params_module = ModuleType("vllm_gr.sampling_params")
    params_module.__dict__["BeamSearchParams"] = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, engine_module.__name__, engine_module)
    monkeypatch.setitem(sys.modules, params_module.__name__, params_module)
    monkeypatch.setattr(benchmark, "build_artifact", lambda *args: ("artifact", "tokenizer"))
    requests = [SimpleNamespace(prompt="prompt", expected_output_len=5)] * 3
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2])
    for mode in (None, True, False):
        args = Namespace(
            num_prompts=3,
            warmup_requests=1,
            n=2,
            constraint_triples=str(tmp_path / "catalog.json"),
            output_json=str(tmp_path / "report.json"),
            output_len=5,
            custom_input_len=None,
            max_model_len=8192,
            max_num_batched_tokens=8192,
            model=str(tmp_path),
            dataset_path="fake",
        )
        if mode is not None:
            args.record_outputs = mode
        benchmark.run_v1_benchmark(args, tokenizer, requests)
        report = json.loads(Path(args.output_json).read_text())
        assert len(report["samples"]) == 3
        assert all(("beams" in sample) == (mode is not False) for sample in report["samples"])
        assert all(reference() is None for reference in output_refs)


def test_tooling_cli_help_from_outside_checkout(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    for script in (
        "examples/offline_inference/beam_search/offline_beam_search_v1.py",
        "benchmarks/qwen/benchmark_v1.py",
        "tools/AB_Tests/run_offline_benchmark.py",
    ):
        result = subprocess.run(
            [sys.executable, str(root / script), "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "--model" in result.stdout
