# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check V1 caller batching, suffix decoding and E2E measurement boundaries."""

from types import SimpleNamespace
from typing import Any, Self

import pytest

from tools.AB_Tests.run_offline_benchmark import measure_e2e


class _Engine:
    def __init__(self, wrong_items: bool = False, wrong_beams: bool = False) -> None:
        self.batches: list[list[str]] = []
        self.wrong_items = wrong_items
        self.wrong_beams = wrong_beams

    def get_tokenizer(self) -> Any:
        return SimpleNamespace(decode=lambda tokens, **kw: ":".join(map(str, tokens)))

    def beam_search_v1(self, prompts: list[dict[str, str]], params: Any) -> list[Any]:
        self.batches.append([p["prompt"] for p in prompts])
        return [
            SimpleNamespace(
                sequences=[
                    SimpleNamespace(tokens=[100, 101, 10, 20, 30, 40, 50], cum_logprob=-float(i))
                    for i in range(params.beam_width - int(self.wrong_beams))
                ]
            )
            for _ in prompts[: len(prompts) - int(self.wrong_items)]
        ]


def test_accuracy_preserves_order_and_handles_partial_batches() -> None:
    from benchmarks.open_one_rec.one_rec_acc_test import run_offline_v1_inference

    prompts = [f"prompt {i} with normalized whitespace" for i in range(7)]
    for batch_size, expected in ((1, [1] * 7), (2, [2, 2, 2, 1]), (4, [4, 2, 1])):
        engine = _Engine()
        outputs = run_offline_v1_inference(
            engine, prompts, SimpleNamespace(beam_width=2, max_tokens=5), batch_size
        )
        assert [len(batch) for batch in engine.batches] == expected
        assert [output.prompt for output in outputs] == prompts
        assert all(len(output.outputs) == 2 for output in outputs)
        assert all(output.outputs[0].text == "10:20:30:40:50" for output in outputs)
        assert all(output.outputs[1].logprobs == -1.0 for output in outputs)
    empty = _Engine()
    assert run_offline_v1_inference(empty, [], object(), 4) == []
    assert empty.batches == []


def test_accuracy_rejects_mismatched_output_counts() -> None:
    from benchmarks.open_one_rec.one_rec_acc_test import run_offline_v1_inference

    for engine in (_Engine(wrong_items=True), _Engine(wrong_beams=True)):
        with pytest.raises(RuntimeError, match="unexpected number"):
            run_offline_v1_inference(
                engine, ["prompt"], SimpleNamespace(beam_width=2, max_tokens=5), 1
            )


def test_e2e_includes_lazy_text_and_uses_only_v1(monkeypatch: Any) -> None:
    events: list[str] = []

    class Beam:
        @property
        def text(self) -> str:
            events.append("decode")
            return "result"

    def run(prompts: Any, params: Any, **kwargs: Any) -> list[Any]:
        events.append("v1")
        return [SimpleNamespace(sequences=[Beam(), Beam()])]

    def clock() -> int:
        events.append("clock")
        return len(events) * 1_000_000

    monkeypatch.setattr("tools.AB_Tests.run_offline_benchmark.time.perf_counter_ns", clock)
    result = measure_e2e(SimpleNamespace(beam_search_v1=run), [1, 2], SimpleNamespace(max_tokens=5))
    assert events == ["clock", "v1", "decode", "decode", "clock"]
    assert result == {"e2e_ms": 4.0, "returned_beams": 2, "aggregate_output_tokens": 10}


def test_benchmark_keeps_artifact_until_engine_shutdown(monkeypatch: Any) -> None:
    from argparse import Namespace
    from pathlib import Path

    import examples.offline_inference.beam_search.offline_beam_search_v1 as example
    from tools.AB_Tests.run_offline_benchmark import create_v1_llm
    from vllm_gr.entrypoints import gr

    artifacts: list[Path] = []
    events: list[str] = []

    def build(model: Path, catalog: Path, path: Path) -> tuple[str, str]:
        path.write_bytes(b"artifact")
        artifacts.append(path)
        return "artifact-digest", "tokenizer-digest"

    class Engine:
        def __init__(self, **kwargs: Any) -> None:
            config = kwargs["vllm_gr_config"]
            assert config["beam"]["execution_mode"] == "v1"
            assert config["beam"]["max_width"] == 128
            assert config["beam"]["max_decode_steps"] == 3
            assert config["constraint_table"]["artifact_digest"] == "artifact-digest"
            assert kwargs["async_scheduling"] is True
            assert kwargs["enable_chunked_prefill"] is False
            assert kwargs["enable_prefix_caching"] is True
            assert artifacts[-1].exists()

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exc: object) -> None:
            assert artifacts[-1].exists()
            events.append("shutdown")

    monkeypatch.setattr(example, "build_artifact", build)
    monkeypatch.setattr(gr, "GRLLM", Engine)
    args = Namespace(model="model", catalog=Path("catalog"), beam_width=128, input_length=1024)
    with pytest.raises(RuntimeError, match="request failed"), create_v1_llm(args, 5):
        raise RuntimeError("request failed")
    assert events == ["shutdown"]
    assert not artifacts[0].exists()
