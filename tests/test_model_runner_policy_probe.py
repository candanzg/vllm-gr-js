# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Collect the manual probes' startup contract in the normal regression suite."""

from typing import Any

import pytest

from tests.st.model_runner_policy_isolation import startup_expectations


@pytest.mark.parametrize(  # type: ignore[misc]
    "runner,sizes,max_seqs,enabled,eager,startup,expected",
    [
        pytest.param("v2", None, 8, True, False, "4", [1, 2, 4], id="default"),
        pytest.param("v2", None, 1, True, False, "4", [1], id="single-capacity"),
        pytest.param("v2", None, 3, True, False, "4", [1, 2], id="bounded-capacity"),
        pytest.param("v2", [7, 1, 3], 7, True, False, "4", [1, 3, 7], id="custom"),
        pytest.param("v2", None, 8, False, False, "4", [], id="graph-disabled"),
        pytest.param("v2", None, 8, True, True, "4", [], id="eager"),
        pytest.param("v1", [1, 3, 7], 8, True, False, None, [1], id="v1-default"),
        pytest.param("v1", [1, 3, 7], 8, True, False, "4", [4], id="v1-startup"),
        pytest.param("v1", None, 8, False, False, "4", [], id="v1-graph-disabled"),
        pytest.param("v1", None, 8, True, True, "4", [], id="v1-eager"),
    ],
)
def test_probe_separates_capture_policy_from_backend_capacity(
    monkeypatch: pytest.MonkeyPatch,
    runner: str,
    sizes: list[int] | None,
    max_seqs: int,
    enabled: bool,
    eager: bool,
    startup: str | None,
    expected: list[int],
) -> None:
    monkeypatch.delenv("VLLM_GR_V1_STARTUP_CAPTURE_BATCH_SIZE", raising=False)
    if startup is not None:
        monkeypatch.setenv("VLLM_GR_V1_STARTUP_CAPTURE_BATCH_SIZE", startup)
    monkeypatch.setenv("VLLM_GR_PREFILL_GRAPH_CAPTURE_NUM_REQS", "1,6,8")
    beam: dict[str, Any] = {"max_width": 128, "graph_enabled": enabled}
    if sizes is not None:
        beam["online_batch_sizes"] = sizes
    assert startup_expectations(
        runner=runner,
        config={"schema_version": 1, "beam": beam, "constraint_table": {"max_top_k": 128}},
        max_num_seqs=max_seqs,
        enforce_eager=eager,
    ) == (1024 if runner == "v2" else 128, expected)


pytestmark = pytest.mark.cpu_test
