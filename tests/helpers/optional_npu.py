# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Module-level gating for optional NPU and vendor integration test modules."""

from __future__ import annotations

import os

import pytest

XATTENTION_OUT_REQUIRED_ENV = "VLLM_GR_TEST_XATTENTION_OUT_REQUIRED"


def skip_or_fail_unavailable(reason: str, *, required_env: str) -> None:
    """Skip an optional integration module unless its required mode is enabled."""
    if os.environ.get(required_env) == "1":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason, allow_module_level=True)
