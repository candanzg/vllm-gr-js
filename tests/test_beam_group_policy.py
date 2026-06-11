# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "vllm_gr" / "v1" / "attention" / "backends" / "beam_group_policy.py"
SPEC = importlib.util.spec_from_file_location("beam_group_policy_test_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

filter_shared_groups_by_runtime_policy = MODULE.filter_shared_groups_by_runtime_policy
resolve_beam_group_runtime_policy = MODULE.resolve_beam_group_runtime_policy


class TestBeamGroupPolicy(unittest.TestCase):
    def test_gpu_policy_keeps_short_queries_only(self) -> None:
        query_start_loc_cpu = torch.tensor([0, 8, 20, 60], dtype=torch.long)
        shared_groups = [(16, [0, 1, 2])]

        filtered = filter_shared_groups_by_runtime_policy(
            shared_groups,
            query_start_loc_cpu,
            block_size=16,
            device_type="cuda",
        )

        self.assertEqual(filtered, [(16, [0, 1])])

    def test_npu_piecewise_policy_buckets_long_queries(self) -> None:
        query_start_loc_cpu = torch.tensor([0, 5133, 10266, 15415, 20564], dtype=torch.long)
        shared_groups = [(16, [0, 1, 2, 3])]

        filtered = filter_shared_groups_by_runtime_policy(
            shared_groups,
            query_start_loc_cpu,
            block_size=16,
            device_type="npu",
            cudagraph_mode="PIECEWISE",
        )

        self.assertEqual(filtered, [(16, [0, 1]), (16, [2, 3])])

    def test_npu_piecewise_policy_relaxes_query_len_cap(self) -> None:
        policy = resolve_beam_group_runtime_policy(
            block_size=16,
            device_type="npu",
            cudagraph_mode="PIECEWISE",
        )

        self.assertEqual(policy.max_query_len_tokens, 8192)
        self.assertTrue(policy.split_by_query_bucket)
        self.assertEqual(policy.query_bucket_width, 16)


if __name__ == "__main__":
    unittest.main()
