# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _purge_modules(*prefixes: str) -> dict[str, types.ModuleType]:
    saved: dict[str, types.ModuleType] = {}
    for name, module in list(sys.modules.items()):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            saved[name] = module
            sys.modules.pop(name, None)
    return saved


def _restore_modules(saved: dict[str, types.ModuleType]) -> None:
    for name, module in saved.items():
        sys.modules[name] = module


def _install_stub_package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


class TestNPUProfilePatch(unittest.TestCase):
    def _install_fake_runtime(self, *, runner_module_name: str | None = None):
        saved = _purge_modules(
            "vllm",
            "vllm_ascend",
            "vllm_gr.worker.npu_profile_patch",
        )

        fake_vllm = types.ModuleType("vllm")
        fake_logger = types.ModuleType("vllm.logger")
        fake_logger.init_logger = lambda name: types.SimpleNamespace(
            debug=lambda *_args, **_kwargs: None,
            info=lambda *_args, **_kwargs: None,
            warning=lambda *_args, **_kwargs: None,
            error=lambda *_args, **_kwargs: None,
        )

        sys.modules.update(
            {
                "vllm": fake_vllm,
                "vllm.logger": fake_logger,
            }
        )

        _install_stub_package("vllm_ascend")
        _install_stub_package("vllm_ascend.worker")
        _install_stub_package("vllm.v1")
        _install_stub_package("vllm.v1.worker")
        model_runner_module = types.ModuleType("vllm_ascend.worker.model_runner_v1")
        gpu_model_runner_module = types.ModuleType("vllm.v1.worker.gpu_model_runner")

        class NPUModelRunner:
            def __init__(self) -> None:
                self.input_batch = types.SimpleNamespace(
                    req_ids=["r1", "r2", "r3"],
                    num_reqs=3,
                )
                self.uniform_decode_query_len = 1
                self.use_aclgraph = True
                self.compilation_config = types.SimpleNamespace(cudagraph_mode="PIECEWISE")

            def _prepare_inputs(self, scheduler_output):
                return (None, None, 12, None, None, None, 1)

        if runner_module_name is not None:
            NPUModelRunner.__module__ = runner_module_name

        model_runner_module.NPUModelRunner = NPUModelRunner
        sys.modules[model_runner_module.__name__] = model_runner_module
        sys.modules[gpu_model_runner_module.__name__] = gpu_model_runner_module
        return saved, NPUModelRunner

    def _install_fake_gpu_runner_runtime(self):
        saved = _purge_modules(
            "vllm",
            "vllm_ascend",
            "vllm_gr.worker.npu_profile_patch",
        )

        fake_vllm = types.ModuleType("vllm")
        fake_logger = types.ModuleType("vllm.logger")
        fake_logger.init_logger = lambda name: types.SimpleNamespace(
            debug=lambda *_args, **_kwargs: None,
            info=lambda *_args, **_kwargs: None,
            warning=lambda *_args, **_kwargs: None,
            error=lambda *_args, **_kwargs: None,
        )

        sys.modules.update(
            {
                "vllm": fake_vllm,
                "vllm.logger": fake_logger,
            }
        )

        _install_stub_package("vllm.v1")
        _install_stub_package("vllm.v1.worker")
        gpu_model_runner_module = types.ModuleType("vllm.v1.worker.gpu_model_runner")

        class GPUModelRunner:
            def __init__(self) -> None:
                self.input_batch = types.SimpleNamespace(req_ids=["r1"], num_reqs=1)
                self.uniform_decode_query_len = 1
                self.use_aclgraph = True
                self.compilation_config = types.SimpleNamespace(cudagraph_mode="PIECEWISE")

            def _prepare_inputs(self, scheduler_output):
                return (None, None, 4, None, None, None, 1)

        gpu_model_runner_module.GPUModelRunner = GPUModelRunner
        sys.modules[gpu_model_runner_module.__name__] = gpu_model_runner_module
        return saved, GPUModelRunner

    def test_patch_installs_and_logs_when_env_enabled(self) -> None:
        saved, NPUModelRunner = self._install_fake_runtime()
        try:
            module = importlib.import_module("vllm_gr.worker.npu_profile_patch")

            self.assertTrue(module.patch_npu_model_runner_profile())
            self.assertFalse(module.patch_npu_model_runner_profile())

            scheduler_output = types.SimpleNamespace(
                total_num_scheduled_tokens=3,
                beam_prefix_groups=[(4, ["r1", "r2"]), (2, ["r3"])],
            )
            runner = NPUModelRunner()

            with (
                mock.patch.dict("os.environ", {"VLLM_GR_NPU_PROFILE": "1"}),
                mock.patch.object(module.logger, "info") as mock_info,
            ):
                result = runner._prepare_inputs(scheduler_output)

            self.assertEqual(result[2], 12)
            mock_info.assert_called_once()
            message, *args = mock_info.call_args[0]
            rendered = message % tuple(args)
            self.assertIn("[GR-NPU-PROFILE]", rendered)
            self.assertIn("runner=NPUModelRunner", rendered)
            self.assertIn("scheduled_tokens=3", rendered)
            self.assertIn("num_input_tokens=12", rendered)
            self.assertIn("padding_ratio=4.0000", rendered)
            self.assertIn("beam_group_count=2", rendered)
            self.assertIn("beam_group_reqs=3", rendered)
        finally:
            _purge_modules("vllm", "vllm_ascend", "vllm_gr.worker.npu_profile_patch")
            _restore_modules(saved)

    def test_patch_skips_log_when_env_disabled(self) -> None:
        saved, NPUModelRunner = self._install_fake_runtime()
        try:
            module = importlib.import_module("vllm_gr.worker.npu_profile_patch")
            self.assertTrue(module.patch_npu_model_runner_profile())

            scheduler_output = types.SimpleNamespace(total_num_scheduled_tokens=1)
            runner = NPUModelRunner()

            with mock.patch.object(module.logger, "info") as mock_info:
                runner._prepare_inputs(scheduler_output)

            mock_info.assert_not_called()
        finally:
            _purge_modules("vllm", "vllm_ascend", "vllm_gr.worker.npu_profile_patch")
            _restore_modules(saved)

    def test_patch_logs_for_gpu_model_runner_base_target(self) -> None:
        saved, GPUModelRunner = self._install_fake_gpu_runner_runtime()
        try:
            module = importlib.import_module("vllm_gr.worker.npu_profile_patch")
            self.assertTrue(module.patch_npu_model_runner_profile())

            scheduler_output = types.SimpleNamespace(total_num_scheduled_tokens=1)
            runner = GPUModelRunner()

            with (
                mock.patch.dict("os.environ", {"VLLM_GR_NPU_PROFILE": "1"}),
                mock.patch.object(module.logger, "info") as mock_info,
            ):
                result = runner._prepare_inputs(scheduler_output)

            self.assertEqual(result[2], 4)
            mock_info.assert_called_once()
            message, *args = mock_info.call_args[0]
            rendered = message % tuple(args)
            self.assertIn("[GR-NPU-PROFILE]", rendered)
            self.assertIn("runner=GPUModelRunner", rendered)
            self.assertIn("scheduled_tokens=1", rendered)
        finally:
            _purge_modules("vllm", "vllm_ascend", "vllm_gr.worker.npu_profile_patch")
            _restore_modules(saved)

    def test_patch_skips_gr_npu_model_runner_duplicate_log(self) -> None:
        saved, NPUModelRunner = self._install_fake_runtime(
            runner_module_name="vllm_gr.worker.gr_npu_model_runner"
        )
        try:
            module = importlib.import_module("vllm_gr.worker.npu_profile_patch")
            self.assertTrue(module.patch_npu_model_runner_profile())

            scheduler_output = types.SimpleNamespace(total_num_scheduled_tokens=1)
            runner = NPUModelRunner()

            with (
                mock.patch.dict("os.environ", {"VLLM_GR_NPU_PROFILE": "1"}),
                mock.patch.object(module.logger, "info") as mock_info,
            ):
                runner._prepare_inputs(scheduler_output)

            mock_info.assert_not_called()
        finally:
            _purge_modules("vllm", "vllm_ascend", "vllm_gr.worker.npu_profile_patch")
            _restore_modules(saved)

    def test_patch_returns_false_when_vllm_ascend_missing(self) -> None:
        saved = _purge_modules("vllm_ascend", "vllm_gr.worker.npu_profile_patch")
        try:
            fake_vllm = types.ModuleType("vllm")
            fake_logger = types.ModuleType("vllm.logger")
            fake_logger.init_logger = lambda name: types.SimpleNamespace(
                debug=lambda *_args, **_kwargs: None,
                info=lambda *_args, **_kwargs: None,
            )
            sys.modules["vllm"] = fake_vllm
            sys.modules["vllm.logger"] = fake_logger

            module = importlib.import_module("vllm_gr.worker.npu_profile_patch")
            self.assertFalse(module.patch_npu_model_runner_profile())
        finally:
            _purge_modules("vllm", "vllm_ascend", "vllm_gr.worker.npu_profile_patch")
            _restore_modules(saved)


if __name__ == "__main__":
    unittest.main()
