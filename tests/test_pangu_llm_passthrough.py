# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"

"""Unit tests for the Pangu LLM passthrough path.

After removing the GR adapter / recommend runtime, ``GRLLM`` is a thin
subclass of vLLM's :class:`~vllm.entrypoints.llm.LLM` that always delegates
construction to the standard vLLM initializer. These tests verify that
Pangu-style checkpoints go through that passthrough path.
"""

import gc
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _has_vllm_engine_runtime() -> bool:
    try:
        import vllm.engine.arg_utils  # noqa: F401
        import vllm.platforms  # noqa: F401
    except Exception:
        return False
    return True


_HAS_VLLM = _has_vllm_engine_runtime()
_SKIP = "vLLM engine/runtime submodules are unavailable in this environment"


@unittest.skipUnless(_HAS_VLLM, _SKIP)
class TestPanguGrllmPassthrough(unittest.TestCase):
    def test_grllm_init_delegates_to_vllm_llm(self):
        """GRLLM should delegate construction to vLLM's LLM.__init__."""
        from vllm_gr.entrypoints.gr import GRLLM

        with mock.patch(
            "vllm.entrypoints.llm.LLM.__init__",
            return_value=None,
        ) as mock_llm_init:
            GRLLM(model="/fake/pangu")

        mock_llm_init.assert_called_once()


class TestPanguPluginInitializeRuntime(unittest.TestCase):
    def test_initialize_runtime_does_not_raise(self):
        from vllm_gr import plugin

        with mock.patch("vllm_gr.patch.re_apply_patches", return_value=0):
            state = plugin.initialize_runtime()
        self.assertIn("patch_count", state)


@unittest.skipUnless(_HAS_VLLM, _SKIP)
class TestPanguGrllmShutdown(unittest.TestCase):
    def test_grllm_shutdown_releases_engine(self):
        from vllm_gr.entrypoints.gr import GRLLM

        llm = GRLLM.__new__(GRLLM)
        llm._shutdown_complete = False
        engine = mock.Mock()
        engine.shutdown = mock.Mock()
        llm.llm_engine = engine

        with mock.patch(
            "vllm_gr.utils.npu_adapter.release_accelerator_memory",
            create=True,
        ):
            with mock.patch.object(gc, "collect") as mock_gc:
                GRLLM.shutdown(llm)

        engine.shutdown.assert_called_once_with()
        mock_gc.assert_called_once_with()
        self.assertIsNone(llm.llm_engine)
        self.assertTrue(llm._shutdown_complete)


@unittest.skipUnless(_HAS_VLLM, _SKIP)
class TestGrllmOfflineBeamSearch(unittest.TestCase):
    def test_beam_search_uses_upstream_lora_sequence_helper(self):
        from vllm.sampling_params import BeamSearchParams

        from vllm_gr.entrypoints.gr import GRLLM

        llm = GRLLM.__new__(GRLLM)
        llm._preprocess_cmpl = mock.Mock(
            return_value=[{"type": "token", "prompt_token_ids": [1, 2]}]
        )
        llm._lora_request_to_seq = mock.Mock(side_effect=RuntimeError("sentinel helper call"))

        self.assertFalse(hasattr(llm, "_get_beam_search_lora_requests"))

        with self.assertRaisesRegex(RuntimeError, "sentinel helper call"):
            GRLLM.beam_search(
                llm,
                prompts=[{"prompt_token_ids": [1, 2]}],
                params=BeamSearchParams(beam_width=2, max_tokens=1),
            )

        llm._preprocess_cmpl.assert_called_once_with([{"prompt_token_ids": [1, 2]}])
        llm._lora_request_to_seq.assert_called_once_with(None, 1)

    def test_prepare_beam_step_requests_applies_eos_via_sampling_params(self):
        from vllm.sampling_params import SamplingParams

        from vllm_gr.entrypoints.gr import _prepare_beam_step_requests

        llm_engine = SimpleNamespace(
            output_processor=mock.Mock(),
            engine_core=mock.Mock(),
        )
        instance = SimpleNamespace(beams=[SimpleNamespace(tokens=[10, 20, 30], lora_request=None)])

        engine_core_requests = _prepare_beam_step_requests(
            llm_engine=llm_engine,
            instances_batch=[instance],
            token=1,
            prompt_start=0,
            tokenizer=SimpleNamespace(eos_token_id=7),
            beam_search_params=SamplingParams(
                max_tokens=1,
                temperature=0.0,
                detokenize=False,
            ),
            sid_end_token_id=99,
            instance_prefill_id={0: "session-0"},
            instance_prev_beam_internal_ids={0: ["parent-0"]},
            instance_fork_info={0: [(0, 30)]},
            instance_prefix_len={0: 2},
            instance_completed_sessions=set(),
        )

        self.assertEqual(engine_core_requests, [])
        llm_engine.output_processor.add_request.assert_called_once()
        ec_req = llm_engine.output_processor.add_request.call_args[0][0]
        self.assertEqual(ec_req.request_id, "session-0")
        self.assertEqual(ec_req.prompt_token_ids, [10, 20, 30])
        self.assertEqual(ec_req.sampling_params.eos_token_id, 99)

        llm_engine.engine_core.beam_request_step_update.assert_called_once()
        update = llm_engine.engine_core.beam_request_step_update.call_args[0][0]
        self.assertEqual(update.eos_token_id, 99)
        self.assertEqual(update.beam_tokens, [[30]])


if __name__ == "__main__":
    unittest.main()
