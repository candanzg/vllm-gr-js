# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Core tests for GR input/output processor behavior."""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import importlib
import os
import sys
import types
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _has_vllm_processor_runtime() -> bool:
    required_modules = [
        "vllm.v1.engine",
        "vllm.v1.engine.input_processor",
        "vllm.v1.engine.output_processor",
        "vllm.outputs",
    ]
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception:
            return False
    return True


_HAS_VLLM_RUNTIME = _has_vllm_processor_runtime()
_SKIP_REASON = "vLLM processor/runtime submodules are unavailable in this environment"


def _make_adapter(
    *,
    task_names=None,
    max_seq_len=64,
    max_candidates=50,
) -> None:
    """Build a FuXiAlphaModelAdapter with the given runtime config."""

    from vllm_gr.adapters.fuxi.adapter import FuXiAlphaModelAdapter

    if task_names is None:
        task_names = ["score"]

    hf_config = types.SimpleNamespace(
        max_seq_len=max_seq_len,
        max_num_candidates_inference=max_candidates,
        item_embedding_dim=50,
        item_id_feature_name="item_id",
        rating_feature_name="rating",
        timestamp_feature_name="timestamp",
        task_names=task_names,
        use_rating_embedding=False,
        use_timestamp_payload=False,
        enable_kv_cache=True,
    )
    vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(hf_config=hf_config),
    )
    gr_config = types.SimpleNamespace(
        max_seq_len=max_seq_len,
        max_candidates=max_candidates,
    )
    return FuXiAlphaModelAdapter(vllm_config, gr_config)


def _decode_request_payload(request) -> None:
    from vllm_gr.adapters.fuxi.payload import FuXiFeaturesPayload
    from vllm_gr.engine.gr_features_codec import decode_gr_features_payload

    return decode_gr_features_payload(request.gr_features, FuXiFeaturesPayload)


# ── GRRequestState / output construction ─────────────────────────


class TestGRRequestState(unittest.TestCase):
    @staticmethod
    def _make_state(user_id="user_123", candidates=None, primary_id_key="item_id") -> None:
        from vllm_gr.engine.gr_output_processor import GRRequestState

        if candidates is None:
            candidates = [
                {"item_id": 10},
                {"item_id": 20},
                {"item_id": 30},
                {"item_id": 40},
                {"item_id": 50},
            ]

        adapter = _make_adapter()
        return GRRequestState(
            request_id="test-001",
            external_req_id="test-001",
            parent_req=None,
            request_index=0,
            lora_request=None,
            output_kind=None,
            prompt=None,
            prompt_token_ids=[1, 2, 3],
            prompt_embeds=None,
            logprobs_processor=None,
            detokenizer=None,
            max_tokens_param=None,
            arrival_time=0.0,
            queue=None,
            log_stats=False,
            stream_interval=1,
            is_gr_request=True,
            gr_request_meta={
                "user_id": user_id,
                "original_candidates": candidates,
                "primary_id_key": primary_id_key,
                "task_names": [],
                "primary_task": "",
                "_adapter": adapter,
            },
        )

    def test_make_gr_output_basic(self) -> None:
        state = self._make_state()
        scores = torch.tensor([[0.1, 0.5, 0.3, 0.8, 0.2]])

        from vllm_gr.outputs import GRRequestOutput

        output = state._make_gr_output(scores, finish_reason="stop")

        self.assertIsInstance(output, GRRequestOutput)
        self.assertEqual(output.user_id, "user_123")
        self.assertEqual(len(output.recommendations), 5)
        self.assertEqual(output.recommendations[0].item_id, 10)
        self.assertAlmostEqual(output.recommendations[0].score, 0.1)

    def test_make_gr_output_preserves_all_candidate_scores(self) -> None:
        candidates = [{"item_id": i} for i in range(10)]
        state = self._make_state(candidates=candidates)
        scores = torch.tensor([[0.1, 0.9, 0.3, 0.8, 0.5, 0.2, 0.7, 0.4, 0.6, 0.15]])

        output = state._make_gr_output(scores, finish_reason="stop")

        self.assertEqual(len(output.recommendations), 10)
        self.assertEqual(output.recommendations[0].item_id, 0)
        self.assertEqual(output.recommendations[1].item_id, 1)
        self.assertAlmostEqual(output.recommendations[1].score, 0.9)

    def test_make_gr_output_rejects_candidate_score_mismatch(self) -> None:
        state = self._make_state(candidates=[{"item_id": 10}, {"item_id": 20}])
        scores = torch.tensor([[0.1]])

        with self.assertRaisesRegex(ValueError, "candidate/score count mismatch"):
            state._make_gr_output(scores, finish_reason="stop")

    def test_make_gr_output_rejects_non_2d_scores(self) -> None:
        state = self._make_state()

        with self.assertRaisesRegex(ValueError, "must be 2D"):
            state._make_gr_output(torch.tensor([0.1, 0.5]), finish_reason="stop")

    def test_make_request_output_keeps_empty_candidate_requests_in_gr_path(self) -> None:
        state = self._make_state(candidates=[])

        output = state.make_request_output(
            new_token_ids=[],
            pooling_output=torch.tensor([0.42]),
            finish_reason="stop",
            stop_reason=None,
        )

        from vllm_gr.outputs import GRRequestOutput

        self.assertIsInstance(output, GRRequestOutput)
        self.assertEqual(output.user_id, "user_123")
        self.assertEqual(output.recommendations, [])

    def test_make_gr_output_uses_explicit_primary_id_key(self) -> None:
        state = self._make_state(
            candidates=[
                {"rank_feature": 999, "item_id": 10},
                {"rank_feature": 555, "item_id": 20},
            ],
            primary_id_key="item_id",
        )
        scores = torch.tensor([[0.1, 0.2]])

        output = state._make_gr_output(scores, finish_reason="stop")

        self.assertEqual(len(output.recommendations), 2)
        self.assertEqual(output.recommendations[0].item_id, 10)


class TestMultiTaskOutput(unittest.TestCase):
    def _make_state_with_tasks(self, task_names, primary_task="") -> None:
        from vllm_gr.engine.gr_output_processor import GRRequestState

        candidates = [
            {"item_id": 10},
            {"item_id": 20},
            {"item_id": 30},
        ]
        adapter = _make_adapter(task_names=task_names)
        return GRRequestState(
            request_id="test-mt",
            external_req_id="test-mt",
            parent_req=None,
            request_index=0,
            lora_request=None,
            output_kind=None,
            prompt=None,
            prompt_token_ids=[1, 2, 3],
            prompt_embeds=None,
            logprobs_processor=None,
            detokenizer=None,
            max_tokens_param=None,
            arrival_time=0.0,
            queue=None,
            log_stats=False,
            stream_interval=1,
            is_gr_request=True,
            gr_request_meta={
                "user_id": "user_mt",
                "original_candidates": candidates,
                "primary_id_key": "item_id",
                "task_names": task_names,
                "primary_task": primary_task,
                "_adapter": adapter,
            },
        )

    def test_primary_task_selection(self) -> None:
        state = self._make_state_with_tasks(task_names=["click", "like"], primary_task="like")
        scores = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        output = state._make_gr_output(scores, finish_reason="stop")

        # primary_task="like" → use row 1
        self.assertEqual(len(output.recommendations), 3)
        self.assertAlmostEqual(output.recommendations[0].score, 0.4)

    def test_per_item_task_scores(self) -> None:
        state = self._make_state_with_tasks(task_names=["click", "like"])
        scores = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        output = state._make_gr_output(scores, finish_reason="stop")

        # default primary_task="" → use row 0
        self.assertAlmostEqual(output.recommendations[0].score, 0.1)


# ── FuXiFeaturesPayload ──────────────────────────────────────────


@unittest.skipUnless(_HAS_VLLM_RUNTIME, _SKIP_REASON)
class TestFuXiFeaturesPayload(unittest.TestCase):
    def test_creation_and_serialization(self) -> None:
        """Payload round-trips through msgspec."""
        import msgspec

        from vllm_gr.adapters.fuxi.payload import FuXiFeaturesPayload

        payload = FuXiFeaturesPayload(
            past_ids=[1, 2, 3],
            past_ratings=[0, 0, 0],
            past_timestamps=[1000, 2000, 3000],
            history_len=3,
            candidate_ids=[10, 20],
            num_candidates=2,
            user_id="user_123",
        )

        self.assertEqual(payload.user_id, "user_123")
        self.assertEqual(payload.history_len, 3)

        data = msgspec.msgpack.encode(payload)
        decoded = msgspec.msgpack.decode(data, type=FuXiFeaturesPayload)
        self.assertEqual(decoded.past_ids, [1, 2, 3])
        self.assertEqual(decoded.user_id, "user_123")


if __name__ == "__main__":
    unittest.main()
