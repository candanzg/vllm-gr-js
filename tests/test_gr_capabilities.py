# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for capability protocol isinstance checks."""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import unittest
from types import SimpleNamespace

try:
    import vllm  # noqa: F401

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

from vllm_gr.adapters.capabilities import (
    GREmbeddingModel,
    GRPoolingModel,
    GRPrefixCacheSupport,
    GRScoringModel,
)


@unittest.skipUnless(_VLLM_AVAILABLE, "vLLM runtime not available")
class TestGRCapabilities(unittest.TestCase):
    """Verify that FuXiAlphaModelAdapter satisfies expected capability protocols."""

    def _make_fuxi_adapter(self) -> None:
        from vllm_gr.adapters.fuxi.adapter import FuXiAlphaModelAdapter

        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    max_seq_len=64,
                    max_num_candidates_inference=50,
                    item_embedding_dim=50,
                    item_id_feature_name="item_id",
                    rating_feature_name="rating",
                    timestamp_feature_name="timestamp",
                    task_names=["score"],
                    use_rating_embedding=False,
                    use_timestamp_payload=False,
                    enable_kv_cache=True,
                ),
            )
        )
        gr_config = SimpleNamespace(max_seq_len=64, max_candidates=50)
        return FuXiAlphaModelAdapter(vllm_config, gr_config)

    def test_fuxi_is_pooling_model(self) -> None:
        adapter = self._make_fuxi_adapter()
        self.assertIsInstance(adapter, GRPoolingModel)

    def test_fuxi_is_scoring_model(self) -> None:
        adapter = self._make_fuxi_adapter()
        self.assertIsInstance(adapter, GRScoringModel)

    def test_fuxi_is_not_embedding_model(self) -> None:
        """FuXi is a scoring model, not an embedding model."""
        adapter = self._make_fuxi_adapter()
        self.assertNotIsInstance(adapter, GREmbeddingModel)

    def test_scoring_and_embedding_are_distinguishable(self) -> None:
        """Scoring and embedding protocols must produce different isinstance results."""

        class _FakeScoringAdapter:
            is_scoring_model = True

            def build_model_kwargs(self, batch_features, device) -> None:
                return {}

        class _FakeEmbeddingAdapter:
            is_embedding_model = True

            def build_model_kwargs(self, batch_features, device) -> None:
                return {}

        self.assertIsInstance(_FakeScoringAdapter(), GRScoringModel)
        self.assertNotIsInstance(_FakeScoringAdapter(), GREmbeddingModel)
        self.assertIsInstance(_FakeEmbeddingAdapter(), GREmbeddingModel)
        self.assertNotIsInstance(_FakeEmbeddingAdapter(), GRScoringModel)

    def test_fuxi_supports_prefix_cache(self) -> None:
        adapter = self._make_fuxi_adapter()
        self.assertIsInstance(adapter, GRPrefixCacheSupport)

    def test_plain_adapter_is_not_pooling_model(self) -> None:
        """A base adapter that doesn't implement the protocol is not detected."""

        class _Plain:
            pass

        self.assertNotIsInstance(_Plain(), GRPoolingModel)

    def test_adapter_without_capability_marker_is_neither_scoring_nor_embedding(self) -> None:
        """A class with build_model_kwargs but no marker matches neither."""

        class _NoMarker:
            def build_model_kwargs(self, batch_features, device) -> None:
                return {}

        self.assertNotIsInstance(_NoMarker(), GRScoringModel)
        self.assertNotIsInstance(_NoMarker(), GREmbeddingModel)

    def test_embedding_model_protocol_exists(self) -> None:
        """GREmbeddingModel is defined (for future use)."""
        self.assertTrue(callable(GREmbeddingModel))


if __name__ == "__main__":
    unittest.main()
