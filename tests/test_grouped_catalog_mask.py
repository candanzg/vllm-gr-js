# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm_gr.ops.sliced_logits import set_vocab_slice_config
from vllm_gr.v1.worker.model_runner_common import InputBatchProxy


def test_input_batch_proxy_builds_distinct_mask_per_beam() -> None:
    set_vocab_slice_config(None, None)
    base_mask = torch.zeros((1, 5), dtype=torch.bool)
    input_batch = SimpleNamespace(
        req_ids=["group"],
        vocab_size=5,
        device=torch.device("cpu"),
        sampling_metadata=SimpleNamespace(
            temperature=torch.tensor([0.0]),
            allowed_token_ids_mask=base_mask,
        ),
    )
    proxy = InputBatchProxy(
        input_batch,
        req_map=["group", "group"],
        idx_map=[0, 0],
        logits_map=[0, 0],
        allowed_token_ids=[[1, 3], [2]],
    )

    metadata = proxy.sampling_metadata

    assert metadata.allowed_token_ids_mask.tolist() == [
        [True, False, True, False, True],
        [True, True, False, True, True],
    ]
    assert metadata.temperature.tolist() == [0.0, 0.0]


def test_input_batch_proxy_maps_full_ids_to_sliced_logits() -> None:
    set_vocab_slice_config(10, 5)
    try:
        input_batch = SimpleNamespace(
            req_ids=["group"],
            vocab_size=20,
            device=torch.device("cpu"),
            sampling_metadata=SimpleNamespace(
                temperature=torch.tensor([0.0]),
                allowed_token_ids_mask=torch.zeros((1, 20), dtype=torch.bool),
            ),
        )
        proxy = InputBatchProxy(
            input_batch,
            req_map=["group", "group"],
            idx_map=[0, 0],
            logits_map=[0, 0],
            allowed_token_ids=[[11, 14], [12]],
        )

        assert proxy.sampling_metadata.allowed_token_ids_mask.tolist() == [
            [True, False, True, True, False],
            [True, True, False, True, True],
        ]
    finally:
        set_vocab_slice_config(None, None)
