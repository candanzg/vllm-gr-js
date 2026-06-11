# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class BeamGroupRuntimePolicy:
    max_query_len_tokens: int
    split_by_query_bucket: bool
    query_bucket_width: int
    min_shared_token_savings: int


def _normalize_device_type(device_type: Any) -> str:
    if device_type is None:
        return ""
    if hasattr(device_type, "type"):
        return str(device_type.type).lower()
    return str(device_type).lower()


def _normalize_cudagraph_mode(cudagraph_mode: Any) -> str:
    if cudagraph_mode is None:
        return ""
    return str(cudagraph_mode).upper()


def resolve_beam_group_runtime_policy(
    *,
    block_size: int,
    device_type: Any = None,
    cudagraph_mode: Any = None,
    min_shared_token_savings: int = 256,
) -> BeamGroupRuntimePolicy:
    normalized_device = _normalize_device_type(device_type)
    normalized_mode = _normalize_cudagraph_mode(cudagraph_mode)

    if normalized_device == "npu" and normalized_mode == "PIECEWISE":
        return BeamGroupRuntimePolicy(
            max_query_len_tokens=max(block_size * 2, 8192),
            split_by_query_bucket=True,
            query_bucket_width=max(block_size, 1),
            min_shared_token_savings=min_shared_token_savings,
        )

    if normalized_device == "npu":
        return BeamGroupRuntimePolicy(
            max_query_len_tokens=max(block_size * 2, 2048),
            split_by_query_bucket=True,
            query_bucket_width=max(block_size, 1),
            min_shared_token_savings=min_shared_token_savings,
        )

    return BeamGroupRuntimePolicy(
        max_query_len_tokens=block_size * 2,
        split_by_query_bucket=False,
        query_bucket_width=max(block_size, 1),
        min_shared_token_savings=min_shared_token_savings,
    )


def _has_enough_shared_savings(
    depth: int,
    group_size: int,
    *,
    block_size: int,
    min_shared_token_savings: int,
) -> bool:
    if depth <= 0 or group_size <= 1:
        return False
    return depth * block_size * (group_size - 1) >= min_shared_token_savings


def filter_shared_groups_by_runtime_policy(
    shared_groups: list[tuple[int, list[int]]],
    query_start_loc_cpu: torch.Tensor,
    *,
    block_size: int,
    device_type: Any = None,
    cudagraph_mode: Any = None,
    min_shared_token_savings: int = 256,
) -> list[tuple[int, list[int]]]:
    if not shared_groups:
        return []

    policy = resolve_beam_group_runtime_policy(
        block_size=block_size,
        device_type=device_type,
        cudagraph_mode=cudagraph_mode,
        min_shared_token_savings=min_shared_token_savings,
    )

    flat_req_ids = [req_id for _, req_ids in shared_groups for req_id in req_ids]
    if not flat_req_ids:
        return []

    flat_req_ids_tensor = torch.tensor(flat_req_ids, dtype=torch.long)
    starts = query_start_loc_cpu[flat_req_ids_tensor]
    ends = query_start_loc_cpu[flat_req_ids_tensor + 1]
    lengths = (ends - starts).tolist()

    filtered_groups: list[tuple[int, list[int]]] = []
    curr_idx = 0
    for depth, req_ids in shared_groups:
        group_len = len(req_ids)
        group_pairs = []
        for req_id, query_len in zip(req_ids, lengths[curr_idx : curr_idx + group_len]):
            if query_len > 0 and query_len <= policy.max_query_len_tokens:
                group_pairs.append((req_id, int(query_len)))
        curr_idx += group_len

        if not group_pairs:
            continue

        if not policy.split_by_query_bucket:
            filtered_ids = [req_id for req_id, _query_len in group_pairs]
            if _has_enough_shared_savings(
                depth,
                len(filtered_ids),
                block_size=block_size,
                min_shared_token_savings=policy.min_shared_token_savings,
            ):
                filtered_groups.append((depth, filtered_ids))
            continue

        bucketed_req_ids: OrderedDict[int, list[int]] = OrderedDict()
        bucket_width = max(policy.query_bucket_width, 1)
        for req_id, query_len in group_pairs:
            bucket_key = query_len // bucket_width
            bucketed_req_ids.setdefault(bucket_key, []).append(req_id)

        for filtered_ids in bucketed_req_ids.values():
            if _has_enough_shared_savings(
                depth,
                len(filtered_ids),
                block_size=block_size,
                min_shared_token_savings=policy.min_shared_token_savings,
            ):
                filtered_groups.append((depth, filtered_ids))

    return filtered_groups
