# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Instance-local specialization of native single-group prefix-cache lookup."""

from functools import wraps
from typing import Any

from vllm.v1.core.block_pool import BlockHashToBlockMap, BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock


def install_gr_cache_lookup(coordinator: Any) -> None:
    """Collapse per-block single-group dispatch without changing cache ownership."""
    from itertools import islice

    from vllm.v1.core.kv_cache_coordinator import UnitaryKVCacheCoordinator
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    if type(coordinator) is not UnitaryKVCacheCoordinator:
        return
    original = coordinator.find_longest_cache_hit
    if getattr(original, "_gr_bulk_lookup", False):
        return
    if getattr(original, "__func__", None) is not UnitaryKVCacheCoordinator.find_longest_cache_hit:
        return
    native_lookup = BlockPool.get_cached_block
    native_get = BlockHashToBlockMap.get_one_block
    native_find = FullAttentionManager.find_longest_cache_hit.__func__

    @wraps(original)
    def find_longest_cache_hit(block_hashes: list[bytes], max_cache_hit_length: int) -> Any:
        pool = coordinator.block_pool
        index = pool.cached_block_hash_to_block
        size = coordinator.block_size
        if (
            coordinator.dcp_world_size != 1
            or coordinator.pcp_world_size != 1
            or coordinator.eagle_group_ids
            or type(coordinator.kv_cache_spec) is not FullAttentionSpec
            or coordinator.kv_cache_spec.block_size != size
            or len(coordinator.single_type_managers) != 1
            or type(coordinator.single_type_managers[0]) is not FullAttentionManager
            or getattr(coordinator.single_type_managers[0].find_longest_cache_hit, "__func__", None)
            is not native_find
            or type(pool) is not BlockPool
            or pool.hash_block_size != size
            or getattr(pool.get_cached_block, "__func__", None) is not native_lookup
            or type(index) is not BlockHashToBlockMap
            or getattr(index.get_one_block, "__func__", None) is not native_get
        ):
            return original(block_hashes, max_cache_hit_length)
        cache = index._cache
        blocks = []
        for block_hash in islice(block_hashes, max_cache_hit_length // size):
            value = cache.get(block_hash + b"\x00\x00\x00\x00")
            if value is None:
                break
            if isinstance(value, KVCacheBlock):
                blocks.append(value)
            elif isinstance(value, dict):
                blocks.append(next(iter(value.values())))
            else:
                index._unexpected_blocks_type(value)
        return (blocks,), len(blocks) * size

    find_longest_cache_hit._gr_bulk_lookup = True  # type: ignore[attr-defined]
    coordinator.find_longest_cache_hit = find_longest_cache_hit
