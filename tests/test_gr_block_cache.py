# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare prefix-cache lookup, reset and hooks with native vLLM."""

from types import SimpleNamespace as NS

import pytest
import torch
from vllm.v1.core.block_pool import BlockHashToBlockMap, BlockPool
from vllm.v1.core.kv_cache_coordinator import UnitaryKVCacheCoordinator
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_gr.v1.engine.gr_block_cache import install_gr_cache_lookup


def request(count=16, gr=True, duplicate=False):
    return NS(
        block_hashes=[(i % 3 if duplicate else i).to_bytes(32, "big") for i in range(count)],
        gr_batch_request=object() if gr else None,
        lora_request=None,
        mm_features=[],
        cache_salt=None,
        prompt_embeds=None,
        all_token_ids=list(range(count * 16)),
    )


def coordinator(pool):
    c = object.__new__(UnitaryKVCacheCoordinator)
    c.block_pool = pool
    c.block_size = 16
    c.hash_block_size = 16
    c.kv_cache_spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=32, dtype=torch.float16
    )
    c.single_type_managers = [object.__new__(FullAttentionManager)]
    c.eagle_group_ids = set()
    c.dcp_world_size = c.pcp_world_size = 1
    return c


def normalized(result):
    blocks, length = result
    return tuple([b.block_id for b in group] for group in blocks), length


@pytest.mark.parametrize("length", [0, 15, 16, 47, 160, 165, 1024])
@pytest.mark.parametrize("case", ["hit", "miss", "hole", "duplicate", "eagle", "cp"])
def test_lookup_matches_native(length, case):
    pool = BlockPool(32, True, 16)
    req = request(10)
    if case != "miss":
        blocks = pool.get_new_blocks(10)
        pool.cache_full_blocks(req, blocks, 0, 10, 16, 0)
        if case == "hole":
            pool.cached_block_hash_to_block.pop(blocks[3].block_hash, blocks[3].block_id)
        elif case == "duplicate":
            extra = pool.get_new_blocks(10)
            pool.cache_full_blocks(req, extra, 0, 10, 16, 0)
    a, b = coordinator(pool), coordinator(pool)
    if case == "eagle":
        a.eagle_group_ids = b.eagle_group_ids = {0}
    elif case == "cp":
        a.dcp_world_size = b.dcp_world_size = 2
        a.block_size = b.block_size = 32
    install_gr_cache_lookup(b)
    installed = b.find_longest_cache_hit
    install_gr_cache_lookup(b)
    assert b.find_longest_cache_hit is installed
    assert normalized(a.find_longest_cache_hit(req.block_hashes, length)) == normalized(
        b.find_longest_cache_hit(req.block_hashes, length)
    )


def test_lookup_refreshes_replaced_map_and_preserves_hooks():
    pool = BlockPool(32, True, 16)
    req = request(4)
    a, b = coordinator(pool), coordinator(pool)
    install_gr_cache_lookup(b)
    blocks = pool.get_new_blocks(4)
    pool.cache_full_blocks(req, blocks, 0, 4, 16, 0)
    assert normalized(b.find_longest_cache_hit(req.block_hashes, 64))[1] == 64
    pool.cached_block_hash_to_block = BlockHashToBlockMap()
    assert normalized(b.find_longest_cache_hit(req.block_hashes, 64)) == (([],), 0)
    extra = pool.get_new_blocks(4)
    pool.cache_full_blocks(req, extra, 0, 4, 16, 0)
    assert normalized(a.find_longest_cache_hit(req.block_hashes, 64)) == normalized(
        b.find_longest_cache_hit(req.block_hashes, 64)
    )
    original = pool.get_cached_block
    calls = []

    def hooked(*args):
        calls.append(args)
        return original(*args)

    pool.get_cached_block = hooked
    assert normalized(b.find_longest_cache_hit(req.block_hashes, 64))[1] == 64
    assert len(calls) == 4


@pytest.mark.parametrize("layer", ["map", "manager"])
def test_lookup_preserves_inner_hooks(layer):
    pool = BlockPool(16, True, 16)
    req = request(4)
    blocks = pool.get_new_blocks(4)
    pool.cache_full_blocks(req, blocks, 0, 4, 16, 0)
    c = coordinator(pool)
    install_gr_cache_lookup(c)
    obj, name = (
        (pool.cached_block_hash_to_block, "get_one_block")
        if layer == "map"
        else (c.single_type_managers[0], "find_longest_cache_hit")
    )
    original = getattr(obj, name)
    calls = []

    def hooked(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    setattr(obj, name, hooked)
    assert normalized(c.find_longest_cache_hit(req.block_hashes, 64))[1] == 64
    assert len(calls) == (4 if layer == "map" else 1)


def test_install_preserves_preexisting_hooks():
    pool = BlockPool(16, True, 16)
    c = coordinator(pool)
    c.find_longest_cache_hit = lambda *args: (([],), 0)
    lookup = c.find_longest_cache_hit
    install_gr_cache_lookup(c)
    assert lookup is c.find_longest_cache_hit
