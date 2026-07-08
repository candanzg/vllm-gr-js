# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vllm-gr operator packages.

After the HSTU / DLRM-v3 / FuXi runtime was removed, the paged-KV attention
ops and jagged-tensor helpers that served it are gone. The only remaining
operator package is :mod:`vllm_gr.ops.npu`, which hosts the NPU (Ascend)
custom-op patches applied by ``vllm_gr.plugin`` / ``vllm_gr.patch``.

Imports stay lazy so GPU-only and test environments can import ``vllm_gr``
without pulling in the Ascend-specific Triton patch module.
"""

__all__ = [
    "apply_split_qkv_rmsnorm_rope_patch",
]


def apply_split_qkv_rmsnorm_rope_patch(*args, **kwargs):
    from vllm_gr.ops.npu.split_qkv_rmsnorm_rope import (
        apply_split_qkv_rmsnorm_rope_patch as _impl,
    )

    return _impl(*args, **kwargs)
