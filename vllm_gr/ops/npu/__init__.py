# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reserved NPU operator namespace.

Ascend fused operators, including split-QKV-RMSNorm-RoPE, are registered by
vLLM-Ascend 0.22.1rc1.
"""

from vllm_gr.ops.npu.rec_constrained_topk import (
    is_available as rec_constrained_topk_npu_available,
)
from vllm_gr.ops.npu.rec_constrained_topk import (
    rec_constrained_topk_npu,
)

__all__ = ["rec_constrained_topk_npu", "rec_constrained_topk_npu_available"]
