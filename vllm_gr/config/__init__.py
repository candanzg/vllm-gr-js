# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Configuration module for vllm-GR.

Provides:
- FuXiConfig: HuggingFace-compatible PretrainedConfig subclass (model arch)
- GRConfig: Typed runtime / serving configuration dataclass
"""

from vllm_gr.config.fuxi_config import FuXiConfig
from vllm_gr.config.gr_config import GRConfig

__all__ = [
    "FuXiConfig",
    "GRConfig",
]
