# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Runtime device policy helpers shared by recommendation model wrappers.

These helpers resolve which accelerator a model's dense forward runs on and,
optionally, which device holds the embedding tables for the two-phase
(preprocess + dense) execution path used by pooling-style recommendation
models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from vllm_gr.utils.npu_adapter import is_cuda_available, is_npu_available


@dataclass(frozen=True)
class RuntimeDevices:
    """Resolved runtime device policy for the two-phase execution path."""

    dense_device: torch.device
    embedding_device: torch.device


def resolve_runtime_devices(
    module: Optional[torch.nn.Module] = None,
    *,
    explicit_dense_device: Optional[torch.device] = None,
    explicit_embedding_device: Optional[torch.device] = None,
) -> RuntimeDevices:
    """Resolve the single runtime policy for dense and embedding phases.

    Current policy:
    - dense phase follows an explicit override, otherwise the active
      accelerator when available
    - embedding phase defaults to CPU unless explicitly overridden

    The embedding decision intentionally does not infer from model weights.
    It must come from an explicit runtime choice or the conservative default.
    """
    if explicit_dense_device is not None:
        dense_device = torch.device(explicit_dense_device)
    else:
        dense_device = _infer_dense_device(module)

    if explicit_embedding_device is not None:
        embedding_device = torch.device(explicit_embedding_device)
    else:
        embedding_device = torch.device("cpu")

    return RuntimeDevices(
        dense_device=dense_device,
        embedding_device=embedding_device,
    )


def _infer_dense_device(
    module: Optional[torch.nn.Module],
) -> torch.device:
    if module is None:
        return _get_active_dense_device()

    for name, param in module.named_parameters():
        if "embedding_collection" not in name:
            return param.device

    for name, buffer in module.named_buffers():
        if "embedding_collection" not in name:
            return buffer.device

    try:
        return next(module.parameters()).device
    except StopIteration:
        return _get_active_dense_device()


def _get_active_dense_device() -> torch.device:
    """Resolve the active accelerator for dense execution."""
    if is_npu_available():
        try:
            return torch.device("npu", torch.npu.current_device())
        except (AttributeError, RuntimeError):
            return torch.device("npu")

    if is_cuda_available():
        try:
            return torch.device("cuda", torch.cuda.current_device())
        except (AttributeError, RuntimeError):
            return torch.device("cuda")

    return torch.device("cpu")


__all__ = [
    "RuntimeDevices",
    "resolve_runtime_devices",
]
