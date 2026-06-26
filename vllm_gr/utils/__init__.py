# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utility helpers for backend detection and safe operator dispatch."""

from vllm_gr.utils.npu_adapter import (
    DeviceAdapter,
    DeviceType,
    get_default_device,
    get_device_adapter,
    get_device_backend,
    get_device_type,
    is_cuda_available,
    is_cuda_device,
    is_npu_available,
    is_npu_device,
    release_accelerator_memory,
    reset_device_adapter,
)
from vllm_gr.utils.reliable_ops import ReliableOps
from vllm_gr.utils.runtime_devices import RuntimeDevices, resolve_runtime_devices

__all__ = [
    "DeviceAdapter",
    "DeviceType",
    "ReliableOps",
    "RuntimeDevices",
    "get_device_adapter",
    "get_default_device",
    "get_device_backend",
    "get_device_type",
    "is_cuda_available",
    "is_cuda_device",
    "is_npu_available",
    "is_npu_device",
    "release_accelerator_memory",
    "reset_device_adapter",
    "resolve_runtime_devices",
]
