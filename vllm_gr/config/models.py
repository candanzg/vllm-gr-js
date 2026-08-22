# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GRAttentionBackend(str, Enum):
    CUSTOM = "CUSTOM"


class ConstraintBackend(str, Enum):
    TRIE = "trie"
    CONSTRAINT_TABLE = "constraint_table"


class NPUDecodeExecution(str, Enum):
    """Execution mode for the NPU incremental beam decode branch."""

    CASCADE = "cascade"
    XLLM = "xllm"


@dataclass(frozen=True)
class GRConfig:
    """Validated vLLM-GR configuration reconstructed in each process."""

    catalog_path: str | None = None
    constraint_backend: ConstraintBackend = ConstraintBackend.TRIE
    attention_backend: GRAttentionBackend | None = None
    vocab_offset: int | None = None
    vocab_slice_size: int | None = None
    # None = unset: the runtime picks per-platform defaults (xllm on NPU when
    # the operator library is present, cascade otherwise and on CUDA).
    npu_decode_execution: NPUDecodeExecution | None = None

    def __post_init__(self) -> None:
        from vllm_gr.config.validation import (
            validate_attention_backend,
            validate_catalog_path,
            validate_constraint_backend,
            validate_npu_decode_execution,
            validate_vocab_offset,
            validate_vocab_slice_size,
        )

        object.__setattr__(self, "catalog_path", validate_catalog_path(self.catalog_path))
        object.__setattr__(
            self,
            "constraint_backend",
            validate_constraint_backend(self.constraint_backend),
        )
        object.__setattr__(
            self,
            "attention_backend",
            validate_attention_backend(self.attention_backend),
        )
        object.__setattr__(self, "vocab_offset", validate_vocab_offset(self.vocab_offset))
        object.__setattr__(
            self, "vocab_slice_size", validate_vocab_slice_size(self.vocab_slice_size)
        )
        object.__setattr__(
            self,
            "npu_decode_execution",
            validate_npu_decode_execution(self.npu_decode_execution),
        )
        if (
            self.constraint_backend is ConstraintBackend.CONSTRAINT_TABLE
            and self.catalog_path is None
        ):
            raise ValueError("vllm-gr constraint_backend='constraint_table' requires catalog_path")

    @property
    def enabled(self) -> bool:
        return (
            self.catalog_path is not None
            or self.constraint_backend is not ConstraintBackend.TRIE
            or self.attention_backend is not None
            or self.vocab_offset is not None
            or self.vocab_slice_size is not None
            or self.npu_decode_execution is not None
        )

    @classmethod
    def from_vllm_config(cls, vllm_config: object | None) -> GRConfig:
        from vllm_gr.config.parser import parse_gr_config

        return parse_gr_config(vllm_config)

    def write_to_vllm_config(self, vllm_config: object) -> None:
        from vllm_gr.config.parser import write_gr_config

        write_gr_config(vllm_config, self)
