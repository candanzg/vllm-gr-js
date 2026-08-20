# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch


@dataclass
class BeamConstraintTables:
    trie: Optional[Any] = None
    constraint_tables: Optional[Mapping[str, torch.Tensor]] = None


@dataclass
class BeamSearchContext:
    current_step: int = 0
    sequence: Optional[torch.Tensor] = None
    temperatures: Optional[torch.Tensor] = None
    tables: BeamConstraintTables = field(default_factory=BeamConstraintTables)
