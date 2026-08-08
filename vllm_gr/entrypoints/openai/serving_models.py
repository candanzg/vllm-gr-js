# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.models.serving import (
    BaseModelPath,
    LoRAModulePath,
    OpenAIServingModels,
)

from vllm_gr.config import GRConfig
from vllm_gr.constraints import BeamConstraintCoordinator, create_constraint_strategy
from vllm_gr.constraints.strategies import TrieConstraintStrategy

_original_init = OpenAIServingModels.__init__


def init_gr(
    self,
    engine_client: EngineClient,
    base_model_paths: list[BaseModelPath],
    *,
    lora_modules: list[LoRAModulePath] | None = None,
):
    _original_init(self, engine_client, base_model_paths, lora_modules=lora_modules)
    self.catalog = None
    self.constraint_strategy = None
    self.constraint_coordinator = BeamConstraintCoordinator(None)
    tokenizer = self.input_processor.tokenizer
    vllm_config = getattr(engine_client, "vllm_config", None)
    gr_config = GRConfig.from_vllm_config(vllm_config)
    if gr_config.catalog_path:
        self.constraint_strategy = create_constraint_strategy(
            backend=gr_config.constraint_backend,
            tokenizer=tokenizer,
            catalog_path=gr_config.catalog_path,
        )
        self.constraint_coordinator = BeamConstraintCoordinator(self.constraint_strategy)
        if isinstance(self.constraint_strategy, TrieConstraintStrategy):
            # Compatibility for existing callers that inspect ``models.catalog``.
            self.catalog = self.constraint_strategy.catalog


def patch_OpenAIServingModels_init():
    OpenAIServingModels.__init__ = init_gr
