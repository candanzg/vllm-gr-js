# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import copy
from typing import Collection

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import BeamSearchParams as BaseBeamSearchParams
from vllm.sampling_params import (
    SamplingParams,
)


def clone_beam_decode_sampling_params(
    params: SamplingParams,
    allowed_token_ids: Collection[int] | None = None,
) -> SamplingParams:
    """Clone sampling params for one logical grouped-beam request.

    Beam width is carried by GR's explicit beam metadata, not by the upstream
    ``SamplingParams.n`` parallel-output field.
    """
    if params.n != 1:
        raise ValueError(
            "GR grouped-beam requests require SamplingParams.n == 1; "
            "use BeamRequestStepUpdate.beam_width for the active beam count."
        )
    cloned = copy(params)
    if allowed_token_ids is not None:
        cloned.allowed_token_ids = sorted(allowed_token_ids)
    return cloned


def validate_beam_logprobs_capacity(runtime: object, beam_width: int) -> None:
    """Fail early when the engine cannot return one candidate per beam slot."""

    vllm_config = getattr(runtime, "vllm_config", None)
    if vllm_config is None:
        # Async clients normally expose the config directly, while a few
        # engine wrappers keep it on their core/engine delegate.
        for owner_name in ("engine_core", "engine"):
            owner = getattr(runtime, owner_name, None)
            vllm_config = getattr(owner, "vllm_config", None)
            if vllm_config is not None:
                break
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        model_config = getattr(runtime, "model_config", None)
    max_logprobs = getattr(model_config, "max_logprobs", None)
    if max_logprobs is None or max_logprobs < 0:
        return
    if max_logprobs < beam_width:
        raise VLLMValidationError(
            "max_logprobs must be greater than or equal to beam_width "
            f"(max_logprobs={max_logprobs}, beam_width={beam_width})",
            parameter="beam_width",
            value=beam_width,
        )


class BeamSearchParams(
    BaseBeamSearchParams,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):  # type: ignore[call-arg]
    """Beam search parameters for text generation."""

    begin_token: str | None = None
    end_token: str | None = None
    # Controls behavior when a beam has no valid catalog token in the model's
    # top-K logprobs (all logprobs filtered to -inf).
    # "fill": fill logprob slots with valid catalog tokens at low score.
    # "null":   leave all logprobs as -inf; the beam is excluded from selection.
    catalog_strategy: str = "fill"

    def __post_init__(self) -> None:
        if self.catalog_strategy not in ("fill", "null"):
            raise VLLMValidationError(
                f"catalog_strategy must be 'fill' or 'null', got {self.catalog_strategy!r}",
                parameter="catalog_strategy",
                value=self.catalog_strategy,
            )
