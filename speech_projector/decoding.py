"""Explicit Hugging Face decoding settings for transcript and speech prompts."""

import math

import torch
from torch import Tensor
from transformers import (
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedTokenizerBase,
)

from speech_projector.models import DecodingConfig, GreedyDecodingConfig, SamplingDecodingConfig


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """Subtract a penalty once for each token already generated in that batch row."""

    def __init__(self, penalty: float) -> None:
        if not math.isfinite(penalty):
            raise ValueError("Presence penalty must be finite")
        self.penalty = penalty

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        # With inputs_embeds, HF supplies generated-only IDs, empty at the first step.
        present = torch.zeros_like(scores, dtype=torch.bool)
        present.scatter_(dim=1, index=input_ids, value=True)
        return scores - present.to(dtype=scores.dtype) * self.penalty


def generation_parameters(
    config: DecodingConfig,
    max_new_tokens: int,
    tokenizer: PreTrainedTokenizerBase,
) -> GenerationConfig:
    if max_new_tokens <= 0:
        raise ValueError("Generation token cap must be positive")
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Generation requires explicit tokenizer padding and EOS token identifiers")
    match config:
        case GreedyDecodingConfig():
            return GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        case SamplingDecodingConfig():
            return GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                min_p=config.min_p,
                repetition_penalty=config.repetition_penalty,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )


def generation_processors(config: DecodingConfig) -> LogitsProcessorList:
    match config:
        case GreedyDecodingConfig():
            return LogitsProcessorList()
        case SamplingDecodingConfig():
            return LogitsProcessorList([PresencePenaltyLogitsProcessor(config.presence_penalty)])
