"""Length-grouped evaluation generation with explicit completion and original order."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from speech_projector.generation import GenerationResult
from speech_projector.inputs import SpeechInput, TranscriptInput, UtteranceInput
from speech_projector.models import Example

if TYPE_CHECKING:
    from speech_projector.llm import FrozenQwen


@dataclass(frozen=True)
class EvaluationGenerationRequest:
    example: Example
    utterance: UtteranceInput


@dataclass(frozen=True)
class EvaluationGenerationOutcome:
    responses: tuple[GenerationResult, ...]
    generation_seconds: float
    generated_tokens_with_eos: int


def generation_input_key(
    wrapper: FrozenQwen, request: EvaluationGenerationRequest
) -> tuple[bool, int]:
    history = (
        request.example.history[-wrapper.config.history_turns :]
        if wrapper.config.history_turns
        else ()
    )
    history_tokens = min(
        wrapper.config.max_history_tokens,
        sum(len(wrapper.tokenizer.encode(turn.text, add_special_tokens=False)) for turn in history),
    )
    match request.utterance:
        case SpeechInput(embeddings=embeddings):
            current_tokens = embeddings.shape[0]
        case TranscriptInput(text=text):
            current_tokens = len(wrapper.tokenizer.encode(text, add_special_tokens=False))
    # Opening questions often yield longer replies; separating them limits finished-row waste.
    return not bool(history), history_tokens + current_tokens


@torch.no_grad()
def generate_requests(
    wrapper: FrozenQwen,
    requests: Sequence[EvaluationGenerationRequest],
    batch_size: int,
) -> EvaluationGenerationOutcome:
    if batch_size < 1:
        raise ValueError("Evaluation generation batch size must be positive")
    if not requests:
        return EvaluationGenerationOutcome((), 0.0, 0)
    started = time.perf_counter()
    indexed = sorted(enumerate(requests), key=lambda item: generation_input_key(wrapper, item[1]))
    generated: list[tuple[int, GenerationResult]] = []
    for start in range(0, len(indexed), batch_size):
        batch = indexed[start : start + batch_size]
        responses = wrapper.generate_batch(
            tuple(request.example for _, request in batch),
            tuple(request.utterance for _, request in batch),
            wrapper.config.max_new_tokens,
        )
        assert len(responses) == len(batch)
        generated.extend(
            (original_index, response)
            for (original_index, _), response in zip(batch, responses, strict=True)
        )
    ordered = tuple(response for _, response in sorted(generated, key=lambda item: item[0]))
    return EvaluationGenerationOutcome(
        responses=ordered,
        generation_seconds=time.perf_counter() - started,
        generated_tokens_with_eos=sum(len(response.token_ids) for response in ordered),
    )
