from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import torch
from transformers import PreTrainedTokenizerBase

from speech_projector.evaluation_generation import (
    EvaluationGenerationRequest,
    generate_requests,
    generation_input_key,
)
from speech_projector.generation import (
    CompletedGeneration,
    GenerationResult,
    TokenLimitedGeneration,
)
from speech_projector.inputs import SpeechInput, TranscriptInput, UtteranceInput
from speech_projector.llm import FrozenQwen
from speech_projector.models import Example, Role, RunConfig, Split, Turn
from speech_projector.teacher_configuration import teacher_compression_runs


@dataclass
class RecordingTokenizer:
    encoded: list[str]

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        self.encoded.append(text)
        return list(range(len(text.split())))


class RecordingWrapper:
    def __init__(self, config: RunConfig, tokenizer: RecordingTokenizer) -> None:
        self.config = config
        self.tokenizer = cast(PreTrainedTokenizerBase, tokenizer)
        self.batches: list[tuple[Example, ...]] = []
        self.utterances: list[tuple[UtteranceInput, ...]] = []

    def generate_batch(
        self,
        examples: Sequence[Example],
        utterances: Sequence[UtteranceInput],
        max_new_tokens: int,
    ) -> tuple[GenerationResult, ...]:
        assert max_new_tokens == self.config.max_new_tokens
        assert not torch.is_grad_enabled()
        self.batches.append(tuple(examples))
        self.utterances.append(tuple(utterances))
        return tuple(
            TokenLimitedGeneration(partial_text=example.example_id, token_ids=(1, 2, 3))
            if example.example_id == "capped"
            else CompletedGeneration(text=example.example_id, token_ids=(1, 9))
            for example in examples
        )


def example(identifier: str, history: tuple[Turn, ...] = ()) -> Example:
    return Example(
        example_id=identifier,
        dialogue_id=f"dialogue-{identifier}",
        split=Split.VALIDATION,
        history=history,
        user_text="CURRENT_TRANSCRIPT_MUST_NOT_BE_READ",
        target_text="GOLD_RESPONSE_MUST_NOT_BE_READ",
        audio_path=Path("audio.wav"),
        feature_path=Path("features.pt"),
        duration=1.0,
        domain="science",
        emotion="neutral",
    )


def speech_request(
    identifier: str, tokens: int, history: tuple[Turn, ...] = ()
) -> EvaluationGenerationRequest:
    return EvaluationGenerationRequest(
        example(identifier, history), SpeechInput(torch.ones(tokens, 2))
    )


def wrapper() -> tuple[RecordingWrapper, RecordingTokenizer]:
    tokenizer = RecordingTokenizer([])
    config = teacher_compression_runs()[0].model_copy(
        update={"max_new_tokens": 9, "history_turns": 2, "max_history_tokens": 8}
    )
    return RecordingWrapper(config, tokenizer), tokenizer


def test_grouping_restores_order_and_preserves_completion_and_exact_token_counts() -> None:
    recording, tokenizer = wrapper()
    history = (Turn(role=Role.USER, text="earlier turn"),)
    requests = (
        speech_request("opening-long", 9),
        speech_request("continuation-long", 4, history),
        speech_request("capped", 1),
        speech_request("continuation-short", 1, history),
        speech_request("opening-medium", 4),
    )
    outcome = generate_requests(cast(FrozenQwen, recording), requests, 2)
    assert tuple(tuple(item.example_id for item in batch) for batch in recording.batches) == (
        ("continuation-short", "continuation-long"),
        ("capped", "opening-medium"),
        ("opening-long",),
    )
    assert outcome.responses == (
        CompletedGeneration(text="opening-long", token_ids=(1, 9)),
        CompletedGeneration(text="continuation-long", token_ids=(1, 9)),
        TokenLimitedGeneration(partial_text="capped", token_ids=(1, 2, 3)),
        CompletedGeneration(text="continuation-short", token_ids=(1, 9)),
        CompletedGeneration(text="opening-medium", token_ids=(1, 9)),
    )
    assert outcome.generated_tokens_with_eos == 11
    assert outcome.generation_seconds >= 0
    assert tokenizer.encoded == ["earlier turn", "earlier turn"]
    assert all(
        isinstance(utterance, SpeechInput) for batch in recording.utterances for utterance in batch
    )


def test_text_sorting_uses_explicit_input_and_only_selected_history() -> None:
    recording, tokenizer = wrapper()
    history = (
        Turn(role=Role.USER, text="discarded history should not be inspected"),
        Turn(role=Role.ASSISTANT, text="one two three four five six seven eight nine"),
        Turn(role=Role.USER, text="later history"),
    )
    request = EvaluationGenerationRequest(
        example("text", history), TranscriptInput("recognized words")
    )
    key = generation_input_key(cast(FrozenQwen, recording), request)
    assert key == (False, 10)
    assert tokenizer.encoded == [history[1].text, history[2].text, "recognized words"]


def test_empty_request_pool_has_zero_work() -> None:
    recording, tokenizer = wrapper()
    outcome = generate_requests(cast(FrozenQwen, recording), (), 4)
    assert outcome.responses == ()
    assert outcome.generation_seconds == 0
    assert outcome.generated_tokens_with_eos == 0
    assert recording.batches == []
    assert tokenizer.encoded == []


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size_fails_before_model_use(batch_size: int) -> None:
    recording, _ = wrapper()
    with pytest.raises(ValueError, match="batch size"):
        generate_requests(cast(FrozenQwen, recording), (speech_request("one", 1),), batch_size)
    assert recording.batches == []


def test_original_order_is_stable_for_equal_length_inputs_and_large_batch() -> None:
    recording, _ = wrapper()
    requests = tuple(speech_request(identifier, 2) for identifier in ("z", "a", "m"))
    outcome = generate_requests(cast(FrozenQwen, recording), requests, 64)
    assert tuple(
        item.text for item in outcome.responses if isinstance(item, CompletedGeneration)
    ) == ("z", "a", "m")
    assert tuple(item.example_id for item in recording.batches[0]) == ("z", "a", "m")
