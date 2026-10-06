from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from speech_projector.generation import (
    CompletedGeneration,
    GenerationResult,
    TokenLimitedGeneration,
)
from speech_projector.inputs import UtteranceInput
from speech_projector.journal import read_journal
from speech_projector.llm import FrozenQwen
from speech_projector.models import Example, Split
from speech_projector.teacher import (
    BootstrapTeacherSelection,
    TeacherConfig,
    TeacherFailure,
    TeacherFailureReason,
    TeacherTarget,
    generate_targets,
    measured_progress,
    select_examples,
    target_example,
)
from speech_projector.teacher_configuration import teacher_compression_runs


@dataclass(frozen=True)
class GenerationCall:
    identifiers: tuple[str, ...]
    token_cap: int


class TeacherQwen(FrozenQwen):
    def __init__(self, responses: tuple[tuple[GenerationResult, ...], ...]) -> None:
        self.config = teacher_compression_runs()[0]
        tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "yes": 1, "no": 2}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
        self.responses = list(responses)
        self.calls: list[GenerationCall] = []

    def generate_batch(
        self,
        examples: Sequence[Example],
        utterances: Sequence[UtteranceInput],
        max_new_tokens: int,
    ) -> tuple[GenerationResult, ...]:
        assert len(examples) == len(utterances)
        self.calls.append(
            GenerationCall(tuple(example.example_id for example in examples), max_new_tokens)
        )
        return self.responses.pop(0)


def example(index: int, split: Split = Split.TRAIN) -> Example:
    return Example(
        example_id=f"{split.value}-{index}",
        dialogue_id=f"dialogue-{index}",
        split=split,
        history=(),
        user_text="no",
        target_text="original dataset target",
        audio_path=Path("audio.wav"),
        duration=1,
        domain="test",
        emotion="neutral",
        feature_path=Path("feature.pt"),
    )


def configuration(directory: Path) -> TeacherConfig:
    return TeacherConfig(
        run=teacher_compression_runs()[0],
        manifest=directory / "source.jsonl",
        output_directory=directory,
    )


def test_completed_journal_replaces_only_target_and_preserves_original(tmp_path: Path) -> None:
    original = example(0)
    completed = CompletedGeneration(text="yes", token_ids=(1, 248046))
    wrapper = TeacherQwen(((completed,),))
    targets = generate_targets(wrapper, (original,), configuration(tmp_path))
    assert read_journal(tmp_path / "targets.jsonl", TeacherTarget) == targets
    distilled = target_example(targets[0], wrapper)
    assert distilled.target_text == "yes"
    assert distilled.model_copy(update={"target_text": original.target_text}) == original
    assert targets[0].example.target_text == "original dataset target"


def test_token_cap_retries_and_never_journals_partial_target(tmp_path: Path) -> None:
    capped = TokenLimitedGeneration(partial_text="unfinished", token_ids=(1,))
    completed = CompletedGeneration(text="yes", token_ids=(1, 248046))
    wrapper = TeacherQwen(((capped,), (completed,)))
    policy = configuration(tmp_path)
    targets = generate_targets(wrapper, (example(0),), policy)
    assert wrapper.calls[0].token_cap == policy.initial_max_new_tokens
    assert wrapper.calls[1].token_cap == policy.retry_max_new_tokens
    assert targets[0].response == completed
    assert targets[0].capped_attempts == (capped,)


def test_unfinished_retry_saves_explicit_failure_without_target(tmp_path: Path) -> None:
    capped = TokenLimitedGeneration(partial_text="unfinished", token_ids=(1,))
    wrapper = TeacherQwen(((capped,), (capped,)))
    with pytest.raises(ValueError, match="did not complete"):
        generate_targets(wrapper, (example(0),), configuration(tmp_path))
    failures = read_journal(tmp_path / "failures.jsonl", TeacherFailure)
    assert failures[0].reason == TeacherFailureReason.TOKEN_LIMIT
    assert not (tmp_path / "targets.jsonl").exists()


def test_teacher_target_budget_is_validated_without_silent_truncation(tmp_path: Path) -> None:
    wrapper = TeacherQwen(())
    wrapper.config = wrapper.config.model_copy(update={"max_target_tokens": 2})
    target = TeacherTarget(
        example=example(0),
        response=CompletedGeneration(text="yes yes", token_ids=(1, 1, 248046)),
        capped_attempts=(),
    )
    with pytest.raises(ValueError, match="exceeds"):
        target_example(target, wrapper)


def test_bootstrap_is_deterministic_nested_subset_without_target_placeholders(
    tmp_path: Path,
) -> None:
    examples = tuple(example(index, split) for split in Split for index in range(4))
    selection = BootstrapTeacherSelection(
        output_manifest=tmp_path / "bootstrap.jsonl",
        training_examples=2,
        validation_examples=1,
        test_examples=1,
    )
    selected = select_examples(examples, selection)
    assert tuple(item.example_id for item in selected) == (
        "train-0",
        "train-1",
        "validation-0",
        "test-0",
    )


def test_resumed_progress_uses_cumulative_time_and_tokens() -> None:
    target = TeacherTarget(
        example=example(0),
        response=CompletedGeneration(text="yes", token_ids=(1, 248046)),
        capped_attempts=(),
    )
    progress = measured_progress((target, target), 10, 4.2)
    assert progress.examples_per_second == 0.2
    assert progress.tokens_per_second == 0.4
    assert progress.peak_vram_gb == 4.2
