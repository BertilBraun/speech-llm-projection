from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from speech_projector.generation import CompletedGeneration
from speech_projector.inputs import TranscriptInput, UtteranceInput
from speech_projector.llm import FrozenQwen
from speech_projector.models import EvaluationCondition, Example, Split
from speech_projector.teacher import TeacherTarget, target_example
from speech_projector.teacher_baselines import evaluate_cached_teacher
from speech_projector.teacher_configuration import teacher_feasibility_run


class BaselineQwen(FrozenQwen):
    def __init__(self) -> None:
        self.config = teacher_feasibility_run()
        vocabulary = Tokenizer(WordLevel({"[UNK]": 0, "Cached": 1, "reply": 2}, unk_token="[UNK]"))
        vocabulary.pre_tokenizer = Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=vocabulary, unk_token="[UNK]")
        self.received_inputs: list[UtteranceInput] = []

    def target_token_count(self, example: Example) -> int:
        return len(self.tokenizer.encode(example.target_text, add_special_tokens=False)) + 1

    def loss(self, example: Example, utterance: UtteranceInput) -> Tensor:
        self.received_inputs.append(utterance)
        assert example.target_text == "Cached reply"
        return torch.tensor(0.75)

    def generate(self, example: Example, utterance: UtteranceInput) -> str:
        raise AssertionError("The actual teacher reference must not be regenerated")


def teacher_target(directory: Path) -> TeacherTarget:
    return TeacherTarget(
        example=Example(
            example_id="one",
            dialogue_id="one-dialogue",
            split=Split.VALIDATION,
            history=(),
            user_text="Actual synthesis transcript",
            target_text="Original dataset response",
            audio_path=directory / "clip.wav",
            feature_path=directory / "feature.pt",
            duration=2,
            domain="test",
            emotion="neutral",
        ),
        response=CompletedGeneration(text="Cached reply", token_ids=(1, 2, 3)),
        capped_attempts=(),
    )


def test_text_reference_uses_saved_teacher_trajectory_and_fresh_loss(tmp_path: Path) -> None:
    wrapper = BaselineQwen()
    target = teacher_target(tmp_path)
    example = target_example(target, wrapper)
    outcome = evaluate_cached_teacher(wrapper, (example,), (target,))
    assert wrapper.received_inputs == [TranscriptInput(target.example.user_text)]
    assert outcome.metrics.cross_entropy == pytest.approx(0.75)
    assert outcome.metrics.target_tokens == 3
    assert outcome.metrics.generation_seconds == 0
    assert outcome.metrics.generated_tokens == 3
    assert outcome.samples[0].generated_response == target.response.text
    assert outcome.samples[0].gold_response == target.response.text
    assert outcome.samples[0].condition == EvaluationCondition.TEXT
    assert outcome.samples[0].semantic_similarity == 1
    assert target.example.target_text == "Original dataset response"


@pytest.mark.parametrize("field", ("user_text", "target_text"))
def test_cached_reference_rejects_wrong_transcript_or_teacher_target(
    tmp_path: Path, field: str
) -> None:
    wrapper = BaselineQwen()
    target = teacher_target(tmp_path)
    altered = target_example(target, wrapper).model_copy(update={field: "Different"})
    with pytest.raises(ValueError, match="differs from evaluation manifest"):
        evaluate_cached_teacher(wrapper, (altered,), (target,))
    assert not wrapper.received_inputs
