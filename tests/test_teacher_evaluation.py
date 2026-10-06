from pathlib import Path

import pytest
import torch
from torch.nn import functional

from speech_projector.llm import TargetScores
from speech_projector.models import EvaluationCondition, Example, MlpProjectorConfig, Split
from speech_projector.projectors import Projector
from speech_projector.teacher_evaluation import (
    compare_target_scores,
    exact_teacher_kl,
    projector_digest,
    summarize_fidelity,
)


def example() -> Example:
    return Example(
        example_id="example",
        dialogue_id="dialogue",
        split=Split.VALIDATION,
        history=(),
        user_text="Where is the canyon?",
        target_text="In Arizona.",
        audio_path=Path("audio.wav"),
        feature_path=Path("audio.pt"),
        duration=1,
        domain="travel",
        emotion="neutral",
    )


def test_aligned_target_prefix_agreement_differs_from_target_accuracy() -> None:
    targets = torch.tensor([0, 2])
    teacher = TargetScores(targets, torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]]))
    speech = TargetScores(targets, torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]))
    metrics = compare_target_scores(example(), EvaluationCondition.SPEECH, speech, teacher)
    assert metrics.top1_agreement == 1
    assert metrics.input_target_accuracy == 0.5
    assert metrics.transcript_target_accuracy == 0.5
    assert metrics.first_token_target_accuracy == 1
    assert metrics.first_8_tokens == 2
    assert metrics.first_8_token_target_accuracy == 0.5
    assert metrics.teacher_to_input_kl > 0
    assert metrics.excess_cross_entropy == pytest.approx(
        metrics.input_cross_entropy - metrics.transcript_cross_entropy
    )
    with pytest.raises(ValueError, match="identical teacher-forced"):
        compare_target_scores(
            example(),
            EvaluationCondition.SPEECH,
            TargetScores(torch.tensor([0, 1]), speech.logits),
            teacher,
        )


@pytest.mark.parametrize("chunk", [1, 2, 16])
def test_online_kl_matches_full_distribution(chunk: int) -> None:
    teacher = torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]], dtype=torch.float32)
    speech = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]], dtype=torch.float32)
    expected = functional.kl_div(
        functional.log_softmax(speech, -1), functional.softmax(teacher, -1), reduction="batchmean"
    )
    assert exact_teacher_kl(teacher, speech, chunk) == pytest.approx(float(expected), abs=1e-7)
    assert exact_teacher_kl(teacher, teacher, chunk) == 0


def test_fidelity_summary_weights_tokens_and_preserves_condition() -> None:
    targets = torch.tensor([0, 1])
    teacher = TargetScores(targets, torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    first = compare_target_scores(example(), EvaluationCondition.SPEECH, teacher, teacher)
    second = compare_target_scores(
        example(),
        EvaluationCondition.SHUFFLED_SPEECH,
        TargetScores(targets, teacher.logits.flip(-1)),
        teacher,
    )
    summary = summarize_fidelity((first, second))
    assert summary.examples == 1 and summary.target_tokens == 2
    assert summary.top1_agreement == 1
    assert summary.excess_cross_entropy == 0
    control = summarize_fidelity((first, second), EvaluationCondition.SHUFFLED_SPEECH)
    assert control.top1_agreement == 0
    assert control.excess_cross_entropy > 0


def test_projector_digest_detects_same_config_changed_weights() -> None:
    projector = Projector(
        MlpProjectorConfig(
            compression_factor=2, encoder_dimension=4, embedding_dimension=8, hidden_dimension=6
        )
    )
    before = projector_digest(projector)
    assert projector_digest(projector) == before
    with torch.no_grad():
        projector.normalization.weight.add_(0.1)
    assert projector_digest(projector) != before
