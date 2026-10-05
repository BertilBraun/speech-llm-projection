from pathlib import Path

import pytest
import torch

from speech_projector.evaluation import (
    EvaluationCondition,
    ExampleLoss,
    LossObservation,
    conditioning_diagnostic,
    load_asr_transcripts,
    match_feature_length,
    summarize_losses,
)


def test_loss_weights_assistant_tokens() -> None:
    cross_entropy, perplexity, tokens = summarize_losses(
        [LossObservation(10.0, 10), LossObservation(30.0, 5)]
    )
    assert tokens == 15
    assert cross_entropy == pytest.approx(40 / 15)
    assert perplexity == pytest.approx(14.391916)


@pytest.mark.parametrize("observations", [[], [LossObservation(0.0, 0)]])
def test_loss_rejects_no_target_tokens(observations: list[LossObservation]) -> None:
    with pytest.raises(ValueError):
        summarize_losses(observations)


def test_conditioning_is_paired_and_reports_direction() -> None:
    losses = [
        ExampleLoss(
            example_id="a",
            dialogue_id="d1",
            condition=EvaluationCondition.SPEECH,
            cross_entropy=2.0,
            target_tokens=5,
        ),
        ExampleLoss(
            example_id="a",
            dialogue_id="d1",
            condition=EvaluationCondition.SHUFFLED_SPEECH,
            cross_entropy=3.0,
            target_tokens=5,
        ),
        ExampleLoss(
            example_id="b",
            dialogue_id="d2",
            condition=EvaluationCondition.SPEECH,
            cross_entropy=3.0,
            target_tokens=8,
        ),
        ExampleLoss(
            example_id="b",
            dialogue_id="d2",
            condition=EvaluationCondition.SHUFFLED_SPEECH,
            cross_entropy=3.0,
            target_tokens=8,
        ),
    ]
    diagnostic = conditioning_diagnostic(
        losses, EvaluationCondition.SPEECH, EvaluationCondition.SHUFFLED_SPEECH
    )
    assert diagnostic.mean_control_minus_correct_ce == 0.5
    assert diagnostic.standard_error == pytest.approx(0.5)
    assert diagnostic.fraction_correct_audio_lower_loss == 0.5
    reordered = [losses[0], losses[3], losses[2], losses[1]]
    with pytest.raises(ValueError, match="match examples"):
        conditioning_diagnostic(
            reordered, EvaluationCondition.SPEECH, EvaluationCondition.SHUFFLED_SPEECH
        )


def test_load_asr_jsonl_requires_schema(tmp_path: Path) -> None:
    path = tmp_path / "asr.jsonl"
    path.write_text('{"example_id":"a","text":"hello"}\n\n', encoding="utf-8")
    records = load_asr_transcripts(path)
    assert records[0].text == "hello"
    path.write_text('{"example_id":"a","text":"hello","extra":true}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_asr_transcripts(path)


@pytest.mark.parametrize("source_length,target_length", [(4, 8), (8, 4), (4, 4)])
def test_shuffled_control_preserves_length_dtype_and_constant_states(
    source_length: int, target_length: int
) -> None:
    features = torch.full((source_length, 768), 2.0, dtype=torch.bfloat16)
    matched = match_feature_length(features, target_length)
    assert matched.shape == (target_length, 768)
    assert matched.dtype == features.dtype
    assert torch.all(matched == 2)
    if source_length == target_length:
        assert matched is features
