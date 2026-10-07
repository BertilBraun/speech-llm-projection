"""A generation-panel comparison must not silently reuse a larger CE cohort."""

import pytest

from speech_projector.evaluation import EvaluationOutcome, ExampleLoss
from speech_projector.followup_matched_metrics import subset_metrics
from speech_projector.models import EvaluationCondition, EvaluationMetrics, SampleGeneration


def test_matched_ce_reweights_exact_ids_and_rejects_missing_generation() -> None:
    outcome = EvaluationOutcome(
        metrics=EvaluationMetrics(examples=3, target_tokens=12, cross_entropy=1.5, perplexity=4.5),
        samples=tuple(
            SampleGeneration(
                example_id=identifier,
                dialogue_id=identifier,
                condition=EvaluationCondition.ASR,
                history=(),
                user_transcript="Hello.",
                gold_response="Reply.",
                generated_response="Hello.",
                duration=1,
            )
            for identifier in ("one", "two", "three")
        ),
        example_losses=tuple(
            ExampleLoss(
                example_id=identifier,
                dialogue_id=identifier,
                condition=EvaluationCondition.ASR,
                cross_entropy=loss,
                target_tokens=tokens,
            )
            for identifier, loss, tokens in (("one", 1, 10), ("two", 3, 1), ("three", 5, 1))
        ),
        diagnostics=(),
    )
    selected = subset_metrics(outcome, EvaluationCondition.ASR, ("one", "two"))
    assert selected.examples == selected.generated_examples == 2
    assert selected.target_tokens == 11
    assert selected.cross_entropy == pytest.approx(13 / 11)
    assert selected.completed_generations is None
    with pytest.raises(ValueError, match="one loss and one generation"):
        subset_metrics(outcome, EvaluationCondition.ASR, ("one", "missing"))
