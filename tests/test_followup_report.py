"""Reports preserve fixed-panel targets and distinct pooled/cohort estimands."""

from pathlib import Path

import pytest

from speech_projector.emotion_preview import Delivery
from speech_projector.evaluation import EvaluationOutcome, ExampleLoss, save_evaluation
from speech_projector.followup_report import (
    BaselineComparisonInput,
    FollowupReportConfig,
    measured_condition,
    write_comparison,
)
from speech_projector.models import EvaluationCondition, EvaluationMetrics, SampleGeneration
from speech_projector.overnight_data import (
    NeuEmotionalExampleSource,
    OrdinaryExampleSource,
    QwenEmotionalExampleSource,
)
from speech_projector.overnight_judge import FinalJudgingQuota
from speech_projector.overnight_launcher import FinalEvaluationSelection
from speech_projector.tts_pilot import PilotEmotion
from tests.test_overnight_evaluation import example


def prepared_comparison(directory: Path) -> FollowupReportConfig:
    examples = tuple(example(identifier, "The deadline changed.") for identifier in ("o", "q", "n"))
    selection = FinalEvaluationSelection(
        quota=FinalJudgingQuota(), examples=examples, generation_example_ids=("o", "q", "n")
    )
    selection_path = directory / "selection.json"
    selection_path.write_text(selection.model_dump_json(), encoding="utf-8")
    source_path = directory / "sources.jsonl"
    sources = (
        OrdinaryExampleSource(example_id="o", source_example_id="o", source_manifest=Path("old")),
        QwenEmotionalExampleSource(
            example_id="q",
            source_example_id="q",
            source_manifest=Path("qwen"),
            base_id="q_base",
            family_id="q_family",
            emotion=Delivery.SAD,
        ),
        NeuEmotionalExampleSource(
            example_id="n",
            source_example_id="n",
            source_manifest=Path("neu"),
            base_id="n_base",
            family_id="n_family",
            emotion=PilotEmotion.ANGRY,
        ),
    )
    source_path.write_text(
        "".join(row.model_dump_json() + "\n" for row in sources), encoding="utf-8"
    )
    outcome = EvaluationOutcome(
        metrics=EvaluationMetrics(examples=3, target_tokens=12, cross_entropy=1.5, perplexity=4.5),
        samples=tuple(
            SampleGeneration(
                example_id=row.example_id,
                dialogue_id=row.dialogue_id,
                condition=EvaluationCondition.ASR,
                history=row.history,
                user_transcript=row.user_text,
                gold_response=row.target_text,
                generated_response="That sounds difficult.",
                duration=row.duration,
            )
            for row in examples
        ),
        example_losses=tuple(
            ExampleLoss(
                example_id=row.example_id,
                dialogue_id=row.dialogue_id,
                condition=EvaluationCondition.ASR,
                cross_entropy=loss,
                target_tokens=tokens,
            )
            for row, loss, tokens in zip(examples, (1, 3, 5), (10, 1, 1), strict=True)
        ),
        diagnostics=(),
    )
    evaluation = directory / "evaluation"
    save_evaluation(outcome, evaluation)
    return FollowupReportConfig(
        selection=selection_path,
        sources=source_path,
        inputs=(
            BaselineComparisonInput(
                name="plain_asr", evaluation_directory=evaluation, condition=EvaluationCondition.ASR
            ),
        ),
        output_directory=directory / "report",
    )


def test_report_keeps_target_weighted_pooled_ce_and_cohort_macro_distinct(tmp_path: Path) -> None:
    configuration = prepared_comparison(tmp_path)
    report = write_comparison(configuration)
    measurement = report.measurements[0]
    assert measurement.pooled.cross_entropy == 1.5
    assert measurement.cohorts.macro_cross_entropy == 3
    assert measurement.cohorts.old_ordinary.target_tokens == 10
    assert measurement.cohorts.old_ordinary.completed_generations is None
    assert (configuration.output_directory / "heldout_comparison.png").is_file()
    assert (configuration.output_directory / "cohort_metrics.csv").is_file()


def test_report_rejects_same_ids_with_different_target_replies(tmp_path: Path) -> None:
    configuration = prepared_comparison(tmp_path)
    path = configuration.inputs[0].evaluation_directory / "evaluation_generations.jsonl"
    samples = tuple(
        SampleGeneration.model_validate_json(line) for line in path.read_text().splitlines()
    )
    path.write_text(
        "".join(
            row.model_copy(update={"gold_response": "Different target"}).model_dump_json() + "\n"
            for row in samples
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different reference"):
        measured_condition(configuration.inputs[0], configuration)


def test_report_rejects_pooled_metrics_unbound_to_actual_losses(tmp_path: Path) -> None:
    configuration = prepared_comparison(tmp_path)
    path = configuration.inputs[0].evaluation_directory / "evaluation.json"
    metrics = EvaluationMetrics.model_validate_json(path.read_bytes())
    path.write_text(
        metrics.model_copy(update={"cross_entropy": 0.1}).model_dump_json(), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="primary loss records"):
        measured_condition(configuration.inputs[0], configuration)
