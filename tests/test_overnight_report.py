"""Reporting must preserve measured coverage and reject inconsistent selection records."""

from pathlib import Path

import pytest

from scripts.package_results import FileArtifact
from speech_projector.data import distribution
from speech_projector.judge import BootstrapInterval, JudgeConfig, JudgeSummary
from speech_projector.models import EvaluationCondition, RunResult, SampleGeneration
from speech_projector.overnight_conversation_execution import (
    ConversationEvaluationProvenance,
    ConversationEvaluationSummary,
)
from speech_projector.overnight_data import Cohort, NeuEmotionalExampleSource
from speech_projector.overnight_evaluation import SweepSelectionPolicy, select_sweep
from speech_projector.overnight_judge import (
    FinalJudgingSummary,
    JudgedTonePreference,
    ToneCalibrationResult,
    ToneJudgeSummary,
)
from speech_projector.overnight_preparation import (
    CohortCoverage,
    CombinedPreparation,
    NeuSourceConfig,
    OvernightPreparationConfig,
    QwenSourceConfig,
)
from speech_projector.overnight_report import (
    render_baselines,
    render_conversation_measurements,
    render_final_judging,
    render_generations,
    render_overnight_report,
)
from speech_projector.tts_pilot import PilotEmotion
from tests.test_overnight_evaluation import candidate


def preparation() -> CombinedPreparation:
    artifact = FileArtifact(
        path=Path("manifest"), source_path=Path("manifest"), bytes=1, sha256="a"
    )
    configuration = OvernightPreparationConfig(
        ordinary_manifest=Path("ordinary"),
        ordinary_asr_manifest=Path("ordinary_asr"),
        qwen=QwenSourceConfig(
            targets=Path("qwen_targets"),
            generation_configuration=Path("qwen_config"),
            audio_directory=Path("qwen_audio"),
        ),
        neu=NeuSourceConfig(
            targets=Path("neu_targets"),
            generation_configuration=Path("neu_config"),
            audio_directory=Path("neu_audio"),
            utterances=Path("neu_utterances"),
        ),
        output_directory=Path("combined"),
    )
    return CombinedPreparation(
        configuration=configuration,
        source_commit="source",
        source_artifacts=(artifact,),
        manifest=artifact,
        sidecar=artifact,
        audio_inventory=artifact,
        fixed_validation=artifact,
        fixed_test=artifact,
        coverage=(
            CohortCoverage(
                cohort=Cohort.ORDINARY,
                split="train",
                examples=20000,
                duration_seconds=distribution((2.0, 4.0)),
            ),
        ),
        exclusions=(),
        cached_features_reused=1,
        missing_features=2,
        ordinary_asr_reused=1,
        generation_example_ids=("a",),
        boundary_checks=("Passed source family split check",),
    )


def result() -> RunResult:
    selected = candidate("quality", 1, 1, 0.1, 10)
    return RunResult(
        config=selected.configuration,
        git_commit="run_source",
        train_examples=40000,
        validation_examples=128,
        test_examples=256,
        projector_parameters=12,
        pseudo_tokens_per_second=10,
        mean_pseudo_tokens=20,
        steps=2000,
        runtime_seconds=1000,
        peak_vram_gb=4,
        examples_per_second=8,
        target_tokens_per_second=80,
        initial_validation_loss=2,
        initial_training_loss=2,
        final_fixed_training_loss=1,
        final_training_loss=1,
        validation=selected.validation.old_ordinary,
        checkpoint_path=Path("projector.safetensors"),
    )


def test_report_rejects_inconsistent_results_and_preserves_unmeasured_test() -> None:
    selected = candidate("quality", 1, 1, 0.1, 10)
    decision = select_sweep((selected,), SweepSelectionPolicy())
    recorded = result()
    text = render_overnight_report(preparation(), decision, (recorded,), (recorded,), ())
    assert "Not evaluated" in text
    assert "1000.0 | 0.500 | 4.000" in text
    assert "PyTorch allocated decimal GB" in text
    assert "linearly resized" in text
    with pytest.raises(ValueError, match="coverage"):
        render_overnight_report(preparation(), decision, (), (), ())
    changed = recorded.model_copy(update={"config": recorded.config.model_copy(update={"seed": 7})})
    with pytest.raises(ValueError, match="configurations differ"):
        render_overnight_report(preparation(), decision, (changed,), (), ())


def test_qualitative_output_exposes_intended_tone_without_inventing_completion() -> None:
    source = NeuEmotionalExampleSource(
        example_id="a",
        source_manifest=Path("neu"),
        source_example_id="a",
        base_id="base",
        family_id="family",
        emotion=PilotEmotion.ANGRY,
    )
    sample = SampleGeneration(
        example_id="a",
        dialogue_id="family",
        condition=EvaluationCondition.SPEECH,
        history=(),
        user_transcript="The deadline moved again.",
        gold_response="That sounds frustrating.",
        generated_response="I hear that this is frustrating.",
        duration=2,
    )
    text = render_generations((sample,), (source,))
    assert "angry; not verified perception" in text
    assert "Generation termination: unknown" in text
    assert sample.generated_response in text
    assert sample.gold_response in text


def test_baseline_reporting_preserves_measured_coverage_and_unknown_caps(tmp_path: Path) -> None:
    path = tmp_path / "baseline" / "test" / "asr" / "evaluation.json"
    path.parent.mkdir(parents=True)
    metrics = result().validation.model_copy(
        update={"examples": 256, "generated_examples": 136, "semantic_similarity": 0.25}
    )
    path.write_text(metrics.model_dump_json(), encoding="utf-8")
    text = render_baselines(tmp_path)
    assert "test | asr | 256 |" in text
    assert "0.2500 | 136 | unknown / unknown" in text
    assert "test | text | Not measured" in text
    assert "not information-complete emotional upper bounds" in text


def test_failed_or_missing_tone_calibration_suppresses_all_quality_rates(tmp_path: Path) -> None:
    malformed_summary = tmp_path / "judge" / "speech" / "summary.json"
    malformed_summary.parent.mkdir(parents=True)
    malformed_summary.write_text("invalid", encoding="utf-8")
    assert "Quality rates are not presented" in render_final_judging(tmp_path, None)
    failed = ToneCalibrationResult(
        configuration=JudgeConfig(),
        cases_sha256="a",
        requested=12,
        valid=12,
        correct_acceptability=11,
        preference_requests=6,
        preference_correct=5,
        passed=False,
        runtime_seconds=3,
        peak_pytorch_allocated_decimal_gb=4,
    )
    text = render_final_judging(tmp_path, failed)
    assert "FAILED" in text and "11/12" in text and "5/6" in text
    assert "Quality rates are suppressed" in text
    passed = failed.model_copy(
        update={"passed": True, "correct_acceptability": 12, "preference_correct": 6}
    )
    with pytest.raises(ValueError):
        render_final_judging(tmp_path, passed)


def test_conversation_report_records_caps_and_the_actual_retention_budget(tmp_path: Path) -> None:
    directory = tmp_path / "conversation"
    directory.mkdir()
    summary = ConversationEvaluationSummary(
        scenarios=3,
        controlled_responses=36,
        rollouts=6,
        rollout_responses=24,
        completed_responses=58,
        token_limited_responses=2,
        generation_seconds=19,
    )
    (directory / "summary.json").write_text(summary.model_dump_json(), encoding="utf-8")
    provenance = ConversationEvaluationProvenance(
        source_commit="actual_source",
        configuration=result().config.model_copy(
            update={"history_turns": 6, "max_history_tokens": 2048}
        ),
        projector_weights_sha256="weights",
        fixtures_sha256="fixtures",
    )
    (directory / "provenance.json").write_text(provenance.model_dump_json(), encoding="utf-8")
    text = render_conversation_measurements(tmp_path)
    assert "58 completed and 2 capped" in text
    assert "History budget 6 turns / 2048 tokens" in text
    assert "actual_source" in text and "weights" in text and "fixtures" in text
    assert "can carry the initial cue in generated text" in text


def test_final_judging_report_requires_path_condition_parity_and_adds_only_recorded_clocks(
    tmp_path: Path,
) -> None:
    calibration = ToneCalibrationResult(
        configuration=JudgeConfig(),
        cases_sha256="a",
        requested=12,
        valid=12,
        correct_acceptability=12,
        preference_requests=6,
        preference_correct=6,
        passed=True,
        runtime_seconds=100,
        peak_pytorch_allocated_decimal_gb=4,
    )
    interval = BootstrapInterval(
        examples=4,
        dialogues=2,
        draws=2000,
        confidence=0.95,
        estimate=0.5,
        lower=0.25,
        upper=0.75,
    )
    ordinary = JudgeSummary(
        requested_examples=2,
        valid_examples=1,
        failed_examples=1,
        acceptable_rate_requested=0.5,
        acceptable_rate_valid=1,
        relevance=3,
        grounded_detail=3,
        naturalness=3,
    )
    tone = ToneJudgeSummary(
        requested=4,
        valid=3,
        failed=1,
        acceptable_rate_requested=0.5,
        acceptable_rate_valid=2 / 3,
        relevance=3,
        grounded_detail=3,
        naturalness=3,
        tone_appropriateness=2,
        acceptable_interval=interval,
    )
    pair = JudgedTonePreference(
        requested=4,
        valid=3,
        failed=1,
        matching_wins=2,
        ties=1,
        matching_losses=0,
        identical_responses=1,
        matching_win_rate=interval,
    )
    summary = FinalJudgingSummary(
        condition=EvaluationCondition.SPEECH,
        ordinary=ordinary,
        old_emotional=tone,
        new_emotional=tone,
        old_preferences=pair,
        new_preferences=pair,
        main_judging_seconds=3,
        paired_judging_seconds=5,
    )
    path = tmp_path / "judge" / "speech" / "summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(summary.model_dump_json(), encoding="utf-8")
    text = render_final_judging(tmp_path, calibration)
    assert "ordinary | 2 / 1 / 1 | 50.0%" in text
    assert "Neu emotional | 4 / 3 / 1 | 50.0%" in text
    assert "speech | 3.0 | 5.0 | 8.0" in text
    path.write_text(
        summary.model_copy(update={"condition": EvaluationCondition.TEXT}).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="condition differs"):
        render_final_judging(tmp_path, calibration)
