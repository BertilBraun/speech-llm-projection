import hashlib
from pathlib import Path

import pytest

from scripts.analyze_teacher_conditioning import (
    EstimatedMargin,
    FidelityMetric,
    HistoryStratum,
    InsufficientDialogues,
    SerializedExample,
    SplitDiagnostics,
    TeacherConditioningReport,
    analyze_split,
    analyze_suite,
    fidelity_margin,
    interval,
    loss_margin,
    read_records,
    render_report,
    summarize_stratum,
    validate_records,
)
from scripts.package_results import FileArtifact
from scripts.summarize_teacher_targets import HistoryGroup
from speech_projector.evaluation import ExampleLoss
from speech_projector.models import (
    EvaluationCondition,
    Example,
    ExperimentStage,
    MlpProjectorConfig,
    Role,
    RunConfig,
    RunResult,
    Split,
    SuiteState,
    Turn,
)
from speech_projector.teacher_evaluation import FidelityProvenance, TeacherFidelity


def examples() -> tuple[Example, ...]:
    return tuple(
        Example(
            example_id=f"example-{index}",
            dialogue_id=f"dialogue-{index}",
            split=Split.VALIDATION,
            history=() if index < 2 else (Turn(role=Role.USER, text="Earlier context"),),
            user_text="Where is Rome?",
            target_text="In Italy.",
            audio_path=Path(f"audio-{index}.wav"),
            feature_path=Path(f"audio-{index}.pt"),
            duration=1,
            domain="travel",
            emotion="neutral",
        )
        for index in range(4)
    )


def losses() -> tuple[ExampleLoss, ...]:
    return tuple(
        ExampleLoss(
            example_id=example.example_id,
            dialogue_id=example.dialogue_id,
            condition=condition,
            cross_entropy=1 if condition == EvaluationCondition.SPEECH else 2 + 2 * (index % 2),
            target_tokens=2 if index % 2 == 0 else 8,
        )
        for condition in (EvaluationCondition.SPEECH, EvaluationCondition.SHUFFLED_SPEECH)
        for index, example in enumerate(examples())
    )


def fidelity() -> tuple[TeacherFidelity, ...]:
    return tuple(
        TeacherFidelity(
            example_id=item.example_id,
            dialogue_id=item.dialogue_id,
            condition=item.condition,
            target_tokens=item.target_tokens,
            input_cross_entropy=item.cross_entropy,
            transcript_cross_entropy=0.5,
            excess_cross_entropy=item.cross_entropy - 0.5,
            top1_agreement=0.8 if item.condition == EvaluationCondition.SPEECH else 0.3,
            input_target_accuracy=0.7 if item.condition == EvaluationCondition.SPEECH else 0.2,
            transcript_target_accuracy=0.75,
            teacher_to_input_kl=0.5 if item.condition == EvaluationCondition.SPEECH else 1.5,
            first_token_agreement=1 if item.condition == EvaluationCondition.SPEECH else 0,
            first_token_target_accuracy=1 if item.condition == EvaluationCondition.SPEECH else 0,
            first_8_tokens=min(8, item.target_tokens),
            first_8_token_agreement=0.9 if item.condition == EvaluationCondition.SPEECH else 0.4,
            first_8_token_target_accuracy=0.6
            if item.condition == EvaluationCondition.SPEECH
            else 0.1,
        )
        for item in losses()
    )


def test_ce_weighting_and_partial_control_alignment_are_explicit() -> None:
    records = losses()[:4] + losses()[4:6]
    result = loss_margin(records, EvaluationCondition.SPEECH, EvaluationCondition.SHUFFLED_SPEECH)
    assert result.diagnostic.paired_examples == 2
    assert result.diagnostic.mean_control_minus_correct_ce == 2
    assert isinstance(result.example_weighted, EstimatedMargin)
    assert isinstance(result.token_weighted, EstimatedMargin)
    assert result.example_weighted.interval.estimate == 2
    assert result.token_weighted.interval.estimate == pytest.approx(2.6)
    assert result.token_weighted.interval.dialogues == 2


@pytest.mark.parametrize("metric", tuple(FidelityMetric))
def test_fidelity_control_orientation_favors_correct_audio(metric: FidelityMetric) -> None:
    result = fidelity_margin(
        fidelity(), EvaluationCondition.SPEECH, EvaluationCondition.SHUFFLED_SPEECH, metric
    )
    assert isinstance(result.estimate, EstimatedMargin)
    assert result.estimate.interval.estimate > 0
    assert result.estimate.interval.examples == 4


def test_natural_history_strata_do_not_remove_or_reassign_context() -> None:
    empty = summarize_stratum(examples()[:2], losses(), fidelity())
    present = summarize_stratum(examples()[2:], losses(), fidelity())
    assert empty.examples == present.examples == 2
    assert all(item.examples == 2 for item in empty.fidelity + present.fidelity)
    assert all(item.metrics.target_tokens == 10 for item in empty.losses + present.losses)
    assert isinstance(interval(()), InsufficientDialogues)


def test_report_uses_condition_counts_for_partial_controls_in_each_history_stratum() -> None:
    selected_losses = losses()[:4] + (losses()[4], losses()[6])
    selected_fidelity = fidelity()[:4] + (fidelity()[4], fidelity()[6])
    overall = summarize_stratum(examples(), selected_losses, selected_fidelity)
    config = RunConfig(
        name="partial-controls",
        stage=ExperimentStage.V0,
        train_examples=4,
        validation_examples=4,
        test_examples=4,
        conditioning_examples=2,
        epochs=1,
        learning_rate=0.001,
        projector=MlpProjectorConfig(compression_factor=5),
    )
    run = RunResult(
        config=config,
        git_commit="test-source",
        train_examples=4,
        validation_examples=4,
        test_examples=4,
        projector_parameters=100,
        pseudo_tokens_per_second=10,
        mean_pseudo_tokens=10,
        steps=1,
        runtime_seconds=1,
        peak_vram_gb=1,
        examples_per_second=4,
        target_tokens_per_second=20,
        initial_validation_loss=2,
        initial_training_loss=2,
        final_fixed_training_loss=1,
        final_training_loss=1,
        validation=overall.losses[0].metrics,
        checkpoint_path=Path("projector.safetensors"),
    )
    split = SplitDiagnostics(
        split=Split.VALIDATION,
        provenance=FidelityProvenance(
            config=config, projector_weights_sha256="test-weights", examples_sha256="test-examples"
        ),
        inputs=(),
        overall=overall,
        history=tuple(
            HistoryStratum(
                history=group,
                diagnostics=summarize_stratum(selected, selected_losses, selected_fidelity),
            )
            for group, selected in (
                (HistoryGroup.EMPTY, examples()[:2]),
                (HistoryGroup.PRESENT, examples()[2:]),
            )
        ),
    )
    report = render_report(
        TeacherConditioningReport(
            run=run,
            manifest=FileArtifact(
                path=Path("manifest"), source_path=Path("manifest"), bytes=0, sha256="test-manifest"
            ),
            splits=(split,),
        )
    )
    level_table = report.split("| Split | Natural history | Correct / control |")[0]
    rows = tuple(line for line in level_table.splitlines() if line.startswith("| validation |"))
    assert len(rows) == 6
    for history, primary_count, control_count in (
        ("all", 4, 2),
        ("empty", 2, 1),
        ("present", 2, 1),
    ):
        assert any(
            line.startswith(f"| validation | {history} | {primary_count} | speech |")
            for line in rows
        )
        assert any(
            line.startswith(f"| validation | {history} | {control_count} | shuffled_speech |")
            for line in rows
        )
    assert overall.loss_margins[0].diagnostic.paired_examples == 2


def test_read_only_journal_does_not_repair_incomplete_suffix(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    content = losses()[0].model_dump_json().encode() + b"\n{unfinished"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="completed journal"):
        read_records(path, ExampleLoss)
    assert path.read_bytes() == content


def test_duplicate_or_missing_primary_records_are_rejected() -> None:
    with pytest.raises(ValueError, match="repeat"):
        validate_records(examples(), losses() + losses()[:1])
    with pytest.raises(ValueError, match="entire selected split"):
        validate_records(examples(), losses()[1:])


def test_provenance_and_full_control_coverage_checked_before_analysis(tmp_path: Path) -> None:
    config = RunConfig(
        name="test",
        stage=ExperimentStage.V0,
        train_examples=4,
        validation_examples=4,
        test_examples=4,
        conditioning_examples=4,
        epochs=1,
        learning_rate=0.001,
        projector=MlpProjectorConfig(compression_factor=5),
    )
    serialized = tuple(
        SerializedExample(item, item.model_dump_json().encode()) for item in examples()
    )
    digest = hashlib.sha256(b"".join(item.line + b"\n" for item in serialized)).hexdigest()
    provenance = FidelityProvenance(
        config=config, projector_weights_sha256="weights", examples_sha256=digest
    )
    provenance_path = tmp_path / "teacher_fidelity_provenance.json"
    provenance_path.write_text(provenance.model_dump_json(), encoding="utf-8")
    (tmp_path / "evaluation_losses.jsonl").write_text(
        "".join(
            item.model_dump_json() + "\n"
            for item in losses()
            + tuple(
                item.model_copy(update={"condition": condition})
                for condition in (
                    EvaluationCondition.ZERO_SPEECH,
                    EvaluationCondition.SPEECH_NO_HISTORY,
                    EvaluationCondition.SHUFFLED_SPEECH_NO_HISTORY,
                )
                for item in losses()[4:]
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "teacher_fidelity.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in fidelity()), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="control journal is incomplete"):
        analyze_split(tmp_path, Split.VALIDATION, serialized)
    complete_fidelity = fidelity() + tuple(
        item.model_copy(update={"condition": EvaluationCondition.ZERO_SPEECH})
        for item in fidelity()[4:]
    )
    (tmp_path / "teacher_fidelity.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in complete_fidelity), encoding="utf-8"
    )
    result = analyze_split(tmp_path, Split.VALIDATION, serialized)
    assert result.overall.examples == 4
    assert len(result.history) == 2
    assert all(item.diagnostics.examples == 2 for item in result.history)
    provenance_path.write_text(
        provenance.model_copy(update={"examples_sha256": "wrong"}).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA differs"):
        analyze_split(tmp_path, Split.VALIDATION, serialized)


def test_suite_analysis_requires_completed_successful_writer_state(tmp_path: Path) -> None:
    state = SuiteState(completed=(), running="training", failed=(), started_at=1, updated_at=2)
    path = tmp_path / "suite_state.json"
    path.write_text(state.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="writers exit successfully"):
        analyze_suite(tmp_path, tmp_path / "manifest", tmp_path / "analysis")
    path.write_text(state.model_copy(update={"running": None}).model_dump_json(), encoding="utf-8")
    (tmp_path / "completed_results.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty"):
        analyze_suite(tmp_path, tmp_path / "manifest", tmp_path / "analysis")
