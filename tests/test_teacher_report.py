from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from scripts.challenge_judge import JudgeChallengeReport
from scripts.judge_teacher_suite import (
    GenerationCapSummary,
    JudgedGenerationSet,
    JudgeSelection,
    JudgmentInputs,
    TeacherJudgingConfig,
    TeacherJudgingReport,
)
from scripts.package_results import FileArtifact, ModelRevision
from scripts.prepare_teacher_data import TeacherDataConfig, TeacherDataPreparation
from scripts.run_judge import CalibrationSummary
from scripts.summarize_teacher_targets import (
    HistoryLengths,
    ResponseLengths,
    TeacherTargetAudit,
)
from speech_projector.cache import CacheStatistics
from speech_projector.data import distribution
from speech_projector.generation import TokenLimitedGeneration
from speech_projector.judge import JudgeConfig, JudgeJournalProvenance, JudgeSummary
from speech_projector.models import (
    EvaluationCondition,
    EvaluationMetrics,
    Example,
    RunConfig,
    RunResult,
    SampleGeneration,
    Split,
    SuiteState,
)
from speech_projector.teacher import (
    TeacherConfig,
    TeacherFailure,
    TeacherFailureReason,
    TeacherProgress,
    TeacherProvenance,
)
from speech_projector.teacher_configuration import (
    teacher_compression_runs,
    teacher_feasibility_run,
    teacher_linear_run,
    teacher_scaling_runs,
)
from speech_projector.teacher_evaluation import FidelitySummary
from speech_projector.teacher_report import (
    RunAnalysis,
    TeacherReportConfig,
    TeacherReportData,
    TeacherReportProvenance,
    best_run,
    judged_set,
    load_teacher_report,
    render_qualitative,
    render_teacher_report,
    save_plots,
    validate_program,
)


def artifact(path: Path) -> FileArtifact:
    return FileArtifact(path=path, source_path=path, bytes=0, sha256="test-local")


def result(config: RunConfig) -> RunResult:
    metrics = EvaluationMetrics(
        examples=config.validation_examples,
        target_tokens=1000,
        cross_entropy=1.0,
        perplexity=2.71828,
        generated_examples=config.semantic_examples,
    )
    return RunResult(
        config=config,
        git_commit="test-source",
        train_examples=config.train_examples,
        validation_examples=config.validation_examples,
        test_examples=config.test_examples,
        projector_parameters=1234,
        pseudo_tokens_per_second=50 / config.projector.compression_factor,
        mean_pseudo_tokens=40,
        steps=2,
        runtime_seconds=3600,
        peak_vram_gb=8,
        examples_per_second=6,
        target_tokens_per_second=1000,
        initial_validation_loss=2,
        initial_training_loss=2,
        final_fixed_training_loss=0.5,
        final_training_loss=0.5,
        validation=metrics,
        test=metrics,
        checkpoint_path=Path("checkpoint/projector.safetensors"),
    )


def complete_results() -> tuple[RunResult, ...]:
    configs = (
        teacher_feasibility_run(),
        *teacher_compression_runs(),
        teacher_linear_run(5),
        *teacher_scaling_runs(),
    )
    return tuple(result(config) for config in configs)


def fidelity(examples: int) -> FidelitySummary:
    return FidelitySummary(
        condition=EvaluationCondition.SPEECH,
        examples=examples,
        target_tokens=1000,
        input_cross_entropy=1,
        transcript_cross_entropy=0.4,
        excess_cross_entropy=0.6,
        top1_agreement=0.8,
        input_target_accuracy=0.75,
        transcript_target_accuracy=0.95,
        teacher_to_input_kl=0.3,
        first_token_agreement=0.5,
        first_token_target_accuracy=0.4,
        first_8_tokens=examples * 8,
        first_8_token_agreement=0.6,
        first_8_token_target_accuracy=0.55,
    )


def judged(name: str, split: Split, count: int, judge: JudgeConfig) -> JudgedGenerationSet:
    return JudgedGenerationSet(
        name=name,
        split=split,
        inputs=JudgmentInputs(
            generations=artifact(Path("generations.jsonl")),
            judge=JudgeJournalProvenance(config=judge, rubric_sha256="test-rubric"),
        ),
        journal=artifact(Path("judge.jsonl")),
        summary=JudgeSummary(
            requested_examples=count,
            valid_examples=count,
            failed_examples=0,
            acceptable_rate_valid=0.75,
            acceptable_rate_requested=0.75,
            relevance=2,
            grounded_detail=2,
            naturalness=2,
        ),
        generation_caps=GenerationCapSummary(completed=count, capped=0, unknown=0),
        seconds=1,
    )


def challenge(judge: JudgeConfig) -> JudgeChallengeReport:
    summary = CalibrationSummary(
        config=judge,
        rubric_sha256="test-rubric",
        cases=(),
        groups=(),
        passed=True,
        seconds=1,
        decoded_response_tokens=20,
    )
    return JudgeChallengeReport(calibration=summary, independent_challenge=summary)


@pytest.fixture
def data(tmp_path: Path) -> TeacherReportData:
    config = TeacherReportConfig(results_root=tmp_path, data_root=tmp_path, output=tmp_path)
    results = complete_results()
    judge = JudgeConfig()
    sets = tuple(
        judged(run.config.name, split, run.config.semantic_examples, judge)
        for run in results
        for split in (Split.VALIDATION, Split.TEST)
    ) + tuple(
        judged(name, split, 512, judge)
        for name in ("text", "asr")
        for split in (Split.VALIDATION, Split.TEST)
    )
    quality = TeacherJudgingReport(
        config=TeacherJudgingConfig(
            results_root=tmp_path,
            output_directory=tmp_path,
            calibration_root=tmp_path,
            candidates=(judge,),
        ),
        selection=JudgeSelection(
            selected=judge,
            attempts=(challenge(judge),),
        ),
        sets=sets,
        comparisons=(),
        seconds=5,
        peak_vram_gb=4,
    )
    runs = tuple(
        RunAnalysis(
            run,
            split,
            (fidelity(run.validation_examples),),
            judged_set(quality, run.config.name, split),
            (),
        )
        for run in results
        for split in (Split.VALIDATION, Split.TEST)
    )
    examples = tuple(
        Example(
            example_id=f"{split.value}-{index}",
            dialogue_id=f"dialogue-{split.value}-{index}",
            split=split,
            history=(),
            user_text="True user input",
            target_text="Native teacher answer",
            audio_path=Path("audio.wav"),
            feature_path=Path("feature.pt"),
            duration=1,
            domain="science",
            emotion="neutral",
        )
        for split in (Split.VALIDATION, Split.TEST)
        for index in range(9)
    )
    samples = tuple(
        (
            name,
            split,
            tuple(
                SampleGeneration(
                    example_id=example.example_id,
                    dialogue_id=example.dialogue_id,
                    condition=condition,
                    history=(),
                    user_transcript=example.user_text,
                    gold_response=example.target_text,
                    generated_response=f"{name} answer {example.example_id}",
                    duration=1,
                )
                for example in examples
                if example.split == split
            ),
        )
        for name, condition in (
            ("text", EvaluationCondition.TEXT),
            ("asr", EvaluationCondition.ASR),
            *((run.config.name, EvaluationCondition.SPEECH) for run in results),
        )
        for split in (Split.VALIDATION, Split.TEST)
    )
    run_config = results[1].config
    provenance = TeacherProvenance(
        config=TeacherConfig(
            run=run_config, manifest=tmp_path / "source.jsonl", output_directory=tmp_path
        ),
        source_commit="test-source",
        input_manifest=artifact(tmp_path / "source.jsonl"),
        model_revision=ModelRevision(
            model_name=run_config.model_name, snapshot_revisions=("test",), main_revision="test"
        ),
        frozen_parameter_sha256="test-weights",
        started_at=datetime(2026, 10, 6, tzinfo=timezone.utc),
    )
    lengths = ResponseLengths(
        words=distribution([100.0]), target_tokens_with_eos=distribution([200.0])
    )
    audit = TeacherTargetAudit(
        provenance=provenance,
        input_journal=artifact(tmp_path / "targets.jsonl"),
        source_manifest=artifact(tmp_path / "source.jsonl"),
        source_examples=21024,
        completed_examples=21024,
        incomplete_suffix_bytes=0,
        splits=(),
        domains=(),
        original_dataset_response=lengths,
        teacher_response=lengths,
        generated_tokens_with_eos=distribution([200.0]),
        history=HistoryLengths(
            selected_turns=distribution([2.0]),
            selected_text_tokens=distribution([50.0]),
            current_user_tokens=distribution([25.0]),
            empty_history_examples=0,
        ),
        history_groups=(),
        completed_without_retry=21024,
        completed_after_retry=0,
        eos_completed_examples=21024,
        issues=(),
        failures=(),
        progress=(
            TeacherProgress(
                completed_examples=21024,
                generated_tokens=4204800,
                elapsed_seconds=10000,
                examples_per_second=2.1024,
                tokens_per_second=420.48,
                peak_vram_gb=8,
            ),
        ),
        fixed_samples=(),
        token_count_definition="test",
        history_count_definition="test",
    )
    preparation = TeacherDataPreparation(
        configuration=TeacherDataConfig(original_root=tmp_path, output_root=tmp_path),
        original_manifest_sha256="test-original",
        metadata_sha256="test-metadata",
        source_manifest_sha256="test-local",
        provenance_sha256="test",
        splits=(),
        subset_definition="Clean, nested teacher inputs.",
        transcript_reference="Actual cleaned synthesis text.",
        target_status="Teacher targets complete",
    )
    cache = CacheStatistics(
        encoder_model="openai/whisper-small",
        hidden_dimension=768,
        native_states_per_second=50,
        selected_hidden_state="last",
        storage_dtype="bfloat16",
        bytes_per_audio_second=76800,
        feature_count=21024,
        extracted_count=1000,
        total_audio_seconds=100000,
        extracted_audio_seconds=5000,
        feature_bytes=7680000000,
        extraction_seconds=20,
        cache_wall_seconds=40,
        cache_examples_per_second=25,
        extraction_audio_seconds_per_second=250,
        asr_seconds=20,
        peak_vram_gb=2,
        masking="valid frames only",
    )
    return TeacherReportData(
        config,
        runs,
        audit,
        preparation,
        cache,
        quality,
        examples,
        tuple(
            item.model_copy(update={"target_text": "Original audit answer"}) for item in examples
        ),
        samples,
        tuple(
            (
                name,
                split,
                EvaluationMetrics(
                    examples=512, target_tokens=1000, cross_entropy=0.4, perplexity=1.5
                ),
            )
            for name in ("text", "asr")
            for split in (Split.VALIDATION, Split.TEST)
        ),
        (),
        TeacherReportProvenance(configuration=config, inputs=()),
        (),
        (),
    )


def test_report_discloses_observed_pilot_failures_and_judge_attempts(
    data: TeacherReportData,
) -> None:
    failure = TeacherFailure(
        example=data.source_examples[0].model_copy(
            update={"user_text": "What is on your playlist?"}
        ),
        reason=TeacherFailureReason.TOKEN_LIMIT,
        attempts=(
            TokenLimitedGeneration(partial_text="Repeated playlist", token_ids=(1,) * 2048),
            TokenLimitedGeneration(partial_text="Repeated playlist", token_ids=(1,) * 4096),
        ),
    )
    failed_challenge = data.quality.selection.attempts[0].model_copy(
        update={
            "independent_challenge": data.quality.selection.attempts[
                0
            ].independent_challenge.model_copy(update={"passed": False})
        }
    )
    quality = data.quality.model_copy(
        update={
            "selection": data.quality.selection.model_copy(
                update={"attempts": (failed_challenge, *data.quality.selection.attempts)}
            )
        }
    )
    report = render_teacher_report(
        replace(
            data,
            quality=quality,
            greedy_pilot_failures=(failure,),
            greedy_pilot_results=(data.runs[0].result,),
        )
    )
    assert "1 raw records" in report
    assert "2048 tokens" in report and "4096 tokens" in report
    assert "What is on your playlist?" in report
    assert "pilot_greedy/teacher_targets/failures.jsonl" in report
    assert "independent passed=False" in report
    assert "original passed=True" in report
    assert "3600.0 seconds" in report
    assert "excluded from the sampled program" in report


def test_complete_program_and_missing_stage_validation() -> None:
    results = complete_results()
    validate_program(results)
    for excluded in (
        "teacher_3000_mlp_10hz",
        "teacher_20000_mlp_2.5hz",
        "teacher_20000_linear_5x",
        "teacher_v0_256_mlp_10hz",
    ):
        with pytest.raises(ValueError):
            validate_program(tuple(item for item in results if item.config.name != excluded))


def test_report_exposes_fidelity_quality_baselines_and_real_denominators(
    data: TeacherReportData,
) -> None:
    report = render_teacher_report(data)
    assert "Teacher top1%" in report
    assert "Appropriate%" in report
    assert "512/512" in report
    assert "first-token/first-eight" in report
    assert "data scaling changes update count" in report
    assert "by construction" in report
    assert "candidate-only" in report


def test_fixed_paired_set_uses_exact_first_eight_per_split(data: TeacherReportData) -> None:
    report = render_qualitative(data)
    assert report.count("## ") == 16
    assert "validation-0" in report and "test-7" in report
    assert "validation-8" not in report and "test-8" not in report
    assert "Original audit answer" in report
    assert "True user input" in report
    assert "unknown completion" in report
    assert "asr answer" in report


def test_best_configuration_is_selected_without_test_scores(data: TeacherReportData) -> None:
    name = "teacher_20000_mlp_5hz"
    changed = tuple(
        replace(
            item,
            judged=item.judged.model_copy(
                update={
                    "summary": item.judged.summary.model_copy(
                        update={
                            "acceptable_rate_requested": 1.0
                            if item.result.config.name == name
                            else 0.5
                        }
                    )
                }
            ),
        )
        if item.split == Split.VALIDATION
        else item
        for item in data.runs
    )
    assert best_run(replace(data, runs=changed)).result.config.name == name


def test_missing_fixed_response_is_an_explicit_failure(data: TeacherReportData) -> None:
    missing = replace(data, samples=tuple(item for item in data.samples if item[0] != "asr"))
    with pytest.raises(ValueError, match="sample missing"):
        render_qualitative(missing)


def test_scientific_curves_are_exported_as_standalone_pngs(data: TeacherReportData) -> None:
    save_plots(data)
    for name in (
        "teacher_data_scaling.png",
        "teacher_compression.png",
        "teacher_architectures.png",
    ):
        content = (data.configuration.output / name).read_bytes()
        assert content[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(content) > 10000


def test_completed_results_without_calibrated_quality_report_fail_clearly(tmp_path: Path) -> None:
    results = complete_results()
    (tmp_path / "suite_state.json").write_text(
        SuiteState(
            completed=tuple(item.config.name for item in results),
            running=None,
            failed=(),
            started_at=0,
            updated_at=1,
        ).model_dump_json()
    )
    (tmp_path / "completed_results.json").write_bytes(
        TypeAdapter(tuple[RunResult, ...]).dump_json(results)
    )
    with pytest.raises(ValueError, match="quality_report.json"):
        load_teacher_report(
            TeacherReportConfig(results_root=tmp_path, data_root=tmp_path, output=tmp_path)
        )


@pytest.mark.parametrize(
    "invalid_gate", ("calibration_config", "challenge_config", "challenge_failed")
)
def test_report_rejects_unmatched_or_failed_selected_judge(
    data: TeacherReportData, invalid_gate: str
) -> None:
    results = complete_results()
    root = data.configuration.results_root
    (root / "suite_state.json").write_text(
        SuiteState(
            completed=tuple(item.config.name for item in results),
            running=None,
            failed=(),
            started_at=0,
            updated_at=1,
        ).model_dump_json()
    )
    (root / "completed_results.json").write_bytes(
        TypeAdapter(tuple[RunResult, ...]).dump_json(results)
    )
    attempt = data.quality.selection.attempts[0]
    other_config = data.quality.selection.selected.model_copy(update={"batch_size": 1})
    match invalid_gate:
        case "calibration_config":
            attempt = attempt.model_copy(
                update={
                    "calibration": attempt.calibration.model_copy(update={"config": other_config})
                }
            )
        case "challenge_config":
            attempt = attempt.model_copy(
                update={
                    "independent_challenge": attempt.independent_challenge.model_copy(
                        update={"config": other_config}
                    )
                }
            )
        case "challenge_failed":
            attempt = attempt.model_copy(
                update={
                    "independent_challenge": attempt.independent_challenge.model_copy(
                        update={"passed": False}
                    )
                }
            )
    quality = data.quality.model_copy(
        update={"selection": data.quality.selection.model_copy(update={"attempts": (attempt,)})}
    )
    quality_path = root / "response_quality/quality_report.json"
    quality_path.parent.mkdir()
    quality_path.write_text(quality.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="selected judge"):
        load_teacher_report(data.configuration)
