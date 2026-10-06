from collections.abc import Sequence
from pathlib import Path

import pytest

from scripts.challenge_judge import JudgeChallengeReport
from scripts.judge_teacher_suite import (
    CalibratedJudge,
    GenerationJob,
    JudgeSelection,
    TeacherJudgingConfig,
    cache_requests,
    control_generation_jobs,
    generation_caps,
    generation_jobs,
    judge_generation_set,
    run_control_judging,
    select_judge,
)
from scripts.run_judge import CalibrationSummary, calibration_cases
from speech_projector.journal import append_record
from speech_projector.judge import JudgeConfig, JudgeRequest, LocalJudge, rubric_digest
from speech_projector.models import (
    EvaluationCondition,
    GenerationDetails,
    GenerationKind,
    Role,
    SampleGeneration,
    Split,
    SuiteState,
    Turn,
)


class CountingJudge(LocalJudge):
    def __init__(self, config: JudgeConfig) -> None:
        self.config = config
        self.calls = 0

    def generate_batch(
        self, requests: Sequence[JudgeRequest], corrections: Sequence[str]
    ) -> tuple[str, ...]:
        self.calls += 1
        return tuple(
            '{"relevance":2,"grounded_detail":2,"naturalness":3,"evidence":"responsive"}'
            for _ in requests
        )


def configuration(directory: Path) -> TeacherJudgingConfig:
    return TeacherJudgingConfig(
        results_root=directory / "results",
        output_directory=directory / "judging",
        calibration_root=directory / "calibration",
        candidates=(
            JudgeConfig(),
            JudgeConfig(model_name="Qwen/Qwen3-4B", revision="pinned-fallback"),
        ),
    )


def calibration(judge: LocalJudge, passed: bool) -> CalibrationSummary:
    return CalibrationSummary(
        config=judge.config,
        rubric_sha256=rubric_digest(),
        cases=calibration_cases(),
        groups=(),
        passed=passed,
        seconds=1,
        decoded_response_tokens=100,
    )


def test_fallback_requires_its_own_passing_calibration(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    loaded: list[JudgeConfig] = []
    folders: list[Path] = []

    def load(candidate: JudgeConfig) -> LocalJudge:
        loaded.append(candidate)
        return CountingJudge(candidate)

    def calibrate_model(judge: LocalJudge, directory: Path) -> JudgeChallengeReport:
        folders.append(directory)
        summary = calibration(judge, judge.config == config.candidates[1])
        return JudgeChallengeReport(calibration=summary, independent_challenge=summary)

    selected = select_judge(config, load, calibrate_model)
    assert loaded == list(config.candidates)
    assert len(set(folders)) == 2
    assert selected.selection.selected == config.candidates[1]
    assert tuple(item.passed for item in selected.selection.attempts) == (False, True)
    assert (config.output_directory / "judge_selection.json").is_file()


def test_all_failed_calibrations_prohibit_quality_scoring(tmp_path: Path) -> None:
    config = configuration(tmp_path)

    def fail(judge: LocalJudge, directory: Path) -> JudgeChallengeReport:
        summary = calibration(judge, False)
        return JudgeChallengeReport(calibration=summary, independent_challenge=summary)

    with pytest.raises(ValueError, match="All candidate-only judges failed"):
        select_judge(config, CountingJudge, fail)
    assert (config.output_directory / "failed_calibrations.json").is_file()
    assert not (config.output_directory / "judge_selection.json").exists()


def samples() -> tuple[SampleGeneration, ...]:
    primary = SampleGeneration(
        example_id="complete",
        dialogue_id="dialogue-1",
        condition=EvaluationCondition.SPEECH,
        history=(),
        user_transcript="Hello",
        gold_response="Teacher response is audit-only",
        generated_response="Hi",
        generation=GenerationDetails(kind=GenerationKind.COMPLETED, token_ids=(1, 2)),
        duration=1,
    )
    return (
        primary,
        primary.model_copy(
            update={
                "example_id": "capped",
                "dialogue_id": "dialogue-2",
                "generation": GenerationDetails(kind=GenerationKind.TOKEN_LIMIT, token_ids=(1, 3)),
            }
        ),
        primary.model_copy(update={"example_id": "historical", "generation": None}),
        primary.model_copy(update={"condition": EvaluationCondition.SHUFFLED_SPEECH}),
    )


def write_generations(path: Path) -> None:
    for sample in samples():
        append_record(path, sample)


def test_cap_counts_use_exact_metadata_and_ignore_control_generations(tmp_path: Path) -> None:
    path = tmp_path / "generations.jsonl"
    write_generations(path)
    counts = generation_caps(path)
    assert (counts.completed, counts.capped, counts.unknown) == (1, 1, 1)


def test_resume_does_not_rejudge_and_changed_generation_sha_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "generations.jsonl"
    write_generations(path)
    config = configuration(tmp_path)
    judge = CountingJudge(config.candidates[0])
    job = GenerationJob("speech", Split.VALIDATION, path, 3)
    first = judge_generation_set(job, config, judge)
    assert first.summary.requested_examples == 3
    assert judge.calls == 1
    assert judge_generation_set(job, config, judge) == first
    assert judge.calls == 1
    cached_requests = (
        config.output_directory / job.name / job.split.value / "requests.jsonl"
    ).read_text(encoding="utf-8")
    assert "audit-only" not in cached_requests
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            '"generated_response":"Hi"', '"generated_response":"Changed"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="generation SHA changed"):
        cache_requests(job, config, judge)
    assert judge.calls == 1


def test_main_full_coverage_and_no_active_writers_are_required(tmp_path: Path) -> None:
    path = tmp_path / "generations.jsonl"
    write_generations(path)
    config = configuration(tmp_path)
    with pytest.raises(ValueError, match="incomplete generation coverage"):
        cache_requests(
            GenerationJob("main", Split.TEST, path, 512),
            config,
            CountingJudge(config.candidates[0]),
        )
    config.results_root.mkdir()
    (config.results_root / "suite_state.json").write_text(
        SuiteState(
            completed=(), running="still-training", failed=(), started_at=1, updated_at=2
        ).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="writers have exited"):
        generation_jobs(config.results_root)


def test_controls_use_candidate_history_and_reuse_primary_judgments(tmp_path: Path) -> None:
    path = tmp_path / "all_generations.jsonl"
    history = (Turn(role=Role.USER, text="Earlier context"),)
    for index in range(8):
        primary = samples()[0].model_copy(
            update={
                "example_id": f"example-{index}",
                "dialogue_id": f"dialogue-{index}",
                "history": history,
            }
        )
        append_record(path, primary)
        for condition in (
            EvaluationCondition.SHUFFLED_SPEECH,
            EvaluationCondition.SPEECH_NO_HISTORY,
            EvaluationCondition.SHUFFLED_SPEECH_NO_HISTORY,
        ):
            append_record(
                path,
                primary.model_copy(
                    update={
                        "condition": condition,
                        "history": history
                        if condition == EvaluationCondition.SHUFFLED_SPEECH
                        else (),
                    }
                ),
            )
    config = configuration(tmp_path)
    judge = CountingJudge(config.candidates[0])
    job = GenerationJob("speech", Split.VALIDATION, path, 8)
    primary_set = judge_generation_set(job, config, judge)
    passed = calibration(judge, True)
    calibrated = CalibratedJudge(
        judge,
        JudgeSelection(
            attempts=(JudgeChallengeReport(calibration=passed, independent_challenge=passed),),
            selected=judge.config,
        ),
    )
    report = run_control_judging(config, calibrated, (job,), (primary_set,))
    assert len(report.sets) == 3
    assert len(report.comparisons) == 2
    assert all(item.comparison.shared_requested == 8 for item in report.comparisons)
    assert judge.calls == 4
    for item in report.sets:
        requests = tuple(
            JudgeRequest.model_validate_json(line)
            for line in (item.journal.path.parent / "requests.jsonl").read_bytes().splitlines()
        )
        expected = history if item.name.endswith("shuffled_speech") else ()
        assert all(request.history == expected for request in requests)
        assert all(request.transcript == "Hello" for request in requests)
    run_control_judging(config, calibrated, (job,), (primary_set,))
    assert judge.calls == 4
    assert primary_set.name == "speech"


def test_incomplete_control_coverage_is_not_silently_ignored(tmp_path: Path) -> None:
    path = tmp_path / "generations.jsonl"
    write_generations(path)
    with pytest.raises(ValueError, match="requires 8 cases"):
        control_generation_jobs((GenerationJob("speech", Split.TEST, path, 3),))
