from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from scripts.compare_judgments import paired_values
from scripts.run_judge import CalibrationCategory, calibration_cases
from speech_projector.journal import append_record, read_journal
from speech_projector.judge import (
    JudgeFailure,
    JudgeMetric,
    JudgeOutcome,
    JudgeRequest,
    JudgeSuccess,
    JudgeVerdict,
    PairedMetricObservation,
    judge_batch,
    judge_one,
    judge_prompt,
    paired_dialogue_bootstrap,
)


def request() -> JudgeRequest:
    return JudgeRequest(
        example_id="hidden-source-id",
        dialogue_id="hidden-dialogue",
        history=(),
        transcript="Where is the Grand Canyon?",
        candidate_response="In Wyoming.",
    )


def test_judge_prompt_is_source_blind_and_excludes_reference_answers() -> None:
    prompt = judge_prompt(request())
    assert "hidden-source-id" not in prompt and "hidden-dialogue" not in prompt
    assert "reference_response" not in prompt
    assert "In Arizona" not in prompt
    assert "In Wyoming." in prompt


def test_judge_rejects_invalid_scores_and_preserves_retry_failure() -> None:
    responses = iter(
        (
            '{"relevance":7,"grounded_detail":0,"naturalness":3,"evidence":"wrong"}',
            '{"relevance":1,"grounded_detail":0,"naturalness":3,"evidence":"wrong named entity"}',
        )
    )
    corrections: list[str] = []

    def generate(item: JudgeRequest, correction: str) -> str:
        assert item == request()
        corrections.append(correction)
        return next(responses)

    result = judge_one(request(), generate, 2)
    assert isinstance(result, JudgeSuccess)
    assert len(result.raw_responses) == 2
    assert not result.verdict.acceptable
    assert corrections[0] == "" and "failed" in corrections[1]

    def invalid(item: JudgeRequest, correction: str) -> str:
        return "not JSON"

    failure = judge_one(request(), invalid, 2)
    assert isinstance(failure, JudgeFailure)
    assert len(failure.validation_errors) == 2


def test_verdict_extra_fields_and_acceptability_invariants() -> None:
    with pytest.raises(ValidationError):
        JudgeVerdict.model_validate_json(
            '{"relevance":3,"grounded_detail":3,"naturalness":3,"acceptable":true,"evidence":"fine","extra":1}'
        )


@pytest.mark.parametrize(
    ("scores", "acceptable"),
    (((3, 0, 3), False), ((1, 3, 3), False), ((3, 3, 1), False), ((2, 2, 2), True)),
)
def test_acceptance_requires_all_quality_dimensions(
    scores: tuple[int, int, int], acceptable: bool
) -> None:
    verdict = JudgeVerdict(
        relevance=scores[0],
        grounded_detail=scores[1],
        naturalness=scores[2],
        evidence="Assessed candidate content",
    )
    assert verdict.acceptable is acceptable
    assert "acceptable" not in verdict.model_dump()


def test_paired_quality_excludes_parse_failures_and_rejects_different_inputs() -> None:
    success = JudgeSuccess(
        request=request(),
        verdict=JudgeVerdict(relevance=3, grounded_detail=0, naturalness=3, evidence="wrong"),
        raw_responses=("valid",),
    )
    failure = JudgeFailure(
        request=request(), raw_responses=("invalid",), validation_errors=("schema error",)
    )
    assert paired_values((success,), (failure,), JudgeMetric.ACCEPTABLE) == ()
    changed = success.model_copy(
        update={"request": request().model_copy(update={"transcript": "Different user"})}
    )
    with pytest.raises(ValueError, match="differ in transcript/history"):
        paired_values((success,), (changed,), JudgeMetric.RELEVANCE)


def test_calibration_contains_valid_generic_conversation_and_bad_specific_generic() -> None:
    cases = calibration_cases()
    assert len(cases) == 13
    conversation = next(
        item for item in cases if item.category == CalibrationCategory.CONVERSATIONAL
    )
    assert "Tokyo" in conversation.request.transcript
    assert "trip" in conversation.request.candidate_response
    assert any(item.category == CalibrationCategory.WRONG_ENTITY for item in cases)


def test_bootstrap_samples_whole_dialogues_with_example_weighting() -> None:
    observations = (
        PairedMetricObservation("a", "shared", 1),
        PairedMetricObservation("b", "shared", 1),
        PairedMetricObservation("c", "other", -1),
    )
    interval = paired_dialogue_bootstrap(observations, draws=2000)
    assert interval.examples == 3 and interval.dialogues == 2
    assert interval.estimate == pytest.approx(1 / 3)
    assert interval.lower == -1 and interval.upper == 1
    assert interval == paired_dialogue_bootstrap(observations, draws=2000)
    with pytest.raises(ValueError, match="two independent"):
        paired_dialogue_bootstrap(observations[:2])


def test_batched_retry_preserves_order_and_only_retries_invalid_rows() -> None:
    second = JudgeRequest(
        example_id="second",
        dialogue_id="another",
        history=(),
        transcript="Hello",
        candidate_response="Hi",
    )
    valid = '{"relevance":3,"grounded_detail":3,"naturalness":3,"evidence":"responsive"}'
    sizes: list[int] = []

    def generate(items: Sequence[JudgeRequest], corrections: Sequence[str]) -> tuple[str, ...]:
        sizes.append(len(items))
        if len(sizes) == 1:
            assert not any(corrections)
            return "invalid", valid
        assert items == (request(),) and "failed" in corrections[0]
        return (valid,)

    outcomes = judge_batch((request(), second), generate, 2)
    assert sizes == [2, 1]
    assert tuple(item.request.example_id for item in outcomes) == ("hidden-source-id", "second")
    assert all(isinstance(item, JudgeSuccess) for item in outcomes)


def test_judge_discriminated_union_journal_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "judge.jsonl"
    outcome = JudgeFailure(
        request=request(), raw_responses=("bad",), validation_errors=("bad JSON",)
    )
    append_record(path, outcome)
    assert read_journal(path, TypeAdapter(JudgeOutcome)) == (outcome,)
