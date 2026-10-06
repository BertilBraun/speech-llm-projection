from pathlib import Path

import pytest

from scripts.challenge_judge import JudgeChallengeReport, challenge_cases
from scripts.run_judge import CalibrationCategory, CalibrationSummary, calibrate, calibration_cases
from speech_projector.judge import JUDGE_SYSTEM, JudgeConfig, LocalJudge, rubric_digest


def summary(passed: bool) -> CalibrationSummary:
    return CalibrationSummary(
        config=JudgeConfig(),
        rubric_sha256=rubric_digest(),
        cases=calibration_cases(),
        groups=(),
        passed=passed,
        seconds=1,
        decoded_response_tokens=100,
    )


def test_challenge_is_independent_and_covers_context_changes_and_social_validity() -> None:
    cases = challenge_cases()
    assert len(cases) == 30
    assert {item.category for item in cases} == set(CalibrationCategory)
    assert len({item.request.example_id for item in cases}) == 30
    original_transcripts = {item.request.transcript for item in calibration_cases()}
    assert not original_transcripts & {item.request.transcript for item in cases}
    for current in ("Nora", "Rome", "Milo", "Oslo"):
        assert current not in JUDGE_SYSTEM
    assert any(item.request.history for item in cases)
    social = tuple(item for item in cases if item.category == CalibrationCategory.CONVERSATIONAL)
    assert len(social) == 5
    assert all("?" not in item.request.transcript for item in social)


@pytest.mark.parametrize(("initial", "independent"), ((True, False), (False, True)))
def test_challenge_gate_requires_both_original_and_independent_success(
    initial: bool, independent: bool
) -> None:
    assert not JudgeChallengeReport(
        calibration=summary(initial), independent_challenge=summary(independent)
    ).passed


def test_incomplete_category_benchmark_is_rejected_before_model_use(tmp_path: Path) -> None:
    model = LocalJudge.__new__(LocalJudge)
    with pytest.raises(ValueError, match="cover every declared quality category"):
        calibrate(model, tmp_path, cases=challenge_cases()[:1])
