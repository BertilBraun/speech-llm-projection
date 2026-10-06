"""Judge source blindness, durable retries, pair identity and scalar score invariants."""

from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from pydantic import TypeAdapter, ValidationError

from scripts.judge_overnight import ToneCalibrationResult, calibrate, calibration_cases
from speech_projector.emotion_preview import Delivery
from speech_projector.judge import JUDGE_SYSTEM, JudgeConfig, JudgeRequest, LocalJudge, judge_prompt
from speech_projector.models import EvaluationCondition, FileArtifact, SampleGeneration
from speech_projector.overnight_data import (
    NeuEmotionalExampleSource,
    OrdinaryExampleSource,
    QwenEmotionalExampleSource,
    SourceSidecar,
)
from speech_projector.overnight_judge import (
    TONE_PAIR_SYSTEM,
    FinalJudgingProgram,
    FinalJudgingQuota,
    FinalJudgingSet,
    NeuToneJudgeRequest,
    QwenToneJudgeRequest,
    ResponseChoice,
    ToneJudgeFailure,
    ToneJudgeRequest,
    ToneJudgeSuccess,
    ToneJudgeVerdict,
    TonePairVerdict,
    build_final_judging_set,
    judge_pair_batch,
    judge_tone_batch,
    paired_response_requests,
    run_pair_judgments,
    run_tone_judgments,
    summarize_pair_judgments,
    tone_pair_prompt,
    tone_prompt,
)
from speech_projector.tts_pilot import PilotEmotion


def request() -> NeuToneJudgeRequest:
    return NeuToneJudgeRequest(
        utterance=JudgeRequest(
            example_id="secret_run_a",
            dialogue_id="secret_family",
            history=(),
            transcript="The deadline changed again.",
            candidate_response="That sounds frustrating; asking for a clear deadline may help.",
        ),
        intended_tone=PilotEmotion.ANGRY,
    )


def test_tone_prompt_hides_source_and_reuses_canonical_emotion_variants() -> None:
    prompt = tone_prompt(request())
    assert "secret_run" not in prompt and "secret_family" not in prompt
    assert '"kind"' not in prompt
    assert "angry" in prompt
    assert "teacher" not in prompt
    adapter = TypeAdapter(ToneJudgeRequest)
    for current in (
        request(),
        request().model_copy(update={"intended_tone": PilotEmotion.HAPPY}),
        QwenToneJudgeRequest(utterance=request().utterance, intended_tone=Delivery.SARCASTIC),
    ):
        assert adapter.validate_json(current.model_dump_json()) == current


@pytest.mark.parametrize(("tone", "acceptable"), ((0, False), (1, False), (2, True), (3, True)))
def test_acceptability_requires_content_and_tone(tone: int, acceptable: bool) -> None:
    verdict = ToneJudgeVerdict(
        relevance=3, grounded_detail=3, naturalness=3, tone_appropriateness=tone, evidence="fit"
    )
    assert verdict.acceptable is acceptable
    assert not verdict.model_copy(update={"grounded_detail": 0}).acceptable
    with pytest.raises(ValidationError):
        TonePairVerdict(choice="unknown", evidence="invalid")


def test_pair_judge_preserves_response_identity_and_randomizes_slots_without_source_leak() -> None:
    first = request()
    second = first.model_copy(
        update={
            "intended_tone": PilotEmotion.SAD,
            "utterance": first.utterance.model_copy(
                update={"example_id": "secret_run_b", "candidate_response": "This sounds heavy."}
            ),
        }
    )
    pairs = paired_response_requests(first, second)
    for pair, original in zip(pairs, (first, second), strict=True):
        correct = (
            pair.context.utterance.candidate_response
            if pair.matching_response == ResponseChoice.A
            else pair.alternative_response
        )
        assert correct == original.utterance.candidate_response
        assert pair.context.intended_tone == original.intended_tone
        rendered = tone_pair_prompt(pair)
        assert "secret_run" not in rendered and "secret_family" not in rendered
        assert "matching_response" not in rendered
    unrelated = second.model_copy(
        update={"utterance": second.utterance.model_copy(update={"transcript": "Other words"})}
    )
    with pytest.raises(ValueError, match="same words/history/family"):
        paired_response_requests(first, unrelated)


def test_json_retries_only_failed_rows_and_preserves_failures() -> None:
    second = request().model_copy(
        update={"utterance": request().utterance.model_copy(update={"example_id": "second"})}
    )
    sizes: list[int] = []
    valid = ToneJudgeVerdict(
        relevance=3, grounded_detail=2, naturalness=3, tone_appropriateness=2, evidence="fit"
    ).model_dump_json()

    def generate(prompts: Sequence[str]) -> tuple[str, ...]:
        sizes.append(len(prompts))
        return ("invalid", valid) if len(sizes) == 1 else (valid,)

    outcomes = judge_tone_batch((request(), second), generate, 2)
    assert sizes == [2, 1]
    assert all(isinstance(item, ToneJudgeSuccess) for item in outcomes)
    assert len(outcomes[0].raw_responses) == 2
    failed = judge_tone_batch((request(),), lambda prompts: ("invalid",), 2)
    assert isinstance(failed[0], ToneJudgeFailure)
    assert len(failed[0].validation_errors) == 2
    pair = paired_response_requests(
        request(), second.model_copy(update={"intended_tone": PilotEmotion.SAD})
    )[0]
    paired = judge_pair_batch((pair,), lambda prompts: ('{"choice":"tie","evidence":"equal"}',), 1)
    assert paired[0].raw_responses == ('{"choice":"tie","evidence":"equal"}',)


class ScriptedJudge(LocalJudge):
    __test__ = False

    def __init__(self) -> None:
        self.config = JudgeConfig(batch_size=16)
        self.device = torch.device("cpu")
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def generate_prompts(self, system_text: str, prompts: Sequence[str]) -> tuple[str, ...]:
        self.calls.append((system_text, tuple(prompts)))
        if system_text == TONE_PAIR_SYSTEM:
            return tuple('{"choice":"A","evidence":"scripted preference"}' for _ in prompts)
        return tuple(
            ToneJudgeVerdict(
                relevance=3,
                grounded_detail=2,
                naturalness=3,
                tone_appropriateness=2,
                evidence="fit",
            ).model_dump_json()
            for _ in prompts
        )


def test_shared_model_content_prompt_is_unchanged_and_tone_resume_binds_inputs(
    tmp_path: Path,
) -> None:
    judge = ScriptedJudge()
    ordinary = request().utterance
    judge.generate_batch((ordinary,), ("",))
    assert judge.calls == [(JUDGE_SYSTEM, (judge_prompt(ordinary),))]
    outcomes = run_tone_judgments(judge, (request(),), tmp_path)
    calls = len(judge.calls)
    assert run_tone_judgments(judge, (request(),), tmp_path) == outcomes
    assert len(judge.calls) == calls
    changed = request().model_copy(update={"intended_tone": PilotEmotion.SAD})
    with pytest.raises(ValueError, match="transcript, history, tone or candidate"):
        run_tone_judgments(judge, (changed,), tmp_path)


def test_final_set_requires_complete_pair_quota_and_excludes_teacher_references() -> None:
    samples: list[SampleGeneration] = []
    sources: list[SourceSidecar] = []
    for index in range(2):
        identifier = f"ordinary_{index}"
        sources.append(
            OrdinaryExampleSource(
                example_id=identifier, source_manifest=Path("o"), source_example_id=identifier
            )
        )
    for cohort in ("qwen", "neu"):
        for base in range(2):
            for index in range(2):
                identifier = f"{cohort}_{base}_{index}"
                if cohort == "qwen":
                    sources.append(
                        QwenEmotionalExampleSource(
                            example_id=identifier,
                            source_manifest=Path(cohort),
                            source_example_id=identifier,
                            base_id=f"{cohort}_{base}",
                            family_id=f"family_{cohort}_{base}",
                            emotion=(Delivery.HAPPY, Delivery.SAD)[index],
                        )
                    )
                else:
                    sources.append(
                        NeuEmotionalExampleSource(
                            example_id=identifier,
                            source_manifest=Path(cohort),
                            source_example_id=identifier,
                            base_id=f"{cohort}_{base}",
                            family_id=f"family_{cohort}_{base}",
                            emotion=(PilotEmotion.HAPPY, PilotEmotion.ANGRY)[index],
                        )
                    )
    for source in sources:
        samples.append(
            SampleGeneration(
                example_id=source.example_id,
                dialogue_id=source.example_id,
                condition=EvaluationCondition.SPEECH,
                history=(),
                user_transcript="The same literal words.",
                gold_response="Teacher reference must be hidden.",
                generated_response="Actual student response.",
                duration=2,
            )
        )
    quota = FinalJudgingQuota(ordinary=2, old_emotional_pairs=2, new_emotional_pairs=2)
    selected = build_final_judging_set(samples, sources, quota)
    assert len(selected.new_emotional) == len(selected.new_preferences) == 4
    assert "Teacher reference" not in selected.model_dump_json()
    assert FinalJudgingSet.model_validate_json(selected.model_dump_json()) == selected
    combined = tuple(
        sample.model_copy(update={"condition": condition})
        for condition in (
            EvaluationCondition.SPEECH,
            EvaluationCondition.TEXT,
            EvaluationCondition.ASR,
        )
        for sample in samples
    )
    sets = tuple(
        build_final_judging_set(combined, sources, quota, condition)
        for condition in (
            EvaluationCondition.SPEECH,
            EvaluationCondition.TEXT,
            EvaluationCondition.ASR,
        )
    )
    identifiers = tuple(item.example_id for current in sets for item in current.ordinary)
    assert len(set(identifiers)) == 6
    for current in sets:
        assert current.ordinary[0].example_id.startswith(current.condition.value + ":")
        assert current.condition.value + ":" not in judge_prompt(current.ordinary[0])
        assert current.condition.value + ":" not in tone_prompt(current.new_emotional[0])
    receipt = FileArtifact(
        path=Path("saved_generations.jsonl"),
        source_path=Path("saved_generations.jsonl").resolve(),
        bytes=123,
        sha256="a" * 64,
    )
    program = FinalJudgingProgram(sets=sets, source_artifacts=(receipt,))
    assert FinalJudgingProgram.model_validate_json(program.model_dump_json()) == program
    with pytest.raises(ValueError, match="unique nonempty conditions"):
        FinalJudgingProgram(sets=(sets[0], sets[0]), source_artifacts=(receipt,))
    with pytest.raises(ValueError, match="speech, text and ASR"):
        build_final_judging_set(combined, sources, quota, EvaluationCondition.ZERO_SPEECH)
    with pytest.raises(ValueError, match="quotas"):
        build_final_judging_set(samples[:-1], sources, quota)


def test_calibration_uses_twelve_candidates_four_intended_tones_and_both_acceptability_labels() -> (
    None
):
    cases = calibration_cases()
    assert len(cases) == 12
    assert sum(item.expected_acceptable for item in cases) == 6
    assert {item.request.intended_tone for item in cases} == {
        PilotEmotion.HAPPY,
        PilotEmotion.SAD,
        PilotEmotion.ANGRY,
        PilotEmotion.FEARFUL,
    }


def test_calibration_rejects_unselective_judge_and_preserves_actual_outputs(tmp_path: Path) -> None:
    result = calibrate(ScriptedJudge(), tmp_path)
    assert not result.passed
    assert result.valid == 12 and result.correct_acceptability == 6
    assert result.preference_requests == 6
    assert (
        ToneCalibrationResult.model_validate_json((tmp_path / "result.json").read_bytes()) == result
    )
    assert len((tmp_path / "single" / "judgments.jsonl").read_text().splitlines()) == 12


def test_identical_replies_are_ties_even_if_judge_selects_slot(tmp_path: Path) -> None:
    first = request()
    second = first.model_copy(
        update={
            "intended_tone": PilotEmotion.SAD,
            "utterance": first.utterance.model_copy(update={"example_id": "other"}),
        }
    )
    pair = paired_response_requests(first, second)[0]
    other_family = pair.model_copy(
        update={
            "context": pair.context.model_copy(
                update={
                    "utterance": pair.context.utterance.model_copy(
                        update={"example_id": "independent", "dialogue_id": "second_family"}
                    )
                }
            )
        }
    )
    judge = ScriptedJudge()
    outcomes = run_pair_judgments(judge, (pair, other_family), tmp_path)
    calls = len(judge.calls)
    assert run_pair_judgments(judge, (pair, other_family), tmp_path) == outcomes
    assert len(judge.calls) == calls
    summary = summarize_pair_judgments(outcomes)
    assert summary.matching_wins == summary.matching_losses == 0
    assert summary.identical_responses == summary.ties == 2
    changed = pair.model_copy(update={"alternative_response": "Another candidate."})
    with pytest.raises(ValueError, match="candidate replies"):
        run_pair_judgments(judge, (changed, other_family), tmp_path)
