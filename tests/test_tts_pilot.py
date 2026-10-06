import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile

from scripts.report_tts_pilot import ModelResults, verify_results
from speech_projector.tts_pilot import (
    PilotEmotion,
    PilotTermination,
    TtsPilotCase,
    TtsPilotClip,
    TtsPilotFailure,
    TtsPilotManifest,
    TtsPilotResult,
    load_pilot_manifest,
)


def case(emotion: PilotEmotion = PilotEmotion.NEUTRAL) -> TtsPilotCase:
    return TtsPilotCase(
        case_id=f"utterance_01_{emotion.value}",
        utterance_id="utterance_01",
        text="I finally heard back about the appointment today.",
        emotion=emotion,
        seed=42,
    )


def clip() -> TtsPilotClip:
    return TtsPilotClip(
        case=case(),
        audio_path="audio/utterance_01_neutral.wav",
        sha256="a" * 64,
        sample_rate=24000,
        audio_seconds=5.0,
        generation_seconds=2.0,
        real_time_factor=0.4,
        termination=PilotTermination.NOT_EXPOSED,
    )


def result() -> TtsPilotResult:
    return TtsPilotResult(
        model_repository="example/model",
        model_revision="b" * 40,
        source_commit="c" * 40,
        backend_description="Offline single-GPU SDK",
        speaker_description="Provided neutral reference",
        startup_seconds=2.0,
        reference_preparation_seconds=0.3,
        warmup_generation_seconds=1.5,
        clips=(clip(),),
        failures=(),
    )


def test_literal_utterance_remains_identical_across_emotions_and_manifest_roundtrip(
    tmp_path: Path,
) -> None:
    manifest = TtsPilotManifest(
        cases=tuple(case(emotion) for emotion in PilotEmotion),
        warmup_text="This is a separate warmup sentence.",
        warmup_seed=41,
    )
    path = tmp_path / "cases.json"
    path.write_text(manifest.model_dump_json(), encoding="utf-8")
    assert load_pilot_manifest(path) == manifest
    assert len({item.text for item in manifest.cases}) == 1
    assert len({item.case_id for item in manifest.cases}) == 4
    assert manifest.cases[0].seed == 42


@pytest.mark.parametrize("problem", ("duplicate_id", "different_text", "duplicate_emotion"))
def test_manifest_rejects_confounded_or_duplicate_comparisons(problem: str) -> None:
    first = case()
    second = case(PilotEmotion.HAPPY)
    match problem:
        case "duplicate_id":
            second = second.model_copy(update={"case_id": first.case_id})
        case "different_text":
            second = second.model_copy(update={"text": "A different literal utterance."})
        case "duplicate_emotion":
            second = second.model_copy(update={"emotion": first.emotion})
    with pytest.raises(ValueError):
        TtsPilotManifest(cases=(first, second), warmup_text="Warm up.", warmup_seed=41)


def test_clip_rtf_must_match_measured_time_and_duration() -> None:
    original = clip()
    with pytest.raises(ValueError):
        TtsPilotClip.model_validate_json(
            original.model_copy(update={"real_time_factor": 1.0}).model_dump_json()
        )
    assert TtsPilotClip.model_validate_json(original.model_dump_json()) == original


@pytest.mark.parametrize("field", ("audio_seconds", "generation_seconds", "real_time_factor"))
@pytest.mark.parametrize("invalid", (float("inf"), float("nan")))
def test_nonfinite_measurements_cannot_be_serialized_as_valid_clips(
    field: str, invalid: float
) -> None:
    with pytest.raises(ValueError):
        TtsPilotClip.model_validate(clip().model_copy(update={field: invalid}).model_dump())


def test_result_does_not_count_a_case_as_both_success_and_failure() -> None:
    failed = TtsPilotFailure(case=case(), error="Failed after generation.")
    original = result()
    with pytest.raises(ValueError):
        TtsPilotResult.model_validate_json(
            original.model_copy(update={"failures": (failed,)}).model_dump_json()
        )


def test_result_does_not_double_count_successful_audio() -> None:
    original = result()
    with pytest.raises(ValueError):
        TtsPilotResult.model_validate_json(
            original.model_copy(update={"clips": (clip(), clip())}).model_dump_json()
        )


def test_unknown_termination_is_preserved_instead_of_inferred_stop() -> None:
    original = result()
    restored = TtsPilotResult.model_validate_json(original.model_dump_json())
    assert restored.clips[0].termination == PilotTermination.NOT_EXPOSED


def test_aggregate_rtf_weights_by_actual_audio_duration(tmp_path: Path) -> None:
    first_case, second_case = case(), case(PilotEmotion.HAPPY)
    manifest = TtsPilotManifest(
        cases=(first_case, second_case), warmup_text="Warm up separately.", warmup_seed=41
    )
    clips: list[TtsPilotClip] = []
    for candidate, duration, generation in ((first_case, 1.0, 2.0), (second_case, 4.0, 1.0)):
        path = tmp_path / f"{candidate.case_id}.wav"
        soundfile.write(path, np.full(int(duration * 1000), 0.1, dtype=np.float32), 1000)
        clips.append(
            TtsPilotClip(
                case=candidate,
                audio_path=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                sample_rate=1000,
                audio_seconds=duration,
                generation_seconds=generation,
                real_time_factor=generation / duration,
                termination=PilotTermination.NOT_EXPOSED,
            )
        )
    measurements = TtsPilotResult.model_validate_json(
        result().model_copy(update={"clips": tuple(clips)}).model_dump_json()
    )
    audit = verify_results(manifest, ModelResults(directory=tmp_path, result=measurements))
    assert audit.aggregate_real_time_factor == pytest.approx(3 / 5)
    assert audit.aggregate_real_time_factor != pytest.approx((2 + 0.25) / 2)
    path = tmp_path / clips[0].audio_path
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="WAV hash mismatch"):
        verify_results(manifest, ModelResults(directory=tmp_path, result=measurements))
