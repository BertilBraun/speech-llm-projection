"""CPU checks of IndexTTS emotion mapping and immutable pilot resume records."""

from pathlib import Path

import pytest

from speech_projector.index_tts_pilot import (
    IndexPilotConfig,
    completed_clip,
    digest,
    emotion_vector,
    prepare_output,
    write_record,
)
from speech_projector.tts_pilot import (
    PilotEmotion,
    PilotTermination,
    TtsPilotCase,
    TtsPilotClip,
    TtsPilotManifest,
)


@pytest.mark.parametrize(
    ("emotion", "index"),
    (
        (PilotEmotion.HAPPY, 0),
        (PilotEmotion.ANGRY, 1),
        (PilotEmotion.SAD, 2),
        (PilotEmotion.NEUTRAL, 7),
    ),
)
def test_index_emotion_vector(emotion: PilotEmotion, index: int) -> None:
    vector = emotion_vector(emotion, 0.7)
    assert len(vector) == 8
    assert vector[index] == 0.7
    assert sum(vector) == pytest.approx(0.7)


@pytest.mark.parametrize("intensity", (0.0, -0.1, 1.01))
def test_index_rejects_invalid_intensity(intensity: float) -> None:
    with pytest.raises(ValueError, match="intensity"):
        emotion_vector(PilotEmotion.NEUTRAL, intensity)


def pilot_records(directory: Path) -> tuple[IndexPilotConfig, TtsPilotManifest]:
    reference = directory / "reference.wav"
    reference.write_bytes(b"test-reference")
    configuration = IndexPilotConfig(
        manifest_path=directory / "source_cases.json",
        output_directory=directory / "output",
        checkpoint_directory=directory / "weights",
        reference_audio=reference,
        reference_sha256=digest(reference),
        model_revision="model-pin",
        vendor_source_commit="vendor-pin",
        source_commit="source-pin",
    )
    case = TtsPilotCase(
        case_id="first_neutral",
        utterance_id="first",
        text="I finally heard back about the appointment.",
        emotion=PilotEmotion.NEUTRAL,
        seed=42,
    )
    manifest = TtsPilotManifest(cases=(case,), warmup_text="Warm up.", warmup_seed=41)
    return configuration, manifest


def test_resume_rejects_reference_and_configuration_changes(tmp_path: Path) -> None:
    configuration, manifest = pilot_records(tmp_path)
    prepare_output(configuration, manifest)
    prepare_output(configuration, manifest)
    changed = configuration.model_copy(update={"emotion_intensity": 0.8})
    with pytest.raises(ValueError, match="configuration changed"):
        prepare_output(changed, manifest)
    configuration.reference_audio.write_bytes(b"changed-reference")
    with pytest.raises(ValueError, match="reference hash"):
        prepare_output(configuration, manifest)


def test_completed_clip_hash_and_case_are_verified(tmp_path: Path) -> None:
    configuration, manifest = pilot_records(tmp_path)
    prepare_output(configuration, manifest)
    case = manifest.cases[0]
    assert completed_clip(configuration, case) is None
    audio = configuration.output_directory / "audio" / "first.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"accepted-waveform")
    clip = TtsPilotClip(
        case=case,
        audio_path="audio/first.wav",
        sha256=digest(audio),
        sample_rate=22050,
        audio_seconds=2,
        generation_seconds=1,
        real_time_factor=0.5,
        termination=PilotTermination.NOT_EXPOSED,
    )
    write_record(configuration.output_directory / "clips" / f"{case.case_id}.json", clip)
    assert completed_clip(configuration, case) == clip
    audio.write_bytes(b"corrupt-waveform")
    with pytest.raises(ValueError, match="WAV hash"):
        completed_clip(configuration, case)


def test_manifest_change_rejected_on_resume(tmp_path: Path) -> None:
    configuration, manifest = pilot_records(tmp_path)
    prepare_output(configuration, manifest)
    changed = manifest.model_copy(update={"warmup_seed": 99})
    with pytest.raises(ValueError, match="manifest changed"):
        prepare_output(configuration, changed)
