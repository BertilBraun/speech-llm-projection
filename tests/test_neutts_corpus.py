"""Resume checks require matching bytes and genuine audio EOS, not record counts."""

from pathlib import Path

import numpy as np
import pytest
import soundfile

from scripts.neutts_pilot_state import digest, write_record
from speech_projector.neutts_batch_benchmark import BatchClipEvidence, BatchMeasurement
from speech_projector.neutts_corpus import (
    persist_completed,
    restore_evidence,
    validate_paired_manifest,
    verify_audio,
)
from speech_projector.tts_pilot import (
    PilotEmotion,
    PilotTermination,
    TtsPilotCase,
    TtsPilotClip,
    TtsPilotManifest,
)


def paired_cases() -> tuple[TtsPilotCase, ...]:
    return tuple(
        TtsPilotCase(
            case_id=f"one_{emotion.value}",
            utterance_id="one",
            text="The appointment reply arrived today.",
            emotion=emotion,
            seed=42,
        )
        for emotion in (PilotEmotion.HAPPY, PilotEmotion.ANGRY)
    )


def measurement(directory: Path, termination: PilotTermination) -> BatchMeasurement:
    evidence: list[BatchClipEvidence] = []
    for case in paired_cases():
        relative = Path("audio") / f"{case.case_id}.wav"
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(path, np.full(24000, 1.01, dtype=np.float32), 24000, subtype="FLOAT")
        evidence.append(
            BatchClipEvidence(
                clip=TtsPilotClip(
                    case=case,
                    audio_path=relative.as_posix(),
                    sha256=digest(path),
                    sample_rate=24000,
                    audio_seconds=1,
                    generation_seconds=2,
                    real_time_factor=2,
                    termination=termination,
                ),
                prompt_tokens=100,
                speech_tokens=25,
                generated_token_ids=(10, 99) if termination == PilotTermination.STOP else (10, 11),
            )
        )
    return BatchMeasurement(
        requested_batch_size=28,
        repetition=0,
        batch_index=0,
        shared_batch_seed=42,
        padded_prompt_width=100,
        generation_token_cap=2000,
        frontend_seconds=0.1,
        backbone_seconds=1.5,
        codec_watermark_seconds=0.4,
        end_to_end_seconds=2,
        clips=tuple(evidence),
    )


def test_hash_verified_resume_and_actual_eos(tmp_path: Path) -> None:
    batch = measurement(tmp_path, PilotTermination.STOP)
    persist_completed(batch, tmp_path)
    verified = verify_audio(paired_cases(), tmp_path, 99)
    assert verified.verified_clips == 2
    assert verified.complete_pairs == 1
    assert verified.total_audio_seconds == 2
    assert verified.overshoot_samples == 48000
    (tmp_path / batch.clips[0].clip.audio_path).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        restore_evidence(paired_cases()[0], tmp_path)


def test_capped_attempts_do_not_count_as_completed(tmp_path: Path) -> None:
    persist_completed(measurement(tmp_path, PilotTermination.TOKEN_LIMIT), tmp_path)
    verified = verify_audio(paired_cases(), tmp_path, 99)
    assert verified.verified_clips == 0
    assert len(verified.missing_cases) == 2


def test_false_stop_record_without_end_token_is_rejected(tmp_path: Path) -> None:
    batch = measurement(tmp_path, PilotTermination.STOP)
    item = batch.clips[0].model_copy(update={"generated_token_ids": (10, 11)})
    write_record(tmp_path / "clips" / f"{item.clip.case.case_id}.json", item)
    with pytest.raises(ValueError, match="actual speech-end EOS"):
        verify_audio(paired_cases(), tmp_path, 99)


def test_completed_clips_cannot_be_replaced(tmp_path: Path) -> None:
    batch = measurement(tmp_path, PilotTermination.STOP)
    persist_completed(batch, tmp_path)
    with pytest.raises(ValueError, match="Refusing to replace"):
        persist_completed(batch, tmp_path)


def test_manifest_requires_complete_emotional_pairs() -> None:
    manifest = TtsPilotManifest(cases=paired_cases(), warmup_text="Warm up.", warmup_seed=42)
    validate_paired_manifest(manifest)
    with pytest.raises(ValueError, match="exactly two"):
        validate_paired_manifest(manifest.model_copy(update={"cases": manifest.cases[:1]}))
