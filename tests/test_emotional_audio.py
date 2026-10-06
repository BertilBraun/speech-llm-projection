"""Audio resumability must preserve the exact case and persisted bytes."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from scripts.inventory_results import stable_digest
from speech_projector.emotion_preview import (
    Delivery,
    PreviewCase,
    PreviewPlan,
    SynthesizedAudio,
    persist_clip,
)
from speech_projector.emotional_audio import (
    EmotionalAudioConfig,
    UncommittedAudioRecovery,
    archive_uncommitted_audio,
    completed_audio,
)


def audio_config(directory: Path) -> EmotionalAudioConfig:
    return EmotionalAudioConfig(
        output=directory,
        source_commit="a" * 40,
        plan=PreviewPlan(
            max_new_tokens=512,
            cases=(
                PreviewCase(
                    case_id="first_happy",
                    text="I noticed your message and wanted to talk today.",
                    delivery=Delivery.HAPPY,
                    instruct="Speak with clear, delighted warmth.",
                    seed=42,
                ),
            ),
        ),
    )


def test_completed_audio_requires_matching_case_and_bytes(tmp_path: Path) -> None:
    config = audio_config(tmp_path)
    clip = persist_clip(
        config.output,
        config.plan,
        config.plan.cases[0],
        SynthesizedAudio(
            waveform=np.ones(2400, dtype=np.float32),
            sample_rate=24000,
            codec_tokens=12,
            runtime_seconds=0.25,
        ),
    )
    assert completed_audio(config) == (clip,)
    audio = config.output / clip.audio.path
    audio.write_bytes(audio.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        completed_audio(config)


def test_completed_audio_rejects_changed_case(tmp_path: Path) -> None:
    config = audio_config(tmp_path)
    persist_clip(
        config.output,
        config.plan,
        config.plan.cases[0],
        SynthesizedAudio(
            waveform=np.ones(2400, dtype=np.float32),
            sample_rate=24000,
            codec_tokens=12,
            runtime_seconds=0.25,
        ),
    )
    changed_case = config.plan.cases[0].model_copy(update={"text": "Changed literal words."})
    changed = config.model_copy(
        update={"plan": config.plan.model_copy(update={"cases": (changed_case,)})}
    )
    with pytest.raises(ValueError, match="case changed"):
        completed_audio(changed)


def test_audio_config_requires_larger_retry_cap(tmp_path: Path) -> None:
    config = audio_config(tmp_path)
    with pytest.raises(ValueError, match="Retry cap must exceed"):
        EmotionalAudioConfig(
            output=tmp_path,
            plan=config.plan,
            source_commit=config.source_commit,
            retry_max_new_tokens=512,
        )


def test_uncommitted_wav_is_preserved_with_hash_proof_then_regenerated(tmp_path: Path) -> None:
    config = audio_config(tmp_path)
    case = config.plan.cases[0]
    source = config.output / "audio" / f"{case.case_id}.wav"
    source.parent.mkdir()
    content = b"unfinished metadata transaction with preserved audio bytes"
    source.write_bytes(content)
    size, digest = stable_digest(source)
    recoveries = archive_uncommitted_audio(config)
    assert len(recoveries) == 1
    recovery = recoveries[0]
    assert recovery.source_path == source.resolve()
    assert recovery.sha256 == digest
    assert recovery.bytes == size
    assert recovery.destination.read_bytes() == content
    assert digest in recovery.destination.name
    assert (
        UncommittedAudioRecovery.model_validate_json(
            recovery.destination.with_suffix(".json").read_bytes()
        )
        == recovery
    )
    assert not source.exists()
    assert archive_uncommitted_audio(config) == ()
    assert completed_audio(config) == ()
    clip = persist_clip(
        config.output,
        config.plan,
        case,
        SynthesizedAudio(
            waveform=np.ones(2400, dtype=np.float32),
            sample_rate=24000,
            codec_tokens=12,
            runtime_seconds=0.25,
        ),
    )
    assert completed_audio(config) == (clip,)
    assert recovery.destination.read_bytes() == content
    assert archive_uncommitted_audio(config) == ()


def test_archive_never_moves_a_committed_clip_even_when_hash_is_wrong(tmp_path: Path) -> None:
    config = audio_config(tmp_path)
    clip = persist_clip(
        config.output,
        config.plan,
        config.plan.cases[0],
        SynthesizedAudio(
            waveform=np.ones(2400, dtype=np.float32),
            sample_rate=24000,
            codec_tokens=12,
            runtime_seconds=0.25,
        ),
    )
    audio = config.output / clip.audio.path
    audio.write_bytes(b"corrupted committed audio")
    assert archive_uncommitted_audio(config) == ()
    assert audio.read_bytes() == b"corrupted committed audio"
    with pytest.raises(ValueError, match="bytes changed"):
        completed_audio(config)


def test_interrupted_atomic_clip_record_commit_leaves_recoverable_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = audio_config(tmp_path)
    final_record = config.output / "clips" / f"{config.plan.cases[0].case_id}.json"
    original_replace: Callable[[Path, Path], Path] = Path.replace

    def interrupted_replace(path: Path, target: Path) -> Path:
        if target == final_record:
            assert path.suffix == ".part"
            assert not final_record.exists()
            raise OSError("Simulated crash before atomic metadata commit")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupted_replace)
    with pytest.raises(OSError, match="Simulated crash"):
        persist_clip(
            config.output,
            config.plan,
            config.plan.cases[0],
            SynthesizedAudio(
                waveform=np.ones(2400, dtype=np.float32),
                sample_rate=24000,
                codec_tokens=12,
                runtime_seconds=0.25,
            ),
        )
    assert not final_record.exists()
    assert final_record.with_suffix(".part").exists()
    recoveries = archive_uncommitted_audio(config)
    assert len(recoveries) == 1
    assert recoveries[0].destination.exists()
