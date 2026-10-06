"""Audio resumability must preserve the exact case and persisted bytes."""

from pathlib import Path

import numpy as np
import pytest

from speech_projector.emotion_preview import (
    Delivery,
    PreviewCase,
    PreviewPlan,
    SynthesizedAudio,
    persist_clip,
)
from speech_projector.emotional_audio import EmotionalAudioConfig, completed_audio


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
