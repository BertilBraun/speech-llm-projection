from pathlib import Path

import numpy as np
import pytest
import soundfile

from speech_projector.data import DataConfig, SourceTurn, build_examples, dialogue_split, load_audio
from speech_projector.models import Role


def source_turn(index: int, duration: float | None = 1.0) -> SourceTurn:
    return SourceTurn(
        conversation_id="sample",
        model_dir="pair",
        turn_index=index,
        speaker="LLM1" if index % 2 == 0 else "LLM2",
        text=f"Spoken turn {index}",
        domain="science",
        emotion="happy",
        segment_audio_path=f"pair/sample/{index}.wav",
        audio_duration=duration,
        audio_cleaned_text=f"Spoken turn {index}",
    )


def test_conversation_history_excludes_audio_and_target(tmp_path: Path) -> None:
    turns = [source_turn(index) for index in range(6)]
    examples, report = build_examples(turns, DataConfig(root=tmp_path))
    last = next(example for example in examples if example.user_text == "Spoken turn 4")
    assert tuple(turn.text for turn in last.history) == ("Spoken turn 2", "Spoken turn 3")
    assert tuple(turn.role for turn in last.history) == (Role.USER, Role.ASSISTANT)
    assert last.target_text == "Spoken turn 5"
    assert report.usable_pairs == 3
    assert len({example.split for example in examples}) == 1


@pytest.mark.parametrize("duration", [None, 0.1, 31.0])
def test_filters_unusable_audio(duration: float | None, tmp_path: Path) -> None:
    turns = [source_turn(0, duration), source_turn(1), source_turn(2), source_turn(3)]
    examples, report = build_examples(turns, DataConfig(root=tmp_path))
    assert len(examples) == 1
    assert report.usable_pairs == 1


def test_seeded_split_is_conversation_level() -> None:
    assert dialogue_split("pair/dialogue", 42) == dialogue_split("pair/dialogue", 42)


def test_audio_loading_resamples_and_averages_channels(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"
    waveform = np.column_stack(
        (np.ones(24000, dtype=np.float32) * 0.2, np.zeros(24000, dtype=np.float32))
    )
    soundfile.write(path, waveform, 24000, subtype="FLOAT")
    loaded = load_audio(path)
    assert loaded.shape == (16000,)
    assert float(np.mean(loaded[100:-100])) == pytest.approx(0.1, abs=0.001)
