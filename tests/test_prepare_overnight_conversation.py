"""Real shared waveform receipts determine immutable feature names and fixture inputs."""

from pathlib import Path

import numpy as np
import pytest
import soundfile

from scripts.inventory_results import stable_digest
from scripts.prepare_overnight_conversation import prepare_followups
from speech_projector.tts_pilot import TtsPilotResult
from tests.test_overnight_conversation import followups


def test_followup_preparation_verifies_exact_waveform_and_uses_content_hash_features(
    tmp_path: Path,
) -> None:
    clips = []
    for index, item in enumerate(followups()):
        path = tmp_path / item.clip.audio_path
        soundfile.write(path, np.full(48000, 0.1 * (index + 1), dtype=np.float32), 24000)
        _, digest = stable_digest(path)
        clips.append(item.clip.model_copy(update={"sha256": digest}))
    result = TtsPilotResult(
        model_repository="neuphonic/neutts-2e",
        model_revision="source_revision",
        source_commit="source_commit",
        backend_description="Measured public SDK",
        speaker_description="Shared Paul voice",
        startup_seconds=1,
        reference_preparation_seconds=1,
        warmup_generation_seconds=1,
        clips=tuple(clips),
        failures=(),
    )
    prepared = prepare_followups(result, tmp_path, tmp_path / "features")
    assert len({item.feature_path for item in prepared}) == 3
    assert prepared[0].feature_path.name == clips[0].sha256 + ".pt"
    changed = result.model_copy(
        update={"clips": (clips[0].model_copy(update={"sample_rate": 16000}),) + tuple(clips[1:])}
    )
    with pytest.raises(ValueError, match="waveform metadata"):
        prepare_followups(changed, tmp_path, tmp_path / "features")
    (tmp_path / clips[0].audio_path).write_bytes(b"corrupted waveform")
    with pytest.raises(ValueError, match="recorded hash"):
        prepare_followups(result, tmp_path, tmp_path / "features")
