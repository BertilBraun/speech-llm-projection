"""CPU-only persistence and identity checks for the isolated NeuTTS pilot adapter."""

from pathlib import Path

import numpy as np
import pytest

from scripts.neutts_pilot_state import (
    NeuTtsPilotConfig,
    PilotDevice,
    digest,
    persist_clip,
    restore_clip,
    validate_configuration,
)
from scripts.prepare_neutts_models import NeuTtsPreparation, PinnedRepository
from speech_projector.tts_pilot import PilotEmotion, TtsPilotCase


@pytest.fixture
def configuration(tmp_path: Path) -> NeuTtsPilotConfig:
    manifest = tmp_path / "cases.json"
    manifest.write_text("manifest identity fixture", encoding="utf-8")
    preparation = NeuTtsPreparation(
        source_commit="sdk_source",
        repositories=(
            PinnedRepository(
                repository="neuphonic/neutts-2e",
                revision="backbone_pin",
                snapshot=tmp_path / "backbone",
                download_seconds=1.0,
            ),
            PinnedRepository(
                repository="neuphonic/neucodec",
                revision="codec_pin",
                snapshot=tmp_path / "codec",
                download_seconds=1.0,
            ),
        ),
        auxiliary_repositories=(),
        sdk_speakers=("paul",),
        sdk_emotions=("neutral",),
    )
    return NeuTtsPilotConfig(
        manifest=manifest,
        manifest_sha256=digest(manifest),
        output_directory=tmp_path / "output",
        repository_directory=tmp_path / "repository",
        preparation=preparation,
        device=PilotDevice.CPU,
    )


@pytest.fixture
def case() -> TtsPilotCase:
    return TtsPilotCase(
        case_id="test_neutral",
        utterance_id="test",
        text="The meeting will begin at four tomorrow afternoon.",
        emotion=PilotEmotion.NEUTRAL,
        seed=42,
    )


def test_manifest_content_change_rejected(configuration: NeuTtsPilotConfig) -> None:
    validate_configuration(configuration)
    configuration.manifest.write_text("changed literal words", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest content"):
        validate_configuration(configuration)


def test_parameter_change_rejected(configuration: NeuTtsPilotConfig) -> None:
    validate_configuration(configuration)
    altered = NeuTtsPilotConfig(
        manifest=configuration.manifest,
        manifest_sha256=configuration.manifest_sha256,
        output_directory=configuration.output_directory,
        repository_directory=configuration.repository_directory,
        preparation=configuration.preparation,
        device=configuration.device,
        temperature=0.9,
    )
    with pytest.raises(ValueError, match="changed configuration"):
        validate_configuration(altered)


def test_saved_clip_reused_with_original_measurement(
    configuration: NeuTtsPilotConfig, case: TtsPilotCase
) -> None:
    waveform = np.linspace(-0.5, 0.5, 24000, dtype=np.float32)
    saved = persist_clip(configuration, case, waveform, 0.75, 24000)
    assert restore_clip(case, configuration.output_directory) == saved
    assert saved.generation_seconds == 0.75
    assert saved.real_time_factor == 0.75


def test_commit_interruption_recovers_partial_waveform(
    configuration: NeuTtsPilotConfig, case: TtsPilotCase
) -> None:
    waveform = np.zeros(24000, dtype=np.float32)
    saved = persist_clip(configuration, case, waveform, 0.5, 24000)
    path = configuration.output_directory / saved.audio_path
    partial = path.with_suffix(".wav.part")
    path.replace(partial)
    assert restore_clip(case, configuration.output_directory) == saved
    assert path.exists()
    assert not partial.exists()


def test_corrupt_audio_rejected(configuration: NeuTtsPilotConfig, case: TtsPilotCase) -> None:
    saved = persist_clip(configuration, case, np.zeros(24000, dtype=np.float32), 0.5, 24000)
    (configuration.output_directory / saved.audio_path).write_bytes(b"corrupt wave")
    with pytest.raises(ValueError, match="hash mismatch"):
        restore_clip(case, configuration.output_directory)
