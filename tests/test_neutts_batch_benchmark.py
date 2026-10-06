"""CPU checks for EOS evidence and batch-level throughput accounting."""

from pathlib import Path

import pytest

from scripts.neutts_pilot_state import NeuTtsPilotConfig, PilotDevice, digest
from scripts.prepare_neutts_models import NeuTtsPreparation, PinnedRepository
from speech_projector.neutts_batch_benchmark import (
    BatchClipEvidence,
    BatchMeasurement,
    NeuTtsBenchmarkConfig,
    completed_tokens,
    prompt_emotions,
)
from speech_projector.tts_pilot import PilotEmotion, PilotTermination, TtsPilotCase, TtsPilotClip


@pytest.mark.parametrize(
    ("tokens", "expected", "termination"),
    (
        ((10, 11, 99, 0, 0), (10, 11, 99), PilotTermination.STOP),
        ((10, 11), (10, 11), PilotTermination.TOKEN_LIMIT),
        ((99, 0), (99,), PilotTermination.STOP),
    ),
)
def test_completed_tokens_retains_eos(
    tokens: tuple[int, ...], expected: tuple[int, ...], termination: PilotTermination
) -> None:
    assert completed_tokens(tokens, 99) == (expected, termination)


def test_prompt_control_uses_provider_neutral_normalization() -> None:
    checked: list[str] = []

    def sdk_checker(emotion: str) -> str | None:
        checked.append(emotion)
        return None if emotion == "neutral" else emotion

    cases = tuple(
        TtsPilotCase(
            case_id=emotion.value,
            utterance_id="same",
            text="The same literal sentence.",
            emotion=emotion,
            seed=42,
        )
        for emotion in (PilotEmotion.NEUTRAL, PilotEmotion.HAPPY, PilotEmotion.SAD)
    )
    assert prompt_emotions(cases, sdk_checker) == (None, "happy", "sad")
    assert checked == ["neutral", "happy", "sad"]


def test_batch_aggregate_uses_wall_time_once() -> None:
    cases = tuple(
        TtsPilotCase(
            case_id=f"case_{emotion.value}",
            utterance_id="same",
            text="The same literal sentence.",
            emotion=emotion,
            seed=42,
        )
        for emotion in (PilotEmotion.HAPPY, PilotEmotion.SAD)
    )
    clips = tuple(
        BatchClipEvidence(
            clip=TtsPilotClip(
                case=case,
                audio_path=f"audio/{case.case_id}.wav",
                sha256="test",
                sample_rate=24000,
                audio_seconds=4,
                generation_seconds=2,
                real_time_factor=0.5,
                termination=PilotTermination.STOP,
            ),
            prompt_tokens=100,
            speech_tokens=200,
            generated_token_ids=(1000, 99),
        )
        for case in cases
    )
    measurement = BatchMeasurement(
        requested_batch_size=2,
        repetition=0,
        batch_index=0,
        shared_batch_seed=42,
        padded_prompt_width=100,
        generation_token_cap=2000,
        frontend_seconds=0.1,
        backbone_seconds=1.5,
        codec_watermark_seconds=0.4,
        end_to_end_seconds=2,
        clips=clips,
    )
    assert measurement.audio_seconds == 8
    assert measurement.aggregate_real_time_factor == 0.25
    assert measurement.end_to_end_seconds != sum(item.clip.generation_seconds for item in clips)


@pytest.mark.parametrize("batch_sizes", ((), (0,), (8,), (1, 1)))
def test_invalid_batch_sizes_rejected(tmp_path: Path, batch_sizes: tuple[int, ...]) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"test")
    repositories = tuple(
        PinnedRepository(
            repository=name, revision="fixed", snapshot=tmp_path / name, download_seconds=None
        )
        for name in ("neuphonic/neutts-2e", "neuphonic/neucodec")
    )
    preparation = NeuTtsPreparation(
        source_commit="fixed-source",
        repositories=(repositories[0], repositories[1]),
        auxiliary_repositories=(),
        sdk_speakers=("paul",),
        sdk_emotions=("happy", "sad"),
    )
    pilot = NeuTtsPilotConfig(
        manifest=manifest,
        manifest_sha256=digest(manifest),
        output_directory=tmp_path / "output",
        repository_directory=tmp_path / "repository",
        preparation=preparation,
        device=PilotDevice.CUDA,
    )
    with pytest.raises(ValueError, match="Batch sizes|batch sizes"):
        NeuTtsBenchmarkConfig(pilot=pilot, source_commit="test", batch_sizes=batch_sizes)
