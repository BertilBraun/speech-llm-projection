from pathlib import Path

import numpy as np
import pytest
import soundfile

from scripts.package_results import FileArtifact
from speech_projector.emotion_preview import Delivery, PreviewCase, PreviewClip
from speech_projector.emotional_dataset import (
    DeliveryAnnotation,
    Domain,
    EmotionalUtterance,
    Intent,
)
from speech_projector.emotional_generation import (
    EmotionalGenerationConfig,
    EmotionalTeacherRequest,
    EmotionalTeacherTarget,
)
from speech_projector.generation import CompletedGeneration
from speech_projector.models import ChatPromptConfig, Example, Split, SystemPromptConfig
from speech_projector.overnight_preparation import (
    QwenSourceConfig,
    artifact,
    load_records,
    qwen_rows,
    validate_boundaries,
    write_immutable,
    write_records,
)


def qwen_source(directory: Path, duration: float) -> QwenSourceConfig:
    configuration = EmotionalGenerationConfig(
        output_directory=directory,
        revision="a" * 40,
        source_git_commit="b" * 40,
        teacher_system="Actual immutable generic teacher policy.",
    )
    write_immutable(directory / "generation.json", configuration.model_dump_json().encode())
    utterance = EmotionalUtterance(
        base_id="base",
        family_id="family",
        split=Split.TRAIN,
        domain=Domain.HOME,
        intent=Intent.GIVE_UPDATE,
        text="The identical literal words across both deliveries.",
        deliveries=(
            DeliveryAnnotation(delivery=Delivery.HAPPY, instruct="Happy audio instruction."),
            DeliveryAnnotation(delivery=Delivery.SAD, instruct="Sad audio instruction."),
        ),
    )
    targets = []
    for index, annotation in enumerate(utterance.deliveries):
        targets.append(
            EmotionalTeacherTarget(
                request=EmotionalTeacherRequest(utterance=utterance, annotation=annotation),
                response=CompletedGeneration(text=f"Teacher response {index}", token_ids=(1, 2)),
                capped_attempts=(),
            )
        )
        case = PreviewCase(
            case_id=f"base_{annotation.delivery.value}",
            text=utterance.text,
            delivery=annotation.delivery,
            instruct=annotation.instruct,
            seed=42,
        )
        path = directory / "audio" / f"{case.case_id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = int(duration * 100)
        soundfile.write(path, np.full(samples, 0.1 + index * 0.1, dtype=np.float32), 100)
        audio = artifact(path).model_copy(update={"path": path.relative_to(directory)})
        clip = PreviewClip(
            case=case,
            audio=audio,
            sample_rate=100,
            samples=samples,
            duration_seconds=duration,
            runtime_seconds=1,
            codec_tokens=12,
            peak_amplitude=0.2,
        )
        write_immutable(
            directory / "clips" / f"{case.case_id}.json", clip.model_dump_json().encode()
        )
    write_records(directory / "targets.jsonl", targets)
    return QwenSourceConfig(
        targets=directory / "targets.jsonl",
        generation_configuration=directory / "generation.json",
        audio_directory=directory,
        expected_pairs=1,
    )


def test_qwen_pair_derives_generic_prompt_and_reuses_exact_wav_bytes(tmp_path: Path) -> None:
    config = qwen_source(tmp_path / "old", 2)
    rows = qwen_rows(config, tmp_path / "derived")
    assert len(rows.examples) == len(rows.sources) == 2
    assert rows.exclusions == ()
    for example, source in zip(rows.examples, rows.sources, strict=True):
        assert example.prompt == SystemPromptConfig(
            system_text="Actual immutable generic teacher policy."
        )
        assert example.history == ()
        assert "tone" not in example.prompt.system_text
        assert example.audio_path.is_file()
        assert source.base_id == "base"
        assert example.feature_path.name == artifact(example.audio_path).sha256 + ".pt"


def test_overlong_audio_excludes_entire_pair_without_mutating_source(tmp_path: Path) -> None:
    config = qwen_source(tmp_path / "old", 31)
    before = tuple(artifact(path) for path in sorted(config.audio_directory.rglob("*.wav")))
    rows = qwen_rows(config, tmp_path / "derived")
    assert rows.examples == rows.sources == ()
    assert len(rows.audio) == 2
    assert len(rows.exclusions) == 1
    assert rows.exclusions[0].durations_seconds == (31, 31)
    assert before == tuple(artifact(path) for path in sorted(config.audio_directory.rglob("*.wav")))


def ordinary(identifier: str, split: Split, path: Path) -> Example:
    return Example(
        example_id=identifier,
        dialogue_id=identifier,
        split=split,
        history=(),
        user_text=identifier,
        target_text="A complete response.",
        audio_path=path,
        duration=1,
        domain="test",
        emotion="",
        feature_path=path.with_suffix(".pt"),
        prompt=ChatPromptConfig(),
    )


@pytest.mark.parametrize("collision", ("dialogue", "prompt", "audio"))
def test_split_boundary_rejects_actual_family_prompt_and_audio_collisions(
    tmp_path: Path, collision: str
) -> None:
    first = ordinary("first", Split.TRAIN, tmp_path / "first.wav")
    second = ordinary("second", Split.VALIDATION, tmp_path / "second.wav")
    match collision:
        case "dialogue":
            second = second.model_copy(update={"dialogue_id": first.dialogue_id})
        case "prompt":
            second = second.model_copy(update={"user_text": " FIRST  "})
    audio = (
        FileArtifact(path=first.audio_path, source_path=first.audio_path, bytes=1, sha256="a" * 64),
        FileArtifact(
            path=second.audio_path,
            source_path=second.audio_path,
            bytes=1,
            sha256=("a" if collision == "audio" else "b") * 64,
        ),
    )
    with pytest.raises(ValueError, match="crosses splits"):
        validate_boundaries((first, second), audio)


def test_source_journal_and_derived_manifest_are_immutable(tmp_path: Path) -> None:
    path = tmp_path / "source.jsonl"
    row = ordinary("first", Split.TRAIN, tmp_path / "first.wav")
    write_records(path, (row,))
    assert load_records(path, Example) == (row,)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        write_records(path, (row.model_copy(update={"target_text": "Changed"}),))
    assert path.read_bytes() == before
    path.write_bytes(before[:-1])
    with pytest.raises(ValueError, match="incomplete"):
        load_records(path, Example)
    assert path.read_bytes() == before[:-1]
