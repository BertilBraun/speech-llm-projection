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
from speech_projector.models import (
    AsrTranscript,
    ChatPromptConfig,
    Example,
    Role,
    Split,
    SystemPromptConfig,
    Turn,
)
from speech_projector.overnight_data import OrdinaryExampleSource
from speech_projector.overnight_preparation import (
    QwenSourceConfig,
    artifact,
    filter_ordinary_prompt_collisions,
    load_records,
    qwen_rows,
    reuse_ordinary_asr,
    student_prompt_digest,
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


def test_ordinary_asr_reuses_recognized_text_under_namespaced_ids(tmp_path: Path) -> None:
    path = tmp_path / "asr.jsonl"
    rows = (
        AsrTranscript(example_id="first", text="Recognized original words."),
        AsrTranscript(example_id="unused", text="Outside chosen source."),
    )
    write_records(path, rows)
    before = path.read_bytes()
    sources = (
        OrdinaryExampleSource(
            example_id="ordinary:first",
            source_manifest=tmp_path / "source.jsonl",
            source_example_id="first",
        ),
    )
    assert reuse_ordinary_asr(path, sources) == (
        AsrTranscript(example_id="ordinary:first", text=rows[0].text),
    )
    assert path.read_bytes() == before


def test_prompt_overlap_filters_entire_train_dialogue_and_preserves_source_and_heldouts(
    tmp_path: Path,
) -> None:
    train = ordinary("train", Split.TRAIN, tmp_path / "train.wav").model_copy(
        update={"user_text": "Please explain the Roaring Twenties."}
    )
    other_train_turn = ordinary("other", Split.TRAIN, tmp_path / "other.wav").model_copy(
        update={"dialogue_id": train.dialogue_id}
    )
    heldout = ordinary("test", Split.TEST, tmp_path / "test.wav").model_copy(
        update={"user_text": " Please explain the roaring twenties. "}
    )
    validation = ordinary("validation", Split.VALIDATION, tmp_path / "validation.wav")
    retained = ordinary("retained", Split.TRAIN, tmp_path / "retained.wav")
    source = (train, other_train_turn, heldout, validation, retained)
    source_path = tmp_path / "source.jsonl"
    write_records(source_path, source)
    original_bytes = source_path.read_bytes()
    filtered = filter_ordinary_prompt_collisions(source)
    assert filtered.examples == (heldout, validation, retained)
    assert source_path.read_bytes() == original_bytes
    assert len(filtered.exclusions) == 1
    exclusion = filtered.exclusions[0]
    assert exclusion.dialogue_id == train.dialogue_id
    assert exclusion.example_ids == (train.example_id, other_train_turn.example_id)
    assert exclusion.overlaps[0].protected_split == Split.TEST
    assert exclusion.overlaps[0].protected_example_ids == (heldout.example_id,)
    assert exclusion.overlaps[0].prompt_sha256 == student_prompt_digest(train)
    assert filtered == filter_ordinary_prompt_collisions(source)
    audio = tuple(
        FileArtifact(
            path=row.audio_path, source_path=row.audio_path, bytes=1, sha256=row.example_id
        )
        for row in filtered.examples
    )
    validate_boundaries(filtered.examples, audio)


def test_prompt_filter_uses_last_two_turns_and_generic_policy_only(tmp_path: Path) -> None:
    suffix = (
        Turn(role=Role.USER, text="Earlier user."),
        Turn(role=Role.ASSISTANT, text="Earlier reply."),
    )
    train = ordinary("train", Split.TRAIN, tmp_path / "train.wav").model_copy(
        update={
            "history": (Turn(role=Role.USER, text="Different older train history."),) + suffix,
            "user_text": "Same current words.",
        }
    )
    heldout = ordinary("test", Split.TEST, tmp_path / "test.wav").model_copy(
        update={
            "history": (Turn(role=Role.USER, text="Different older test history."),) + suffix,
            "user_text": train.user_text,
        }
    )
    assert filter_ordinary_prompt_collisions((train, heldout)).examples == (heldout,)
    other_policy = train.model_copy(
        update={"prompt": SystemPromptConfig(system_text="Other policy")}
    )
    assert filter_ordinary_prompt_collisions((other_policy, heldout)).examples == (
        other_policy,
        heldout,
    )


def test_prompt_filter_keeps_heldout_and_dialogue_leakage_guards_strict(tmp_path: Path) -> None:
    validation = ordinary("validation", Split.VALIDATION, tmp_path / "validation.wav")
    test = ordinary("test", Split.TEST, tmp_path / "test.wav").model_copy(
        update={"user_text": validation.user_text}
    )
    with pytest.raises(ValueError, match="heldout student prompt crosses splits"):
        filter_ordinary_prompt_collisions((validation, test))
    train = ordinary("train", Split.TRAIN, tmp_path / "train.wav").model_copy(
        update={"dialogue_id": validation.dialogue_id}
    )
    with pytest.raises(ValueError, match="conversation crosses splits"):
        filter_ordinary_prompt_collisions((validation, train))
