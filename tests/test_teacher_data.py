from pathlib import Path

import pytest

from scripts.audit_synthesis_alignment import SynthesisAlignment, compare, source_example_id
from scripts.prepare_teacher_data import (
    TeacherDataConfig,
    audit,
    clean_candidates,
    input_readiness,
    prepared_example,
    preserve_asr,
    select_heldout,
    write_feature_plan,
    write_immutable,
)
from speech_projector.cache import load_asr
from speech_projector.data import SourceSpeaker, SourceTurn
from speech_projector.models import AsrTranscript, Example, Role, Split, Turn


def aligned(index: int, split: Split = Split.TRAIN) -> SynthesisAlignment:
    source = SourceTurn(
        conversation_id=f"{split.value}-{index}",
        model_dir="model",
        turn_index=0,
        speaker=SourceSpeaker.FIRST,
        text=f"Question number {index}!",
        domain="test",
        emotion="neutral",
        segment_audio_path=f"{split.value}-{index}.wav",
        audio_duration=2.0,
        audio_original_text=f"Question number {index}!",
        audio_substituted_text=f"Question number {index}!",
        audio_cleaned_text=f"Question number {index}.",
    )
    example = Example(
        example_id=source_example_id(source),
        dialogue_id=source.dialogue_id,
        split=split,
        history=(Turn(role=Role.ASSISTANT, text="Previous reply"),),
        user_text=source.text,
        target_text=f"Original assistant response {index}",
        audio_path=Path(f"missing/{split.value}-{index}.wav"),
        duration=2.0,
        domain="test",
        emotion="neutral",
        feature_path=Path(f"missing/{split.value}-{index}.pt"),
    )
    return compare(example, source, index)


def test_clean_prefix_removes_mismatched_and_substituted_audio() -> None:
    records = tuple(aligned(index) for index in range(5))
    sources = {record.example.example_id: record.source for record in records}
    sources[records[1].example.example_id] = records[1].source.model_copy(
        update={"audio_original_text": "Wrong assistant response"}
    )
    sources[records[3].example.example_id] = records[3].source.model_copy(
        update={"audio_substituted_text": "Lexically substituted response"}
    )
    clean = clean_candidates(tuple(record.example for record in records), sources)
    assert tuple(record.example for record in clean.alignments) == tuple(
        records[index].example for index in (0, 2, 4)
    )
    assert clean.material_exclusions == 1
    assert clean.substitution_exclusions == 2


def test_cleaned_transcript_preserves_original_supervision_and_existing_cache(
    tmp_path: Path,
) -> None:
    original = aligned(1)
    audio = tmp_path / "existing.wav"
    feature = tmp_path / "existing.pt"
    audio.write_bytes(b"existing audio")
    feature.write_bytes(b"existing features")
    example = original.example.model_copy(update={"audio_path": audio, "feature_path": feature})
    record = original.model_copy(update={"example": example})
    prepared = prepared_example(record, tmp_path / "new")
    assert prepared.user_text == original.cleaned_text
    assert prepared.target_text == example.target_text
    assert prepared.history == example.history
    assert prepared.audio_path == audio
    assert prepared.feature_path == feature
    assert record.example.user_text == original.source.text


def test_heldout_pair_collision_removes_complete_dialogue_and_replenishes(tmp_path: Path) -> None:
    training = prepared_example(aligned(0), tmp_path)
    candidates = tuple(aligned(index, Split.VALIDATION) for index in range(4))
    collision = candidates[0].model_copy(
        update={
            "source": candidates[0].source.model_copy(
                update={"audio_cleaned_text": training.user_text}
            ),
            "example": candidates[0].example.model_copy(
                update={"target_text": training.target_text}
            ),
        }
    )
    same_dialogue = candidates[1].model_copy(
        update={
            "example": candidates[1].example.model_copy(
                update={"dialogue_id": collision.example.dialogue_id}
            )
        }
    )
    selected, excluded = select_heldout(
        (collision, same_dialogue, candidates[2], candidates[3]), (training,), 2, tmp_path
    )
    assert selected == candidates[2:]
    assert excluded == (collision.example.dialogue_id,)


@pytest.mark.parametrize("failure", ["missing_audio", "identical_audio"])
def test_waveform_leakage_blocks_dataset_release(tmp_path: Path, failure: str) -> None:
    examples: list[Example] = []
    for index, split in enumerate(Split):
        example = prepared_example(aligned(index, split), tmp_path)
        example.audio_path.parent.mkdir(parents=True, exist_ok=True)
        if failure == "identical_audio" or split != Split.TEST:
            example.audio_path.write_bytes(
                b"same waveform" if failure == "identical_audio" else split.value.encode()
            )
        examples.append(example)
    with pytest.raises(ValueError, match="missing|leakage"):
        audit(examples, tmp_path)


def test_source_provenance_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "provenance.jsonl"
    write_immutable(path, b"source target and transcript")
    write_immutable(path, b"source target and transcript")
    with pytest.raises(ValueError, match="Refusing to change"):
        write_immutable(path, b"different corpus")


def test_cache_plan_includes_only_missing_features(tmp_path: Path) -> None:
    examples = [prepared_example(aligned(index), tmp_path) for index in range(3)]
    examples[1].feature_path.parent.mkdir(parents=True, exist_ok=True)
    examples[1].feature_path.write_bytes(b"cached features")
    manifest = tmp_path / "examples_source.jsonl"
    manifest.write_bytes(b"canonical source manifest")
    configuration = TeacherDataConfig(original_root=tmp_path / "original", output_root=tmp_path)
    plan = write_feature_plan(examples, configuration, manifest)
    assert plan.existing_features == 1
    assert plan.pending_example_ids == (examples[0].example_id, examples[2].example_id)
    assert plan.pending_audio_seconds == 4.0
    assert examples[1].feature_path.read_bytes() == b"cached features"


def test_asr_reuse_is_filtered_by_selected_heldout_and_idempotent(tmp_path: Path) -> None:
    original_root = tmp_path / "original"
    original_root.mkdir()
    configuration = TeacherDataConfig(original_root=original_root, output_root=tmp_path)
    examples = tuple(
        prepared_example(aligned(index, split), tmp_path) for index, split in enumerate(Split)
    )
    records = tuple(
        AsrTranscript(example_id=example.example_id, text="Existing transcription")
        for example in examples
    )
    (original_root / "asr_transcripts.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8"
    )
    preserve_asr(examples, configuration)
    preserve_asr(examples, configuration)
    assert load_asr(tmp_path / "asr_transcripts.jsonl") == list(records[1:])


def test_transcript_overlap_is_reported_separately_from_response_pair(tmp_path: Path) -> None:
    training = prepared_example(aligned(0), tmp_path)
    validation = prepared_example(aligned(1, Split.VALIDATION), tmp_path).model_copy(
        update={"user_text": training.user_text}
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_bytes(b"input manifest")
    report = input_readiness((training, validation), manifest)
    assert report.cleaned_user_transcript_overlap[0].shared_transcripts == 1
    assert report.cleaned_user_transcript_overlap[0].left_examples_with_overlap == 1
    assert report.bootstrap[0].missing_features == 1
