from pathlib import Path

import pytest

from scripts.audit_synthesis_alignment import SynthesisAlignment, compare, source_example_id
from scripts.prepare_clean_v0 import (
    clean_heldout,
    require_cached_features,
    select_clean_prefix,
    verify_alignments,
    write_immutable,
)
from speech_projector.data import SourceSpeaker, SourceTurn
from speech_projector.models import Example, Role, Split, Turn


def audited_example(index: int, split: Split, *, mismatch: bool) -> SynthesisAlignment:
    source = SourceTurn(
        conversation_id=f"{split.value}-{index}",
        model_dir="model",
        turn_index=0,
        speaker=SourceSpeaker.FIRST,
        text=f"User asks question {index}.",
        domain="test",
        emotion="neutral",
        segment_audio_path=f"audio-{index}.wav",
        audio_duration=2.0,
        audio_original_text="Assistant response" if mismatch else f"User asks question {index}.",
        audio_substituted_text="Assistant response" if mismatch else f"User asks question {index}.",
        audio_cleaned_text="Assistant response" if mismatch else f"User asks question {index}.",
    )
    example = Example(
        example_id=source_example_id(source),
        dialogue_id=source.dialogue_id,
        split=split,
        history=(Turn(role=Role.ASSISTANT, text="Previous history"),),
        user_text=source.text,
        target_text="Assistant response",
        audio_path=Path(f"audio-{index}.wav"),
        duration=2.0,
        domain="test",
        emotion="neutral",
        feature_path=Path(f"feature-{index}.pt"),
    )
    return compare(example, source, index)


def test_clean_training_preserves_order_history_and_targets() -> None:
    records = tuple(
        audited_example(index, Split.TRAIN, mismatch=index in (1, 4)) for index in range(8)
    )
    examples = tuple(record.example for record in records)
    selected, indices = select_clean_prefix(examples, (records[1], records[4]), 5)
    assert indices == (0, 2, 3, 5, 6)
    assert selected == tuple(examples[index] for index in indices)


@pytest.mark.parametrize(
    "failure", ["wrong_position", "duplicate_index", "non_material", "too_few"]
)
def test_training_exclusions_fail_when_unverifiable(failure: str) -> None:
    records = tuple(audited_example(index, Split.TRAIN, mismatch=index == 1) for index in range(3))
    examples = tuple(record.example for record in records)
    match failure:
        case "wrong_position":
            exclusions = (records[1].model_copy(update={"split_index": 0}),)
            count = 1
        case "duplicate_index":
            exclusions = (records[1], records[1])
            count = 1
        case "non_material":
            exclusions = (records[0],)
            count = 1
        case "too_few":
            exclusions = (records[1],)
            count = 3
    with pytest.raises(ValueError):
        select_clean_prefix(examples, exclusions, count)


def test_fixed_heldout_only_removes_material_mismatches() -> None:
    records = tuple(audited_example(index, Split.TEST, mismatch=index == 1) for index in range(4))
    examples = tuple(record.example for record in records)
    selected, excluded = clean_heldout(examples, records)
    assert selected == (examples[0], examples[2], examples[3])
    assert excluded == (examples[1].example_id,)
    with pytest.raises(ValueError, match="complete fixed split"):
        verify_alignments(examples, records[:-1], complete=True)


def test_feature_cache_missing_fails_without_creating_features(tmp_path: Path) -> None:
    example = audited_example(0, Split.TRAIN, mismatch=False).example
    feature_path = tmp_path / "feature.pt"
    example = example.model_copy(update={"feature_path": feature_path})
    with pytest.raises(ValueError, match="cached Whisper feature is missing"):
        require_cached_features((example,))
    assert not feature_path.exists()
    feature_path.write_bytes(b"existing cached tensor")
    require_cached_features((example,))
    assert feature_path.read_bytes() == b"existing cached tensor"


def test_control_inputs_are_idempotent_and_cannot_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "new-control" / "examples.jsonl"
    write_immutable(path, b"original control")
    write_immutable(path, b"original control")
    with pytest.raises(ValueError, match="Refusing to replace"):
        write_immutable(path, b"a different control")
    assert path.read_bytes() == b"original control"
