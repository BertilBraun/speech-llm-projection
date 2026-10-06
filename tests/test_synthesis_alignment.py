from pathlib import Path

import pytest

from scripts.audit_synthesis_alignment import AlignmentKind, compare, difference, summarize
from speech_projector.data import SourceSpeaker, SourceTurn
from speech_projector.models import Example, Split


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("Same words", "Same words", AlignmentKind.EXACT),
        ("Same words?! 😄", "same words", AlignmentKind.FORMATTING),
        ("The user asks", "The assistant answers", AlignmentKind.LEXICAL),
        ("Ushuaïa", "Ushua a", AlignmentKind.LEXICAL),
        ("LLM1 answers", "Speaker Actor_15 answers", AlignmentKind.LEXICAL),
    ],
)
def test_difference_preserves_lexical_changes(
    left: str, right: str, expected: AlignmentKind
) -> None:
    assert difference(left, right) == expected


def example_and_source() -> tuple[Example, SourceTurn]:
    example = Example(
        example_id="example",
        dialogue_id="models/dialogue",
        split=Split.TRAIN,
        history=(),
        user_text="The user asks.",
        target_text="The assistant answers.",
        audio_path=Path("audio.wav"),
        duration=2.0,
        domain="test",
        emotion="neutral",
        feature_path=Path("feature.pt"),
    )
    source = SourceTurn(
        conversation_id="dialogue",
        model_dir="models",
        turn_index=0,
        speaker=SourceSpeaker.FIRST,
        text=example.user_text,
        domain="test",
        emotion="neutral",
        segment_audio_path="audio.wav",
        audio_duration=2.0,
        audio_original_text=example.target_text,
        audio_substituted_text=example.target_text,
        audio_cleaned_text=example.target_text,
    )
    return example, source


def test_shifted_audio_is_counted_as_target_leakage() -> None:
    example, source = example_and_source()
    alignment = compare(example, source, 7)
    summary = summarize(Split.TRAIN, (alignment,))
    assert alignment.turn_vs_audio_original == AlignmentKind.LEXICAL
    assert alignment.audio_original_vs_target == AlignmentKind.EXACT
    assert alignment.cleaned_text == example.target_text
    assert alignment.split_index == 7
    assert summary.turn_original_lexical_differences == 1
    assert summary.mismatched_audio_matches_assistant_target == 1


@pytest.mark.parametrize("invalid_source", ["missing_text", "wrong_turn"])
def test_alignment_rejects_unverifiable_source(invalid_source: str) -> None:
    example, source = example_and_source()
    match invalid_source:
        case "missing_text":
            source = source.model_copy(update={"audio_cleaned_text": None})
        case "wrong_turn":
            source = source.model_copy(update={"text": "A different dialogue turn."})
    with pytest.raises(ValueError):
        compare(example, source, 0)
