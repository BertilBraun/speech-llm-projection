from pathlib import Path

import pytest

from scripts.teacher_asr_quality import aggregate, load_asr_snapshot, observations
from speech_projector.models import AsrTranscript, Example, Split


def example(identifier: str, text: str) -> Example:
    return Example(
        example_id=identifier,
        dialogue_id=identifier,
        split=Split.VALIDATION,
        history=(),
        user_text=text,
        target_text="Unused response",
        audio_path=Path(f"{identifier}.wav"),
        duration=1.0,
        domain="test",
        emotion="neutral",
        feature_path=Path(f"{identifier}.pt"),
    )


@pytest.mark.parametrize(
    ("reference", "recognized", "hits", "substitutions", "deletions", "insertions"),
    (
        ("Hello, WORLD!", "hello world", 2, 0, 0, 0),
        ("one two three", "one four three", 2, 1, 0, 0),
        ("one two three", "one three", 2, 0, 1, 0),
        ("one two", "one extra two", 2, 0, 0, 1),
        ("one two", "", 0, 0, 2, 0),
    ),
)
def test_lexical_alignment_counts(
    reference: str,
    recognized: str,
    hits: int,
    substitutions: int,
    deletions: int,
    insertions: int,
) -> None:
    record = observations(
        (example("selected", reference),),
        (AsrTranscript(example_id="selected", text=recognized),),
    )[0]
    assert record.edits.hits == hits
    assert record.edits.substitutions == substitutions
    assert record.edits.deletions == deletions
    assert record.edits.insertions == insertions
    assert record.word_error_rate == pytest.approx(
        (substitutions + deletions + insertions) / (hits + substitutions + deletions)
    )


def test_pooled_word_error_rate_uses_reference_word_weighting() -> None:
    records = observations(
        (example("short", "one"), example("long", "one two three four")),
        (
            AsrTranscript(example_id="short", text="wrong"),
            AsrTranscript(example_id="long", text="one two three four"),
        ),
    )
    metrics = aggregate(records)
    assert metrics.reference_words == 5
    assert metrics.word_error_rate == pytest.approx(0.2)
    assert metrics.example_word_error_rate.mean == pytest.approx(0.5)


def test_missing_selected_transcription_is_explicit_pending_error() -> None:
    with pytest.raises(ValueError, match="pending for 1 selected"):
        observations((example("missing", "reference"),), ())


def test_duplicate_journal_identifiers_reject_ambiguous_transcriptions() -> None:
    with pytest.raises(ValueError, match="duplicate identifiers"):
        observations(
            (example("selected", "reference"),),
            (
                AsrTranscript(example_id="selected", text="first"),
                AsrTranscript(example_id="selected", text="second"),
            ),
        )


def test_reused_unselected_asr_records_do_not_enter_heldout_metrics() -> None:
    records = observations(
        (example("selected", "reference"),),
        (
            AsrTranscript(example_id="selected", text="reference"),
            AsrTranscript(example_id="old-heldout", text="unused old record"),
        ),
    )
    assert tuple(record.example_id for record in records) == ("selected",)
    assert aggregate(records).word_error_rate == 0.0


def test_empty_normalized_synthesis_reference_is_invalid() -> None:
    with pytest.raises(ValueError, match="reference is empty"):
        observations(
            (example("selected", "...!"),),
            (AsrTranscript(example_id="selected", text="recognized"),),
        )


def test_asr_audit_rejects_partial_line_without_repairing_source(tmp_path: Path) -> None:
    path = tmp_path / "asr.jsonl"
    content = AsrTranscript(example_id="selected", text="recognized").model_dump_json().encode()
    path.write_bytes(content)
    with pytest.raises(ValueError, match="incomplete final line"):
        load_asr_snapshot(path)
    assert path.read_bytes() == content
    assert tuple(tmp_path.iterdir()) == (path,)
