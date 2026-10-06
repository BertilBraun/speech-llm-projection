from pathlib import Path

import pytest
from pydantic import ValidationError

from speech_projector.journal import JournalRecovery, append_record, read_journal
from speech_projector.models import Record


class JournalEntry(Record):
    value: int


def test_journal_recovers_only_suffix_and_archives_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    append_record(path, JournalEntry(value=4))
    with path.open("ab") as handle:
        handle.write(b'{"value":')
    assert read_journal(path, JournalEntry) == (JournalEntry(value=4),)
    recoveries = read_journal(tmp_path / "journal_recovery.jsonl", JournalRecovery)
    assert len(recoveries) == 1
    assert recoveries[0].archived_suffix.read_bytes() == b'{"value":'
    assert path.read_bytes().endswith(b"\n")
    append_record(path, JournalEntry(value=5))
    assert read_journal(path, JournalEntry) == (JournalEntry(value=4), JournalEntry(value=5))


def test_complete_invalid_record_is_not_silently_recovered(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    original = b'{"value":"wrong"}\n{"value":'
    path.write_bytes(original)
    with pytest.raises(ValidationError):
        read_journal(path, JournalEntry)
    assert path.read_bytes() == original
    assert not tuple(tmp_path.glob("*.bin"))
