"""Durable typed JSONL records with archived recovery of an uncommitted final line."""

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from pydantic import TypeAdapter

from speech_projector.models import Record

JournalRecord = TypeVar("JournalRecord", bound=Record)


class JournalRecovery(Record):
    journal: Path
    archived_suffix: Path
    recovered_at: datetime
    removed_incomplete_bytes: int
    suffix_sha256: str


def append_record(path: Path, record: Record) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(record.model_dump_json() + "\n")
        output.flush()
        os.fsync(output.fileno())


def read_journal(
    path: Path, record_type: type[JournalRecord] | TypeAdapter[JournalRecord]
) -> tuple[JournalRecord, ...]:
    if not path.exists():
        return ()
    match record_type:
        case TypeAdapter():
            adapter = record_type
        case _:
            adapter = TypeAdapter(record_type)
    content = path.read_bytes()
    complete = content.rfind(b"\n") + 1
    records = tuple(adapter.validate_json(line) for line in content[:complete].splitlines())
    if complete == len(content):
        return records
    suffix = content[complete:]
    digest = hashlib.sha256(suffix).hexdigest()
    archive = path.with_name(f"{path.name}.incomplete-{digest}.bin")
    if not archive.exists():
        with archive.open("xb") as output:
            output.write(suffix)
            output.flush()
            os.fsync(output.fileno())
    with path.open("r+b") as output:
        output.truncate(complete)
        output.flush()
        os.fsync(output.fileno())
    append_record(
        path.parent / "journal_recovery.jsonl",
        JournalRecovery(
            journal=path,
            archived_suffix=archive,
            recovered_at=datetime.now(timezone.utc),
            removed_incomplete_bytes=len(suffix),
            suffix_sha256=digest,
        ),
    )
    return records
