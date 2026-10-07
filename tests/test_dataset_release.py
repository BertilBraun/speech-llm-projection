"""Dataset release checks preserve words, family splits and original WAV bytes."""

import hashlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
import pytest
import soundfile

from scripts.inventory_results import write_record
from speech_projector.dataset_release import (
    DatasetExportConfig,
    DatasetExportReceipt,
    ExportAudio,
    QwenCorpusSource,
    ReleasedExample,
    read_release_records,
    safe_audio_path,
    validate_pairs,
    verify_dataset_export,
    waveform_bytes,
    write_shard,
)
from speech_projector.emotional_dataset import Domain, Intent
from speech_projector.models import Split
from speech_projector.tts_pilot import PilotEmotion


def paired_rows(directory: Path) -> tuple[ExportAudio, ...]:
    rows: list[ExportAudio] = []
    for emotion in (PilotEmotion.HAPPY, PilotEmotion.ANGRY):
        path = directory / f"one_{emotion.value}.wav"
        soundfile.write(path, np.full(2400, 1.01, dtype=np.float32), 24000, subtype="FLOAT")
        rows.append(
            ExportAudio(
                ReleasedExample(
                    case_id=path.stem,
                    utterance_id="one",
                    family_id="family_one",
                    split=Split.TRAIN,
                    domain=Domain.HOME,
                    intent=Intent.GIVE_UPDATE,
                    text="The appointment reply arrived in the mail today.",
                    delivery=emotion,
                    teacher_response=f"An appropriate reply for {emotion.value}.",
                    audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    sample_rate=24000,
                    duration_seconds=0.1,
                ),
                path,
            )
        )
    return tuple(rows)


def test_parquet_preserves_exact_float_wav_and_pair_metadata(tmp_path: Path) -> None:
    rows = paired_rows(tmp_path)
    assert validate_pairs(rows, 1) == 0
    destination = tmp_path / "train-00000.parquet"
    shard = write_shard(destination, rows, Split.TRAIN)
    table = parquet.read_table(destination)
    assert table.num_rows == shard.examples == 2
    assert table.column("delivery").to_pylist() == ["happy", "angry"]
    assert table.column("text").to_pylist() == [row.example.text for row in rows]
    for index, row in enumerate(rows):
        assert table.column("audio")[index].as_py()["bytes"] == row.waveform_path.read_bytes()
    assert shard.sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "replacement",
    (
        {"delivery": PilotEmotion.HAPPY},
        {"text": "A different sentence."},
        {"split": Split.TEST},
        {"family_id": "another_family"},
        {"domain": Domain.WORK},
        {"teacher_response": " "},
    ),
)
def test_mismatched_pairs_are_rejected(tmp_path: Path, replacement: dict[str, str]) -> None:
    first, second = paired_rows(tmp_path)
    altered = ExportAudio(second.example.model_copy(update=replacement), second.waveform_path)
    with pytest.raises(ValueError):
        validate_pairs((first, altered), 1)


def test_family_split_leakage_across_different_utterances_is_rejected(tmp_path: Path) -> None:
    first, second = paired_rows(tmp_path)
    additional = tuple(
        ExportAudio(
            row.example.model_copy(
                update={
                    "case_id": f"other_{row.example.delivery.value}",
                    "utterance_id": "other",
                    "split": Split.TEST,
                }
            ),
            row.waveform_path,
        )
        for row in (first, second)
    )
    with pytest.raises(ValueError, match="family crosses"):
        validate_pairs((first, second, *additional), 2)


def test_missing_delivery_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="every configured"):
        validate_pairs(paired_rows(tmp_path)[:1], 1)


def test_audio_corruption_is_rejected(tmp_path: Path) -> None:
    row = paired_rows(tmp_path)[0]
    row.waveform_path.write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="WAV hash changed"):
        waveform_bytes(row)


def test_audio_path_cannot_escape_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="inside its source"):
        safe_audio_path(source, Path("../outside.wav"))


def test_export_verification_reads_embedded_wavs_and_detects_changed_shard(tmp_path: Path) -> None:
    rows = paired_rows(tmp_path)
    output = tmp_path / "release"
    configuration = DatasetExportConfig(
        source=QwenCorpusSource(directory=tmp_path / "immutable"),
        output_directory=output,
        expected_utterances=1,
    )
    shard = write_shard(output / "data" / "train-00000.parquet", rows, Split.TRAIN)
    (output / "examples.jsonl").write_text(
        "".join(row.example.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )
    receipt = DatasetExportReceipt(
        configuration=configuration,
        utterances=1,
        examples=2,
        identical_teacher_pairs=0,
        train_examples=2,
        validation_examples=0,
        test_examples=0,
        audio_seconds=0.2,
        source_audio_bytes=sum(row.waveform_path.stat().st_size for row in rows),
        shards=(shard,),
    )
    write_record(output / "export_receipt.json", receipt)
    verified = verify_dataset_export(configuration)
    assert verified.examples == 2
    assert verified.audio_bytes == receipt.source_audio_bytes
    (output / shard.path).write_bytes(b"corrupted shard")
    with pytest.raises(ValueError, match="shard hash changed"):
        verify_dataset_export(configuration)


def test_incomplete_source_record_fails_without_recovering_or_modifying_it(tmp_path: Path) -> None:
    row = paired_rows(tmp_path)[0].example
    content = row.model_dump_json().encode() + b'\n{"unfinished":'
    path = tmp_path / "source.jsonl"
    path.write_bytes(content)
    with pytest.raises(ValueError):
        read_release_records(path, ReleasedExample)
    assert path.read_bytes() == content
    assert not (tmp_path / "journal_recovery.jsonl").exists()
