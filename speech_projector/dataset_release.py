"""Export existing paired corpora as portable, byte-preserving Hugging Face Parquet."""

import hashlib
import io
import json
import shutil
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, TypeVar, cast

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import soundfile
from pydantic import Field, PlainSerializer

from scripts.inventory_results import stable_digest, write_record
from speech_projector.emotion_preview import Delivery, PreviewClip
from speech_projector.emotional_audio import CodecTermination
from speech_projector.emotional_dataset import Domain, EmotionalUtterance, Intent
from speech_projector.emotional_generation import EmotionalTeacherTarget
from speech_projector.models import Record, Split
from speech_projector.neu_dataset import NeuUtterance
from speech_projector.neu_generation import NeuTeacherTarget
from speech_projector.neutts_batch_benchmark import BatchClipEvidence
from speech_projector.neutts_corpus import NeuCorpusResult
from speech_projector.tts_pilot import PilotEmotion, PilotTermination


def portable_path(path: Path) -> str:
    return path.as_posix()


PortablePath = Annotated[Path, PlainSerializer(portable_path, return_type=str)]
ReleaseRecord = TypeVar("ReleaseRecord", bound=Record)


def read_release_records(path: Path, record_type: type[ReleaseRecord]) -> tuple[ReleaseRecord, ...]:
    return tuple(record_type.model_validate_json(line) for line in path.read_bytes().splitlines())


class QwenCorpusSource(Record):
    kind: Literal["qwen"] = "qwen"
    directory: PortablePath


class NeuCorpusSource(Record):
    kind: Literal["neu"] = "neu"
    directory: PortablePath


CorpusSource = Annotated[QwenCorpusSource | NeuCorpusSource, Field(discriminator="kind")]


class DatasetExportConfig(Record):
    source: CorpusSource
    output_directory: PortablePath
    expected_utterances: int = Field(default=5000, gt=0)
    rows_per_shard: int = Field(default=250, ge=2)


class ReleasedExample(Record):
    case_id: str
    utterance_id: str
    family_id: str
    split: Split
    domain: Domain
    intent: Intent
    text: str
    delivery: Delivery | PilotEmotion
    teacher_response: str
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_rate: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)


@dataclass(frozen=True)
class ExportAudio:
    example: ReleasedExample
    waveform_path: Path


class ExportShard(Record):
    path: PortablePath
    split: Split
    examples: int
    bytes: int
    sha256: str


class DatasetExportReceipt(Record):
    configuration: DatasetExportConfig
    utterances: int
    examples: int
    identical_teacher_pairs: int
    train_examples: int
    validation_examples: int
    test_examples: int
    audio_seconds: float
    source_audio_bytes: int
    shards: tuple[ExportShard, ...]


class DatasetExportVerification(Record):
    verified_at: datetime
    examples: int
    shards: int
    audio_bytes: int
    export_receipt_sha256: str


def safe_audio_path(directory: Path, relative_path: Path) -> Path:
    path = (directory / relative_path).resolve(strict=True)
    if relative_path.is_absolute() or not path.is_relative_to(directory.resolve()):
        raise ValueError("Audio receipt must name a file inside its source directory")
    return path


def qwen_examples(source: QwenCorpusSource) -> tuple[ExportAudio, ...]:
    text_directory = source.directory / "text_generation_4b"
    audio_directory = source.directory / "audio_omni"
    utterances = read_release_records(text_directory / "utterances.jsonl", EmotionalUtterance)
    targets = read_release_records(text_directory / "teacher_targets.jsonl", EmotionalTeacherTarget)
    canonical = {utterance.base_id: utterance for utterance in utterances}
    if len(canonical) != len(utterances):
        raise ValueError("Duplicate canonical utterance")
    expected = {
        (utterance.base_id, annotation.delivery)
        for utterance in utterances
        for annotation in utterance.deliveries
    }
    if {
        (target.request.utterance.base_id, target.request.annotation.delivery) for target in targets
    } != expected:
        raise ValueError("Teacher targets do not cover canonical utterances")
    rows: list[ExportAudio] = []
    for target in targets:
        utterance = target.request.utterance
        annotation = target.request.annotation
        if canonical.get(utterance.base_id) != utterance or annotation not in utterance.deliveries:
            raise ValueError("Teacher request differs from the canonical utterance/delivery")
        case_id = f"{utterance.base_id}_{annotation.delivery.value}"
        clip = PreviewClip.model_validate_json(
            (audio_directory / "clips" / f"{case_id}.json").read_bytes()
        )
        termination = CodecTermination.model_validate_json(
            (audio_directory / "terminations" / f"{case_id}.json").read_bytes()
        )
        if (
            clip.case.case_id != case_id
            or clip.case.text != utterance.text
            or clip.case.delivery != annotation.delivery
            or clip.case.instruct != annotation.instruct
            or termination.case_id != case_id
            or termination.codec_tokens != clip.codec_tokens
            or not termination.accepted
        ):
            raise ValueError(f"Audio identity/completion differs from teacher: {case_id}")
        rows.append(
            ExportAudio(
                ReleasedExample(
                    case_id=case_id,
                    utterance_id=utterance.base_id,
                    family_id=utterance.family_id,
                    split=utterance.split,
                    domain=utterance.domain,
                    intent=utterance.intent,
                    text=utterance.text,
                    delivery=annotation.delivery,
                    teacher_response=target.response.text,
                    audio_sha256=clip.audio.sha256,
                    sample_rate=clip.sample_rate,
                    duration_seconds=clip.duration_seconds,
                ),
                safe_audio_path(audio_directory, clip.audio.path),
            )
        )
    return tuple(rows)


def neu_examples(source: NeuCorpusSource) -> tuple[ExportAudio, ...]:
    audio_directory = source.directory / "audio"
    result = NeuCorpusResult.model_validate_json((audio_directory / "result.json").read_bytes())
    if result.missing_cases:
        raise ValueError("Neu corpus is incomplete")
    utterances = read_release_records(source.directory / "utterances.jsonl", NeuUtterance)
    targets = read_release_records(source.directory / "teacher_targets.jsonl", NeuTeacherTarget)
    canonical = {utterance.assignment.base_id: utterance for utterance in utterances}
    if len(canonical) != len(utterances):
        raise ValueError("Duplicate canonical utterance")
    if {target.case.case_id for target in targets} != {
        f"{utterance.assignment.base_id}_{emotion.value}"
        for utterance in utterances
        for emotion in utterance.assignment.emotions
    }:
        raise ValueError("Teacher targets do not cover canonical utterances")
    rows: list[ExportAudio] = []
    for target in targets:
        case = target.case
        utterance = canonical[case.utterance_id]
        if (
            case.text != utterance.text
            or case.emotion not in utterance.assignment.emotions
            or case.case_id != f"{case.utterance_id}_{case.emotion.value}"
        ):
            raise ValueError(
                f"Teacher request differs from canonical Neu utterance: {case.case_id}"
            )
        evidence = BatchClipEvidence.model_validate_json(
            (audio_directory / "clips" / f"{case.case_id}.json").read_bytes()
        )
        clip = evidence.clip
        if (
            clip.case.model_copy(update={"seed": case.seed}) != case
            or clip.termination != PilotTermination.STOP
            or not evidence.generated_token_ids
            or evidence.generated_token_ids[-1] != result.speech_end_token_id
        ):
            raise ValueError(f"Neu clip lacks matching identity or true EOS: {case.case_id}")
        rows.append(
            ExportAudio(
                ReleasedExample(
                    case_id=case.case_id,
                    utterance_id=case.utterance_id,
                    family_id=utterance.assignment.family_id,
                    split=utterance.assignment.split,
                    domain=utterance.assignment.domain,
                    intent=utterance.assignment.intent,
                    text=utterance.text,
                    delivery=case.emotion,
                    teacher_response=target.response.text,
                    audio_sha256=clip.sha256,
                    sample_rate=clip.sample_rate,
                    duration_seconds=clip.audio_seconds,
                ),
                safe_audio_path(audio_directory, Path(clip.audio_path)),
            )
        )
    return tuple(rows)


def validate_pairs(rows: Sequence[ExportAudio], expected_utterances: int) -> int:
    if len(rows) != 2 * expected_utterances:
        raise ValueError("Release must contain every configured two-delivery pair")
    cases: set[str] = set()
    families: dict[str, Split] = {}
    pairs: dict[str, list[ReleasedExample]] = {}
    for row in rows:
        example = row.example
        if example.case_id in cases or not example.teacher_response.strip():
            raise ValueError("Duplicate case ID or empty teacher response")
        cases.add(example.case_id)
        if example.family_id in families and families[example.family_id] != example.split:
            raise ValueError("Design family crosses dataset splits")
        families[example.family_id] = example.split
        pairs.setdefault(example.utterance_id, []).append(example)
    if len(pairs) != expected_utterances:
        raise ValueError("Unexpected number of unique utterance IDs")
    identical = 0
    for pair in pairs.values():
        if len(pair) != 2:
            raise ValueError("Every utterance must have exactly two deliveries")
        first, second = pair
        if (
            first.delivery == second.delivery
            or first.text != second.text
            or first.family_id != second.family_id
            or first.split != second.split
            or first.domain != second.domain
            or first.intent != second.intent
        ):
            raise ValueError("Paired delivery content or metadata differs")
        identical += first.teacher_response.strip() == second.teacher_response.strip()
    return identical


def waveform_bytes(row: ExportAudio) -> bytes:
    content = row.waveform_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != row.example.audio_sha256:
        raise ValueError(f"WAV hash changed: {row.example.case_id}")
    waveform, sample_rate = soundfile.read(io.BytesIO(content), dtype="float32")
    if (
        sample_rate != row.example.sample_rate
        or sample_rate != 24000
        or waveform.ndim != 1
        or not np.isfinite(waveform).all()
        or abs(len(waveform) / sample_rate - row.example.duration_seconds) > 1 / sample_rate
    ):
        raise ValueError(f"Invalid WAV format or duration: {row.example.case_id}")
    return content


def export_batches(rows: Sequence[ExportAudio], size: int) -> Iterator[Sequence[ExportAudio]]:
    for offset in range(0, len(rows), size):
        yield rows[offset : offset + size]


def copy_generation_evidence(source: CorpusSource, output: Path) -> None:
    match source:
        case QwenCorpusSource():
            files = (
                Path("generation_config.json"),
                Path("draft_4b_config.json"),
                Path("omni_audio_config.json"),
                Path("text_generation_4b/utterances.jsonl"),
                Path("text_generation_4b/teacher_targets.jsonl"),
                Path("audio_omni/summary.json"),
                Path("audio_omni/provenance.json"),
                Path("supporting_evidence/vllm_requirements.txt"),
                Path("supporting_evidence/omni_requirements.txt"),
            )
        case NeuCorpusSource():
            files = (
                Path("neu_drafts_config.json"),
                Path("neu_targets_config.json"),
                Path("utterances.jsonl"),
                Path("teacher_targets.jsonl"),
                Path("cases.json"),
                Path("dataset_provenance.json"),
                Path("teacher_source.json"),
                Path("audio/config.json"),
                Path("audio/result.json"),
                Path("analysis/source_handoff_57f3d8a.json"),
                Path("analysis/source_handoff_verification_57f3d8a.json"),
            )
    for relative in files:
        destination = output / "generation" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.directory / relative, destination)


def write_shard(path: Path, rows: Sequence[ExportAudio], split: Split) -> ExportShard:
    examples = tuple(row.example for row in rows)
    table = arrow.Table.from_pylist([example.model_dump(mode="json") for example in examples])
    audio = arrow.StructArray.from_arrays(
        (
            arrow.array([waveform_bytes(row) for row in rows], type=arrow.binary()),
            arrow.array([f"{example.case_id}.wav" for example in examples]),
        ),
        names=("bytes", "path"),
    )
    table = table.append_column("audio", audio)
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(table, path, compression="zstd", row_group_size=100)
    size, digest = stable_digest(path)
    return ExportShard(
        path=Path("data") / path.name,
        split=split,
        examples=len(rows),
        bytes=size,
        sha256=digest,
    )


def export_dataset(config: DatasetExportConfig) -> DatasetExportReceipt:
    source = config.source.directory.resolve(strict=True)
    output = config.output_directory.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Release output must be separate from immutable source data")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Release output must be empty; preserve previous export evidence")
    match config.source:
        case QwenCorpusSource():
            rows = qwen_examples(config.source)
        case NeuCorpusSource():
            rows = neu_examples(config.source)
    identical = validate_pairs(rows, config.expected_utterances)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "examples.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(row.example.model_dump_json() + "\n")
    shards: list[ExportShard] = []
    for split in Split:
        selected = sorted(
            (row for row in rows if row.example.split == split),
            key=lambda row: row.example.case_id,
        )
        for index, batch in enumerate(export_batches(selected, config.rows_per_shard)):
            path = output / "data" / f"{split.value}-{index:05d}.parquet"
            shards.append(write_shard(path, batch, split))
    copy_generation_evidence(config.source, output)
    documentation = Path(__file__).resolve().parents[1] / "docs"
    match config.source:
        case QwenCorpusSource():
            card = documentation / "dataset_cards" / "qwen.md"
        case NeuCorpusSource():
            card = documentation / "dataset_cards" / "neu.md"
            shutil.copyfile(
                documentation / "licenses" / "NeuTTS-Open-License-1.0.txt",
                output / "NeuTTS-Open-License-1.0.txt",
            )
    shutil.copyfile(card, output / "README.md")
    shutil.copyfile(documentation / "dataset_cards" / "LICENSE.md", output / "LICENSE.md")
    counts = Counter(row.example.split for row in rows)
    receipt = DatasetExportReceipt(
        configuration=config,
        utterances=config.expected_utterances,
        examples=len(rows),
        identical_teacher_pairs=identical,
        train_examples=counts[Split.TRAIN],
        validation_examples=counts[Split.VALIDATION],
        test_examples=counts[Split.TEST],
        audio_seconds=sum(row.example.duration_seconds for row in rows),
        source_audio_bytes=sum(row.waveform_path.stat().st_size for row in rows),
        shards=tuple(shards),
    )
    write_record(output / "export_receipt.json", receipt)
    schema = ReleasedExample.model_json_schema()
    (output / "example_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return receipt


def verify_dataset_export(config: DatasetExportConfig) -> DatasetExportVerification:
    output = config.output_directory
    serialized = (output / "export_receipt.json").read_bytes()
    receipt = DatasetExportReceipt.model_validate_json(serialized)
    if receipt.configuration != config:
        raise ValueError("Verification configuration differs from export receipt")
    rows = read_release_records(output / "examples.jsonl", ReleasedExample)
    expected = {row.case_id: row for row in rows}
    if len(expected) != len(rows) or len(rows) != receipt.examples:
        raise ValueError("Duplicate or missing examples in released metadata")
    seen: set[str] = set()
    total_bytes = 0
    for shard in receipt.shards:
        path = safe_audio_path(output, shard.path)
        if stable_digest(path) != (shard.bytes, shard.sha256):
            raise ValueError(f"Export shard hash changed: {shard.path}")
        table = parquet.read_table(path)
        if table.num_rows != shard.examples:
            raise ValueError("Shard example count differs from receipt")
        examples = tuple(
            ReleasedExample.model_validate(row) for row in table.drop_columns(["audio"]).to_pylist()
        )
        audio = cast(arrow.StructArray, table.column("audio").combine_chunks())
        for index, example in enumerate(examples):
            if (
                example.case_id in seen
                or expected.get(example.case_id) != example
                or example.split != shard.split
            ):
                raise ValueError("Shard identity, split or metadata differs from release index")
            content: bytes = audio.field("bytes")[index].as_py()
            name: str = audio.field("path")[index].as_py()
            if (
                hashlib.sha256(content).hexdigest() != example.audio_sha256
                or name != f"{example.case_id}.wav"
            ):
                raise ValueError(f"Embedded original WAV bytes differ: {example.case_id}")
            total_bytes += len(content)
            seen.add(example.case_id)
    if seen != set(expected) or total_bytes != receipt.source_audio_bytes:
        raise ValueError("Released shards do not cover all source audio bytes and examples")
    verification = DatasetExportVerification(
        verified_at=datetime.now(timezone.utc),
        examples=len(seen),
        shards=len(receipt.shards),
        audio_bytes=total_bytes,
        export_receipt_sha256=hashlib.sha256(serialized).hexdigest(),
    )
    write_record(output / "export_verification.json", verification)
    return verification
