"""Read-only snapshots and paired-utterance reporting during or after generation."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TypeVar

from pydantic import Field

from scripts.inventory_results import stable_digest, write_record
from scripts.package_results import FileArtifact
from speech_projector.data import Distribution, distribution
from speech_projector.emotion_preview import Delivery, PreviewClip
from speech_projector.emotional_dataset import (
    Domain,
    EmotionalDatasetConfig,
    EmotionalUtterance,
    assemble_utterance,
    build_draft_requests,
    normalized_utterance,
)
from speech_projector.emotional_generation import EmotionalTeacherTarget
from speech_projector.models import Record, Split

RecordType = TypeVar("RecordType", bound=Record)


class EmotionalReportConfig(Record):
    dataset_directory: Path
    audio_directory: Path
    output_directory: Path
    sample_count: int = Field(default=20, gt=0)


class JournalSnapshot(Record):
    source_path: Path
    source_present: bool
    snapshot: FileArtifact
    bytes_read: int
    incomplete_tail_bytes: int
    records: int


class MissingStage(str, Enum):
    TEACHER_TARGET = "teacher_target"
    AUDIO = "audio"


class MissingDelivery(Record):
    base_id: str
    delivery: Delivery
    stages: tuple[MissingStage, ...]


class EmotionalPairRecord(Record):
    utterance: EmotionalUtterance
    teacher_targets: tuple[EmotionalTeacherTarget, ...]
    audio: tuple[PreviewClip, ...]


class EmotionalDatasetReport(Record):
    captured_at: datetime
    configuration: EmotionalReportConfig
    journal_snapshots: tuple[JournalSnapshot, JournalSnapshot]
    planned_utterances: int
    available_utterances: int
    planned_delivery_examples: int
    available_teacher_targets: int
    available_audio_clips: int
    complete_teacher_pairs: int
    complete_audio_pairs: int
    complete_teacher_and_audio_pairs: int
    identical_teacher_response_pairs: int
    missing_utterance_ids: tuple[str, ...]
    missing_deliveries: tuple[MissingDelivery, ...]
    domain_counts: tuple[tuple[Domain, int], ...]
    split_counts: tuple[tuple[Split, int], ...]
    delivery_counts: tuple[tuple[Delivery, int], ...]
    utterance_words: Distribution | None
    teacher_response_words: Distribution | None
    audio_duration_seconds: Distribution | None
    all_pairs_path: Path


def snapshot_journal(
    source: Path,
    record_type: type[RecordType],
    destination: Path,
    output_root: Path,
) -> tuple[tuple[RecordType, ...], JournalSnapshot]:
    present = source.exists()
    content = source.read_bytes() if present else b""
    complete = content.rfind(b"\n") + 1
    prefix = content[:complete]
    records = tuple(record_type.model_validate_json(line) for line in prefix.splitlines())
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    partial.write_bytes(prefix)
    partial.replace(destination)
    snapshot = JournalSnapshot(
        source_path=source.resolve(),
        source_present=present,
        snapshot=FileArtifact(
            path=destination.relative_to(output_root),
            source_path=destination.resolve(),
            bytes=len(prefix),
            sha256=hashlib.sha256(prefix).hexdigest(),
        ),
        bytes_read=len(content),
        incomplete_tail_bytes=len(content) - complete,
        records=len(records),
    )
    return records, snapshot


def load_clips(directory: Path) -> tuple[PreviewClip, ...]:
    clips: list[PreviewClip] = []
    for path in sorted((directory / "clips").glob("*.json")):
        clip = PreviewClip.model_validate_json(path.read_bytes())
        audio = (directory / clip.audio.path).resolve()
        if not audio.is_relative_to(directory.resolve()):
            raise ValueError("Audio artifact path leaves the selected audio directory")
        if stable_digest(audio) != (clip.audio.bytes, clip.audio.sha256):
            raise ValueError(f"Completed audio hash or size differs: {clip.case.case_id}")
        clips.append(clip)
    return tuple(clips)


def pair_records(
    utterances: tuple[EmotionalUtterance, ...],
    targets: tuple[EmotionalTeacherTarget, ...],
    clips: tuple[PreviewClip, ...],
) -> tuple[EmotionalPairRecord, ...]:
    if len({item.base_id for item in utterances}) != len(utterances):
        raise ValueError("Utterance IDs are duplicated")
    if len({normalized_utterance(item.text) for item in utterances}) != len(utterances):
        raise ValueError("Utterance literal texts are exact normalized duplicates")
    by_id = {item.base_id: item for item in utterances}
    by_text = {item.text: item for item in utterances}
    target_index: dict[tuple[str, Delivery], EmotionalTeacherTarget] = {}
    for target in targets:
        utterance = target.request.utterance
        annotation = target.request.annotation
        key = (utterance.base_id, annotation.delivery)
        if by_id.get(utterance.base_id) != utterance or annotation not in utterance.deliveries:
            raise ValueError(
                "Teacher target differs from an available canonical utterance/delivery"
            )
        if key in target_index:
            raise ValueError("Teacher target ID/delivery pair is duplicated")
        target_index[key] = target
    audio_index: dict[tuple[str, Delivery], PreviewClip] = {}
    for clip in clips:
        utterance = by_text.get(clip.case.text)
        if utterance is None:
            raise ValueError("Audio clip literal text is not in the available utterance snapshot")
        annotation = next(
            (item for item in utterance.deliveries if item.delivery == clip.case.delivery), None
        )
        if annotation is None or annotation.instruct != clip.case.instruct:
            raise ValueError("Audio clip differs from the canonical delivery instruction")
        key = (utterance.base_id, annotation.delivery)
        if key in audio_index:
            raise ValueError("Audio ID/delivery pair is duplicated")
        audio_index[key] = clip
    return tuple(
        EmotionalPairRecord(
            utterance=utterance,
            teacher_targets=tuple(
                target_index[(utterance.base_id, annotation.delivery)]
                for annotation in utterance.deliveries
                if (utterance.base_id, annotation.delivery) in target_index
            ),
            audio=tuple(
                audio_index[(utterance.base_id, annotation.delivery)]
                for annotation in utterance.deliveries
                if (utterance.base_id, annotation.delivery) in audio_index
            ),
        )
        for utterance in utterances
    )


def measured_distribution(values: list[float]) -> Distribution | None:
    return distribution(values) if values else None


def render_report(report: EmotionalDatasetReport, pairs: tuple[EmotionalPairRecord, ...]) -> str:
    lines = [
        "# Paired emotional dataset snapshot",
        "",
        f"Captured: {report.captured_at.isoformat()}. "
        "Journals were independently read once; this is a partial snapshot while writers run, "
        "not a claim that the whole program completed.",
        "",
        f"Utterances: {report.available_utterances}/{report.planned_utterances}; "
        f"teacher targets: {report.available_teacher_targets}/{report.planned_delivery_examples}; "
        f"audio clips: {report.available_audio_clips}/{report.planned_delivery_examples}.",
        "",
        f"Complete teacher pairs: {report.complete_teacher_pairs}; complete audio pairs: "
        f"{report.complete_audio_pairs}; both: {report.complete_teacher_and_audio_pairs}.",
        "",
        f"Exactly identical teacher replies: {report.identical_teacher_response_pairs}/"
        f"{report.complete_teacher_pairs} complete teacher pairs "
        "(case-sensitive saved text equality; "
        "diagnostic only, no rejection).",
        "",
        "Intended delivery labels come from synthesis instructions, not verified listening. "
        "Teacher replies are model-generated targets, not independently established gold. "
        "Word statistics use Unicode lexical tokens, keeping contractions as one word. "
        "Audio durations cover only completed, hash-verified clips.",
        "",
        "## Available sample distributions",
        "",
        "| Measure | Minimum | Median | p90 | p99 | Maximum | Mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, measured in (
        ("Utterance words", report.utterance_words),
        ("Teacher response words", report.teacher_response_words),
        ("Audio seconds", report.audio_duration_seconds),
    ):
        if measured is None:
            lines.append(f"| {label} | unavailable | | | | | |")
        else:
            lines.append(
                f"| {label} | {measured.minimum:.3f} | {measured.median:.3f} | "
                f"{measured.p90:.3f} | {measured.p99:.3f} | {measured.maximum:.3f} | "
                f"{measured.mean:.3f} |"
            )
    lines.extend(("", "## Available domain / split / delivery counts", ""))
    for label, counts in (
        ("Domains", report.domain_counts),
        ("Splits", report.split_counts),
        ("Planned deliveries for available texts", report.delivery_counts),
    ):
        lines.extend(
            (label + ": " + ", ".join(f"{name.value}={count}" for name, count in counts), "")
        )
    lines.extend(("## First saved utterances in canonical order", ""))
    for pair in pairs[: report.configuration.sample_count]:
        utterance = pair.utterance
        lines.extend(
            (
                f"### {utterance.base_id}",
                "",
                f"{utterance.domain.value}; {utterance.intent.value}; {utterance.split.value}; "
                f"family {utterance.family_id}.",
                "",
                utterance.text,
                "",
            )
        )
        for annotation in utterance.deliveries:
            target = next(
                (item for item in pair.teacher_targets if item.request.annotation == annotation),
                None,
            )
            clip = next(
                (item for item in pair.audio if item.case.delivery == annotation.delivery), None
            )
            lines.extend(
                (
                    f"**{annotation.delivery.value}**",
                    "",
                    f"Unspoken TTS instruction: {annotation.instruct}",
                    "",
                    "Teacher target: " + (target.response.text if target else "pending"),
                    "",
                )
            )
            if clip is None:
                lines.extend(("Audio: pending.", ""))
            else:
                audio = (
                    (report.configuration.audio_directory / clip.audio.path).resolve().as_posix()
                )
                lines.extend(
                    (
                        f"[Audio](<{audio}>) — {clip.duration_seconds:.3f}s; "
                        f"case {clip.case.case_id}; SHA256 `{clip.audio.sha256}`.",
                        "",
                    )
                )
    lines.extend(
        (
            "Complete pair records and explicit pending stage IDs are saved in the adjacent "
            "JSON/JSONL files. Missing delivery stages list only available utterances; "
            "the report separately lists all still-undrafted base IDs.",
            "",
        )
    )
    return "\n".join(lines)


def write_emotional_report(configuration: EmotionalReportConfig) -> EmotionalDatasetReport:
    output = configuration.output_directory
    if output.resolve() in (
        configuration.dataset_directory.resolve(),
        configuration.audio_directory.resolve(),
    ):
        raise ValueError("Report output must be separate from source dataset/audio directories")
    output.mkdir(parents=True, exist_ok=True)
    dataset_config = EmotionalDatasetConfig.model_validate_json(
        (configuration.dataset_directory / "configuration.json").read_bytes()
    )
    utterances, utterance_snapshot = snapshot_journal(
        configuration.dataset_directory / "utterances.jsonl",
        EmotionalUtterance,
        output / "snapshots/utterances.jsonl",
        output,
    )
    targets, target_snapshot = snapshot_journal(
        configuration.dataset_directory / "teacher_targets.jsonl",
        EmotionalTeacherTarget,
        output / "snapshots/teacher_targets.jsonl",
        output,
    )
    clips = load_clips(configuration.audio_directory)
    pairs = pair_records(utterances, targets, clips)
    planned = tuple(
        assignment
        for request in build_draft_requests(dataset_config)
        for assignment in request.assignments
    )
    if tuple(item.base_id for item in utterances) != tuple(
        assignment.base_id for assignment in planned[: len(utterances)]
    ):
        raise ValueError("Available utterances do not follow the exact planned ID prefix")
    if any(
        assemble_utterance(assignment, row.text) != row
        for assignment, row in zip(planned[: len(utterances)], utterances, strict=True)
    ):
        raise ValueError("Available utterance metadata differs from its quota assignment")
    missing: list[MissingDelivery] = []
    for pair in pairs:
        for annotation in pair.utterance.deliveries:
            stages = tuple(
                stage
                for stage, available in (
                    (
                        MissingStage.TEACHER_TARGET,
                        any(
                            target.request.annotation == annotation
                            for target in pair.teacher_targets
                        ),
                    ),
                    (
                        MissingStage.AUDIO,
                        any(clip.case.delivery == annotation.delivery for clip in pair.audio),
                    ),
                )
                if not available
            )
            if stages:
                missing.append(
                    MissingDelivery(
                        base_id=pair.utterance.base_id, delivery=annotation.delivery, stages=stages
                    )
                )
    pair_path = output / "pairs.jsonl"
    partial = pair_path.with_suffix(".part")
    partial.write_text("".join(pair.model_dump_json() + "\n" for pair in pairs), encoding="utf-8")
    partial.replace(pair_path)
    report = EmotionalDatasetReport(
        captured_at=datetime.now(timezone.utc),
        configuration=configuration,
        journal_snapshots=(utterance_snapshot, target_snapshot),
        planned_utterances=dataset_config.utterance_count,
        available_utterances=len(utterances),
        planned_delivery_examples=2 * dataset_config.utterance_count,
        available_teacher_targets=len(targets),
        available_audio_clips=len(clips),
        complete_teacher_pairs=sum(len(pair.teacher_targets) == 2 for pair in pairs),
        complete_audio_pairs=sum(len(pair.audio) == 2 for pair in pairs),
        complete_teacher_and_audio_pairs=sum(
            len(pair.teacher_targets) == len(pair.audio) == 2 for pair in pairs
        ),
        identical_teacher_response_pairs=sum(
            len(pair.teacher_targets) == 2
            and pair.teacher_targets[0].response.text == pair.teacher_targets[1].response.text
            for pair in pairs
        ),
        missing_utterance_ids=tuple(
            assignment.base_id for assignment in planned[len(utterances) :]
        ),
        missing_deliveries=tuple(missing),
        domain_counts=tuple(
            (domain, sum(row.domain == domain for row in utterances)) for domain in Domain
        ),
        split_counts=tuple(
            (split, sum(row.split == split for row in utterances)) for split in Split
        ),
        delivery_counts=tuple(
            (
                delivery,
                sum(
                    annotation.delivery == delivery
                    for row in utterances
                    for annotation in row.deliveries
                ),
            )
            for delivery in Delivery
        ),
        utterance_words=measured_distribution(
            [float(len(normalized_utterance(row.text).split())) for row in utterances]
        ),
        teacher_response_words=measured_distribution(
            [float(len(normalized_utterance(target.response.text).split())) for target in targets]
        ),
        audio_duration_seconds=measured_distribution([clip.duration_seconds for clip in clips]),
        all_pairs_path=pair_path.resolve(),
    )
    write_record(output / "dataset_report.json", report)
    (output / "dataset_report.md").write_text(render_report(report, pairs), encoding="utf-8")
    return report
