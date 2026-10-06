"""Partial-snapshot, paired-target and immutable-audio reporting checks."""

from pathlib import Path

import numpy as np
import pytest

from scripts.inventory_results import stable_digest, write_record
from speech_projector.emotion_preview import (
    PreviewCase,
    PreviewPlan,
    SynthesizedAudio,
    persist_clip,
)
from speech_projector.emotional_dataset import (
    EmotionalDatasetConfig,
    EmotionalUtterance,
    GeneratedDraftBatch,
    GeneratedDraftText,
    accept_draft_batch,
    build_draft_requests,
)
from speech_projector.emotional_generation import EmotionalTeacherRequest, EmotionalTeacherTarget
from speech_projector.emotional_report import (
    EmotionalReportConfig,
    MissingStage,
    pair_records,
    snapshot_journal,
    write_emotional_report,
)
from speech_projector.generation import CompletedGeneration
from speech_projector.journal import append_record


def utterances(configuration: EmotionalDatasetConfig) -> tuple[EmotionalUtterance, ...]:
    request = build_draft_requests(configuration)[0]
    return accept_draft_batch(
        request,
        GeneratedDraftBatch(
            utterances=tuple(
                GeneratedDraftText(
                    base_id=assignment.base_id,
                    text=f"Please help me find the parcel labelled {assignment.base_id} today.",
                )
                for assignment in request.assignments
            )
        ),
        (),
    )


def teacher_target(utterance: EmotionalUtterance, index: int) -> EmotionalTeacherTarget:
    return EmotionalTeacherTarget(
        request=EmotionalTeacherRequest(
            utterance=utterance, annotation=utterance.deliveries[index]
        ),
        response=CompletedGeneration(
            text="I can help you check the package now.", token_ids=(1, 2)
        ),
        capped_attempts=(),
    )


def test_snapshot_ignores_incomplete_tail_without_mutating_source(tmp_path: Path) -> None:
    row = utterances(EmotionalDatasetConfig(utterance_count=1))[0]
    source = tmp_path / "live.jsonl"
    content = row.model_dump_json().encode() + b'\n{"unfinished":'
    source.write_bytes(content)
    records, snapshot = snapshot_journal(
        source, EmotionalUtterance, tmp_path / "snapshots/copy.jsonl", tmp_path
    )
    assert records == (row,) and source.read_bytes() == content
    assert snapshot.incomplete_tail_bytes == len(b'{"unfinished":')
    assert snapshot.records == 1
    assert stable_digest(tmp_path / snapshot.snapshot.path) == (
        snapshot.snapshot.bytes,
        snapshot.snapshot.sha256,
    )
    assert not (tmp_path / "journal_recovery.jsonl").exists()


def test_partial_pair_report_preserves_missing_ids_and_exact_agreement(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    audio = tmp_path / "audio"
    output = tmp_path / "report"
    configuration = EmotionalDatasetConfig(utterance_count=3)
    rows = utterances(configuration)
    write_record(dataset / "configuration.json", configuration)
    for row in rows[:2]:
        append_record(dataset / "utterances.jsonl", row)
    for target in (
        teacher_target(rows[0], 0),
        teacher_target(rows[0], 1),
        teacher_target(rows[1], 0),
    ):
        append_record(dataset / "teacher_targets.jsonl", target)
    cases = tuple(
        PreviewCase(
            case_id=f"{rows[0].base_id}_{annotation.delivery.value}",
            text=rows[0].text,
            delivery=annotation.delivery,
            instruct=annotation.instruct,
            seed=42 + index,
        )
        for index, annotation in enumerate(rows[0].deliveries)
    )
    plan = PreviewPlan(cases=cases)
    for case in cases:
        persist_clip(
            audio,
            plan,
            case,
            SynthesizedAudio(np.linspace(-0.2, 0.2, 2400, dtype=np.float32), 24000, 8, 0.1),
        )
    report = write_emotional_report(
        EmotionalReportConfig(
            dataset_directory=dataset,
            audio_directory=audio,
            output_directory=output,
        )
    )
    assert report.planned_utterances == 3 and report.available_utterances == 2
    assert report.available_teacher_targets == 3 and report.available_audio_clips == 2
    assert report.complete_teacher_pairs == report.complete_audio_pairs == 1
    assert report.complete_teacher_and_audio_pairs == report.identical_teacher_response_pairs == 1
    assert report.missing_utterance_ids == (rows[2].base_id,)
    assert report.audio_duration_seconds is not None
    assert report.audio_duration_seconds.mean == pytest.approx(0.1)
    assert any(
        item.base_id == rows[1].base_id and MissingStage.TEACHER_TARGET in item.stages
        for item in report.missing_deliveries
    )
    readable = (output / "dataset_report.md").read_text(encoding="utf-8")
    assert "diagnostic only, no rejection" in readable and "pending" in readable
    assert "not independently established gold" in readable
    assert report.all_pairs_path.exists()


def test_empty_live_snapshot_has_explicit_unavailable_distributions(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    write_record(dataset / "configuration.json", EmotionalDatasetConfig(utterance_count=2))
    report = write_emotional_report(
        EmotionalReportConfig(
            dataset_directory=dataset,
            audio_directory=tmp_path / "audio",
            output_directory=tmp_path / "report",
        )
    )
    assert report.available_utterances == report.available_teacher_targets == 0
    assert (
        report.utterance_words
        is report.teacher_response_words
        is report.audio_duration_seconds
        is None
    )
    assert all(not item.source_present for item in report.journal_snapshots)
    assert len(report.missing_utterance_ids) == 2


def test_duplicate_or_mismatched_teacher_target_is_not_silently_counted() -> None:
    rows = utterances(EmotionalDatasetConfig(utterance_count=2))
    target = teacher_target(rows[0], 0)
    with pytest.raises(ValueError, match="duplicated"):
        pair_records(rows, (target, target), ())
    with pytest.raises(ValueError, match="differs"):
        pair_records(rows[1:], (target,), ())


def test_audio_byte_mismatch_blocks_completed_clip_claim(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    audio = tmp_path / "audio"
    configuration = EmotionalDatasetConfig(utterance_count=1)
    row = utterances(configuration)[0]
    write_record(dataset / "configuration.json", configuration)
    append_record(dataset / "utterances.jsonl", row)
    annotation = row.deliveries[0]
    case = PreviewCase(
        case_id="clip",
        text=row.text,
        delivery=annotation.delivery,
        instruct=annotation.instruct,
        seed=42,
    )
    clip = persist_clip(
        audio,
        PreviewPlan(cases=(case,)),
        case,
        SynthesizedAudio(np.ones(100, dtype=np.float32), 24000, 8, 0.1),
    )
    (audio / clip.audio.path).write_bytes(b"altered")
    with pytest.raises(ValueError, match="hash or size"):
        write_emotional_report(
            EmotionalReportConfig(
                dataset_directory=dataset,
                audio_directory=audio,
                output_directory=tmp_path / "report",
            )
        )
