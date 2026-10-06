"""Quota, structural repairs, pinned older data and crash recovery for fresh Neu pairs."""

from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.inventory_results import write_record
from scripts.neutts_pilot_state import digest
from speech_projector.emotional_dataset import (
    EmotionalDatasetConfig,
    GeneratedDraftBatch,
    GeneratedDraftText,
    accept_draft_batch,
    build_draft_requests,
    normalized_utterance,
)
from speech_projector.journal import append_record, read_journal
from speech_projector.models import Split
from speech_projector.neu_dataset import (
    EMOTION_PAIRS,
    NeuAcceptedBatch,
    NeuDatasetProvenance,
    NeuDraftAssignment,
    NeuDraftFailure,
    NeuDraftRequest,
    NeuUtterance,
    accept_neu_batch,
    build_neu_requests,
    neu_cases,
    neu_draft_prompt,
    run_neu_construction,
    write_neu_cases,
)
from speech_projector.tts_pilot import PilotEmotion


def generated(request: NeuDraftRequest) -> GeneratedDraftBatch:
    return GeneratedDraftBatch(
        utterances=tuple(
            GeneratedDraftText(
                base_id=assignment.base_id,
                text=(
                    f"Please help me collect the package numbered {assignment.base_id} "
                    "before tomorrow."
                ),
            )
            for assignment in request.assignments
        )
    )


def provenance(directory: Path, configuration: EmotionalDatasetConfig) -> NeuDatasetProvenance:
    old_path = directory / "old_utterances.jsonl"
    old_request = build_draft_requests(EmotionalDatasetConfig(utterance_count=1))[0]
    old = accept_draft_batch(
        old_request,
        GeneratedDraftBatch(
            utterances=(
                GeneratedDraftText(
                    base_id=old_request.assignments[0].base_id,
                    text="Please check the delivery receipt before our appointment tomorrow.",
                ),
            )
        ),
        (),
    )
    append_record(old_path, old[0])
    return NeuDatasetProvenance(
        configuration=configuration, previous_utterances=old_path, previous_sha256=digest(old_path)
    )


def test_fresh_balanced_quota_preserves_whole_scenario_family_splits() -> None:
    config = EmotionalDatasetConfig()
    requests = build_neu_requests(config)
    assignments = tuple(assignment for request in requests for assignment in request.assignments)
    assert len(assignments) == len({assignment.base_id for assignment in assignments}) == 5000
    assert Counter(assignment.emotions for assignment in assignments) == Counter(
        {pair: 1250 for pair in EMOTION_PAIRS}
    )
    assert Counter(assignment.split for assignment in assignments) == Counter(
        {Split.TRAIN: 4550, Split.VALIDATION: 230, Split.TEST: 220}
    )
    original = tuple(
        assignment for request in build_draft_requests(config) for assignment in request.assignments
    )
    assert tuple((row.family_id, row.split) for row in assignments) == tuple(
        (row.family_id, row.split) for row in original
    )
    assert all(row.base_id != old.base_id for row, old in zip(assignments, original, strict=True))


@pytest.mark.parametrize(
    "emotions",
    [
        (PilotEmotion.NEUTRAL, PilotEmotion.HAPPY),
        (PilotEmotion.ANGRY, PilotEmotion.ANGRY),
        (PilotEmotion.HAPPY, PilotEmotion.AFRAID),
    ],
)
def test_unsupported_and_alias_deliveries_are_rejected(
    emotions: tuple[PilotEmotion, PilotEmotion],
) -> None:
    first = build_neu_requests(EmotionalDatasetConfig(utterance_count=1))[0].assignments[0]
    with pytest.raises(ValidationError):
        NeuDraftAssignment(
            base_id=first.base_id,
            family_id=first.family_id,
            split=first.split,
            domain=first.domain,
            intent=first.intent,
            situation=first.situation,
            fact_target=first.fact_target,
            emotions=emotions,
        )


def test_exact_old_text_duplicate_and_short_rows_have_specific_feedback() -> None:
    request = build_neu_requests(EmotionalDatasetConfig(utterance_count=1))[0]
    batch = generated(request)
    with pytest.raises(ValueError, match="old/new"):
        accept_neu_batch(request, batch, {normalized_utterance(batch.utterances[0].text)})
    bad = GeneratedDraftBatch(
        utterances=(
            GeneratedDraftText(
                base_id=request.assignments[0].base_id, text="Where is that parcel?"
            ),
        )
    )
    with pytest.raises(ValueError, match="neu_base_00000: word_count=4"):
        accept_neu_batch(request, bad, set())


def test_required_repair_is_saved_and_successful_resume_never_regenerates(tmp_path: Path) -> None:
    source = provenance(
        tmp_path, EmotionalDatasetConfig(utterance_count=3, max_acceptance_attempts=2)
    )
    calls: list[str] = []

    def generate(request: NeuDraftRequest) -> GeneratedDraftBatch:
        calls.append(request.batch_id)
        if not request.feedback:
            return GeneratedDraftBatch(
                utterances=tuple(
                    GeneratedDraftText(base_id=row.base_id, text="Too short.")
                    for row in request.assignments
                )
            )
        assert "Required structural corrections:" in neu_draft_prompt(request)
        return generated(request)

    output = tmp_path / "new"
    rows = run_neu_construction(output, source, generate)
    assert len(rows) == 3 and len(calls) == 2
    assert len(read_journal(output / "acceptance_failures.jsonl", NeuDraftFailure)) == 1
    assert run_neu_construction(output, source, generate) == rows
    assert len(calls) == 2
    manifest = write_neu_cases(output, rows, 42)
    assert len(manifest.cases) == 6 and manifest.cases == neu_cases(rows, 42)
    assert all(
        manifest.cases[index].text == manifest.cases[index + 1].text for index in range(0, 6, 2)
    )


def test_accepted_transaction_restores_partial_prefix_without_provider(tmp_path: Path) -> None:
    source = provenance(tmp_path, EmotionalDatasetConfig(utterance_count=3))
    output = tmp_path / "new"
    request = build_neu_requests(source.configuration)[0]
    rows = accept_neu_batch(request, generated(request), set())
    write_record(
        output / "accepted_batches" / f"{request.batch_id}.json",
        NeuAcceptedBatch(request=request, utterances=rows),
    )
    append_record(output / "utterances.jsonl", rows[0])

    def unavailable(request: NeuDraftRequest) -> GeneratedDraftBatch:
        raise AssertionError("Committed acceptance must avoid a second provider generation")

    assert run_neu_construction(output, source, unavailable) == rows
    assert read_journal(output / "utterances.jsonl", NeuUtterance) == rows


def test_old_manifest_hash_change_blocks_resume(tmp_path: Path) -> None:
    source = provenance(tmp_path, EmotionalDatasetConfig(utterance_count=1))
    source.previous_utterances.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned SHA256"):
        run_neu_construction(tmp_path / "new", source, generated)
