"""Quota, literal-text boundary and crash-resume checks for paired drafts."""

from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from speech_projector.emotion_preview import Delivery
from speech_projector.emotional_dataset import (
    DeliveryAnnotation,
    Domain,
    DraftAcceptanceFailure,
    DraftAssignment,
    DraftRequest,
    EmotionalDatasetConfig,
    EmotionalUtterance,
    GeneratedDraftBatch,
    GeneratedDraftText,
    Intent,
    accept_draft_batch,
    build_draft_requests,
    descriptive_delivery,
    draft_prompt,
    expressive_instruction,
    load_utterances,
    normalized_utterance,
    run_draft_construction,
    sentence_count,
)
from speech_projector.journal import append_record, read_journal
from speech_projector.models import Record


def generated(request: DraftRequest) -> GeneratedDraftBatch:
    return GeneratedDraftBatch(
        utterances=tuple(
            GeneratedDraftText(
                base_id=assignment.base_id,
                text=f"Please help me find the missing parcel numbered {assignment.base_id} today.",
            )
            for assignment in request.assignments
        )
    )


def test_full_quota_and_family_split_are_deterministic() -> None:
    configuration = EmotionalDatasetConfig()
    requests = build_draft_requests(configuration)
    assert requests == build_draft_requests(configuration)
    assert len(requests) == 500
    assignments = tuple(assignment for request in requests for assignment in request.assignments)
    assert len(assignments) == len({assignment.base_id for assignment in assignments}) == 5000
    assert Counter((assignment.domain, assignment.intent) for assignment in assignments) == Counter(
        {(domain, intent): 50 for domain in Domain for intent in Intent}
    )
    assert len({assignment.family_id for assignment in assignments}) == 500
    assert len({(assignment.family_id, assignment.split) for assignment in assignments}) == 500
    assert all(len(set(assignment.deliveries)) == 2 for assignment in assignments)
    assert all(Delivery.WORRIED not in assignment.deliveries for assignment in assignments)


def test_prompt_schema_contains_literal_outputs_without_explanations() -> None:
    request = build_draft_requests(EmotionalDatasetConfig(utterance_count=10))[0]
    prompt = draft_prompt(request)
    assert "there is no hidden dialogue history" in prompt
    assert "Do not write delivery explanations" in prompt
    assert "BOTH assigned deliveries" in prompt
    assert set(GeneratedDraftText.model_fields) == {"base_id", "text"}
    assert "fact_target" in prompt and "JSON schema" in prompt


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Please help me move these boxes before tomorrow.", 1),
        ("The order arrived. Please help me check it.", 2),
        ("It changed. I saw it. Could you check it?", 3),
    ],
)
def test_sentence_counts(text: str, expected: int) -> None:
    assert sentence_count(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Too few words here.",
        "One sentence here. Another sentence here. Third one here. Fourth one here.",
    ],
)
def test_only_required_length_and_sentence_boundaries_reject(text: str) -> None:
    request = build_draft_requests(EmotionalDatasetConfig(utterance_count=1))[0]
    batch = GeneratedDraftBatch(
        utterances=(GeneratedDraftText(base_id=request.assignments[0].base_id, text=text),)
    )
    with pytest.raises(ValueError):
        accept_draft_batch(request, batch, ())


def test_preferred_length_does_not_add_an_editorial_rejection() -> None:
    request = build_draft_requests(EmotionalDatasetConfig(utterance_count=1))[0]
    text = " ".join(["Please"] + ["check"] * 60 + ["the package before tomorrow."])
    accepted = accept_draft_batch(
        request,
        GeneratedDraftBatch(
            utterances=(GeneratedDraftText(base_id=request.assignments[0].base_id, text=text),)
        ),
        (),
    )
    assert accepted[0].text == text
    assert len(accepted[0].deliveries) == 2
    assert all(
        "exactly the supplied words" in delivery.instruct for delivery in accepted[0].deliveries
    )


def test_normalized_exact_duplicates_rejected_atomically() -> None:
    request = build_draft_requests(EmotionalDatasetConfig(utterance_count=2))[0]
    first, second = request.assignments
    batch = GeneratedDraftBatch(
        utterances=(
            GeneratedDraftText(
                base_id=first.base_id, text="I'm checking the same parcel again this morning."
            ),
            GeneratedDraftText(
                base_id=second.base_id, text="I’m checking the SAME parcel again this morning!"
            ),
        )
    )
    assert normalized_utterance(batch.utterances[0].text) == normalized_utterance(
        batch.utterances[1].text
    )
    with pytest.raises(ValueError, match="duplicate"):
        accept_draft_batch(request, batch, ())


def test_existing_text_and_wrong_id_order_rejected() -> None:
    request = build_draft_requests(EmotionalDatasetConfig(utterance_count=2))[0]
    batch = generated(request)
    existing = accept_draft_batch(request, batch, ())
    with pytest.raises(ValueError, match="duplicate"):
        accept_draft_batch(request, batch, existing)
    with pytest.raises(ValueError, match="ordered assignments"):
        accept_draft_batch(
            request, GeneratedDraftBatch(utterances=tuple(reversed(batch.utterances))), ()
        )


def test_completed_construction_resumes_without_generation(tmp_path: Path) -> None:
    configuration = EmotionalDatasetConfig(utterance_count=13, batch_size=5)
    calls: list[str] = []

    def generate(request: DraftRequest) -> GeneratedDraftBatch:
        calls.append(request.batch_id)
        return generated(request)

    first = run_draft_construction(tmp_path, configuration, generate)
    assert len(first) == 13 and len(calls) == 3
    assert run_draft_construction(tmp_path, configuration, generate) == first
    assert len(calls) == 3
    assert load_utterances(tmp_path / "utterances.jsonl") == first


def test_partial_batch_resume_uses_original_full_request(tmp_path: Path) -> None:
    configuration = EmotionalDatasetConfig(utterance_count=5)
    request = build_draft_requests(configuration)[0]
    accepted = accept_draft_batch(request, generated(request), ())
    append_record(tmp_path / "utterances.jsonl", accepted[0])
    with (tmp_path / "utterances.jsonl").open("ab") as stream:
        stream.write(b'{"unfinished":')
    seen: list[DraftRequest] = []

    def generate(remaining: DraftRequest) -> GeneratedDraftBatch:
        seen.append(remaining)
        return generated(remaining)

    assert run_draft_construction(tmp_path, configuration, generate) == accepted
    assert seen == [request]
    assert (tmp_path / "journal_recovery.jsonl").exists()


def test_resume_rejects_changed_configuration_and_cached_text(tmp_path: Path) -> None:
    configuration = EmotionalDatasetConfig(utterance_count=2)
    request = build_draft_requests(configuration)[0]
    first = accept_draft_batch(request, generated(request), ())[0]
    append_record(tmp_path / "utterances.jsonl", first)

    def altered(request: DraftRequest) -> GeneratedDraftBatch:
        batch = generated(request)
        changed = GeneratedDraftText(
            base_id=first.base_id, text="Please help me find a completely different parcel today."
        )
        return GeneratedDraftBatch(utterances=(changed, batch.utterances[1]))

    with pytest.raises(ValueError, match="accepted prefix"):
        run_draft_construction(tmp_path, configuration, altered)
    with pytest.raises(ValueError, match="configuration changed"):
        run_draft_construction(tmp_path, EmotionalDatasetConfig(utterance_count=3), generated)


@pytest.mark.parametrize(
    "deliveries", [(Delivery.HAPPY, Delivery.HAPPY), (Delivery.NEUTRAL, Delivery.WORRIED)]
)
def test_invalid_delivery_pairs_rejected(deliveries: tuple[Delivery, Delivery]) -> None:
    assignment = build_draft_requests(EmotionalDatasetConfig(utterance_count=1))[0].assignments[0]
    with pytest.raises(ValidationError):
        DraftAssignment(**(assignment.model_dump() | {"deliveries": deliveries}))
    with pytest.raises(ValueError):
        expressive_instruction(Delivery.WORRIED)


def test_canonical_boundary_has_exactly_two_delivery_annotations() -> None:
    request = build_draft_requests(EmotionalDatasetConfig(utterance_count=1))[0]
    row = accept_draft_batch(request, generated(request), ())[0]
    assert EmotionalUtterance.model_validate_json(row.model_dump_json()) == row
    with pytest.raises(ValidationError):
        EmotionalUtterance(
            **(
                row.model_dump()
                | {
                    "deliveries": (
                        DeliveryAnnotation(delivery=Delivery.NEUTRAL, instruct="Speak clearly."),
                    )
                }
            )
        )
    with pytest.raises(ValidationError):
        GeneratedDraftText(base_id="test", text="   ")


def test_required_duplicate_repair_changes_prompt_and_preserves_failure(tmp_path: Path) -> None:
    configuration = EmotionalDatasetConfig(utterance_count=2)
    requests_seen: list[DraftRequest] = []

    def generate(request: DraftRequest) -> GeneratedDraftBatch:
        requests_seen.append(request)
        if request.feedback:
            return generated(request)
        return GeneratedDraftBatch(
            utterances=tuple(
                GeneratedDraftText(
                    base_id=assignment.base_id,
                    text="Please help me find the missing package before tomorrow.",
                )
                for assignment in request.assignments
            )
        )

    rows = run_draft_construction(tmp_path, configuration, generate)
    assert len(rows) == 2 and len(requests_seen) == 2
    assert requests_seen[1].batch_id == "batch_0000_repair_1"
    assert "duplicate" in draft_prompt(requests_seen[1])
    failures = read_journal(tmp_path / "acceptance_failures.jsonl", DraftAcceptanceFailure)
    assert len(failures) == 1 and "duplicate" in failures[0].error
    assert (tmp_path / "accepted_batches/batch_0000.json").exists()


def test_repaired_batch_transaction_survives_mid_append_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = EmotionalDatasetConfig(utterance_count=2)
    calls: list[DraftRequest] = []

    def generate(request: DraftRequest) -> GeneratedDraftBatch:
        calls.append(request)
        return generated(request)

    def interrupted_append(path: Path, record: Record) -> None:
        append_record(path, record)
        if path.name == "utterances.jsonl":
            raise RuntimeError("test-local interruption after durable first row")

    monkeypatch.setattr("speech_projector.emotional_dataset.append_record", interrupted_append)
    with pytest.raises(RuntimeError, match="interruption"):
        run_draft_construction(tmp_path, configuration, generate)
    assert len(load_utterances(tmp_path / "utterances.jsonl")) == 1
    monkeypatch.setattr("speech_projector.emotional_dataset.append_record", append_record)
    assert len(run_draft_construction(tmp_path, configuration, generate)) == 2
    assert len(calls) == 1


def test_structural_retries_are_bounded_without_quality_judging(tmp_path: Path) -> None:
    configuration = EmotionalDatasetConfig(utterance_count=1, max_acceptance_attempts=2)

    def too_short(request: DraftRequest) -> GeneratedDraftBatch:
        return GeneratedDraftBatch(
            utterances=(
                GeneratedDraftText(base_id=request.assignments[0].base_id, text="Too short."),
            )
        )

    with pytest.raises(ValueError, match="corrections exhausted"):
        run_draft_construction(tmp_path, configuration, too_short)
    assert len(read_journal(tmp_path / "acceptance_failures.jsonl", DraftAcceptanceFailure)) == 2
    assert load_utterances(tmp_path / "utterances.jsonl") == ()
    assert "Say exactly" not in descriptive_delivery(Delivery.HAPPY)
    assert "Speak" not in descriptive_delivery(Delivery.HAPPY)
