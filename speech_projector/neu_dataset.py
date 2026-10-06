"""Fresh paired utterances for the four approved NeuTTS emotional deliveries."""

from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import Field, model_validator

from scripts.inventory_results import write_record
from scripts.neutts_pilot_state import digest
from speech_projector.emotional_dataset import (
    Domain,
    EmotionalDatasetConfig,
    GeneratedDraftBatch,
    GeneratedDraftText,
    Intent,
    build_draft_requests,
    load_utterances,
    normalized_utterance,
    sentence_count,
)
from speech_projector.journal import append_record, read_journal
from speech_projector.models import Record, Split
from speech_projector.tts_pilot import PilotEmotion, TtsPilotCase, TtsPilotManifest

EMOTION_PAIRS = (
    (PilotEmotion.HAPPY, PilotEmotion.SAD),
    (PilotEmotion.HAPPY, PilotEmotion.FEARFUL),
    (PilotEmotion.HAPPY, PilotEmotion.ANGRY),
    (PilotEmotion.SAD, PilotEmotion.ANGRY),
)
EXPLICIT_EMOTION_WORDS = frozenset(
    (
        "happy",
        "happiness",
        "sad",
        "sadness",
        "angry",
        "anger",
        "fearful",
        "fear",
        "afraid",
        "furious",
        "excited",
        "frustrated",
        "worried",
        "scared",
        "anxious",
        "heartbroken",
        "sarcastic",
        "depressed",
        "unhappy",
        "terrified",
        "joyful",
        "delighted",
        "miserable",
    )
)


class NeuDraftAssignment(Record):
    base_id: str
    family_id: str
    split: Split
    domain: Domain
    intent: Intent
    situation: str
    fact_target: str
    emotions: tuple[PilotEmotion, PilotEmotion]

    @model_validator(mode="after")
    def validate_emotions(self) -> "NeuDraftAssignment":
        if self.emotions not in EMOTION_PAIRS:
            raise ValueError("Assignment must use one approved contrasting emotion pair")
        return self


class NeuDraftRequest(Record):
    batch_id: str
    configuration: EmotionalDatasetConfig
    assignments: tuple[NeuDraftAssignment, ...] = Field(min_length=1)
    feedback: tuple[str, ...] = ()


class NeuUtterance(Record):
    assignment: NeuDraftAssignment
    text: str = Field(min_length=1)


class NeuAcceptedBatch(Record):
    request: NeuDraftRequest
    utterances: tuple[NeuUtterance, ...]


class NeuDraftFailure(Record):
    request: NeuDraftRequest
    generated: GeneratedDraftBatch
    error: str


class NeuDatasetProvenance(Record):
    configuration: EmotionalDatasetConfig
    previous_utterances: Path
    previous_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_neu_requests(configuration: EmotionalDatasetConfig) -> tuple[NeuDraftRequest, ...]:
    requests = build_draft_requests(configuration)
    index = 0
    converted: list[NeuDraftRequest] = []
    for request in requests:
        assignments: list[NeuDraftAssignment] = []
        for assignment in request.assignments:
            assignments.append(
                NeuDraftAssignment(
                    base_id=f"neu_base_{index:05d}",
                    family_id=assignment.family_id,
                    split=assignment.split,
                    domain=assignment.domain,
                    intent=assignment.intent,
                    situation=assignment.situation,
                    fact_target=assignment.fact_target,
                    emotions=EMOTION_PAIRS[index % len(EMOTION_PAIRS)],
                )
            )
            index += 1
        converted.append(
            NeuDraftRequest(
                batch_id=f"neu_{request.batch_id}",
                configuration=configuration,
                assignments=tuple(assignments),
            )
        )
    return tuple(converted)


def neu_draft_prompt(request: NeuDraftRequest) -> str:
    prompt = (
        "Write a FRESH natural everyday USER utterance for every assigned ID. "
        "Each text must plausibly carry BOTH assigned emotional deliveries without changing "
        "any spoken words. The emotion assignments describe AUDIO delivery only: NEVER put "
        "them into the spoken text. Write neutral concrete events, plans, facts or requests. "
        "Do not say 'I feel happy', 'I am angry', 'I am sad', 'I am afraid', or combine "
        "opposing emotional self-descriptions. Example of a neutral literal utterance: "
        "'They asked me to give the presentation tomorrow, and everyone will be there.' "
        "The same words can sound happy or fearful without stating either feeling. "
        "Avoid evaluative good-news/bad-news explanations. "
        f"Aim for {request.configuration.preferred_min_words}–"
        f"{request.configuration.preferred_max_words} words; the required minimum is "
        f"{request.configuration.min_words} words. Use 1–3 sentences. "
        "Keep necessary facts inside the literal words; there is no hidden dialogue history. "
        "Use diverse concrete contexts and phrasing. Do not supply assistant replies, emotion "
        "labels, delivery explanations, stage directions or instructions in the utterances. "
        "Return JSON only, every assigned ID exactly once and in order. "
        "Forbidden emotion-label words in literal text: "
        + ", ".join(sorted(EXPLICIT_EMOTION_WORDS))
        + ".\n\nAssignments:\n["
        + ",".join(assignment.model_dump_json() for assignment in request.assignments)
        + "]"
    )
    if request.feedback:
        prompt += "\n\nRequired structural corrections:\n" + "\n".join(request.feedback)
        prompt += (
            "\nReplace the offending rows with new distinct wording. Return all assigned IDs "
            "in order; preserve valid rows where possible."
        )
    return prompt + '\n\nReturn {"utterances":[{"base_id":"assigned ID","text":"literal text"}]}'


def accept_neu_batch(
    request: NeuDraftRequest, generated: GeneratedDraftBatch, previous_texts: set[str]
) -> tuple[NeuUtterance, ...]:
    if tuple(row.base_id for row in generated.utterances) != tuple(
        assignment.base_id for assignment in request.assignments
    ):
        raise ValueError("Generated IDs must match the exact ordered assignments")
    unique = previous_texts.copy()
    errors: list[str] = []
    for row in generated.utterances:
        normalized = normalized_utterance(row.text)
        words = len(normalized.split())
        sentences = sentence_count(row.text)
        emotion_words = explicit_emotion_words(row.text)
        if emotion_words:
            errors.append(
                f"{row.base_id}: explicit emotion-label words={emotion_words}; "
                "replace them with neutral concrete spoken facts or requests"
            )
        if words < request.configuration.min_words:
            errors.append(
                f"{row.base_id}: word_count={words}; minimum={request.configuration.min_words}"
            )
        if not 1 <= sentences <= 3:
            errors.append(f"{row.base_id}: sentence_count={sentences}; required range=1–3")
        if normalized in unique:
            errors.append(f"{row.base_id}: exact normalized duplicate of an old/new utterance")
        unique.add(normalized)
    if errors:
        raise ValueError("; ".join(errors))
    return tuple(
        NeuUtterance(assignment=assignment, text=row.text)
        for assignment, row in zip(request.assignments, generated.utterances, strict=True)
    )


def explicit_emotion_words(text: str) -> tuple[str, ...]:
    return tuple(sorted(set(normalized_utterance(text).split()) & EXPLICIT_EMOTION_WORDS))


def validate_neu_provenance(directory: Path, provenance: NeuDatasetProvenance) -> set[str]:
    if digest(provenance.previous_utterances) != provenance.previous_sha256:
        raise ValueError("Older utterance manifest differs from its pinned SHA256")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "dataset_provenance.json"
    if path.exists():
        if NeuDatasetProvenance.model_validate_json(path.read_bytes()) != provenance:
            raise ValueError("Dataset resume configuration or older source identity differs")
    else:
        write_record(path, provenance)
    return {
        normalized_utterance(row.text) for row in load_utterances(provenance.previous_utterances)
    }


def run_neu_construction(
    directory: Path,
    provenance: NeuDatasetProvenance,
    generate: Callable[[NeuDraftRequest], GeneratedDraftBatch],
    limit: int | None = None,
) -> tuple[NeuUtterance, ...]:
    if limit is not None and (
        not 1 <= limit <= provenance.configuration.utterance_count
        or (
            limit % provenance.configuration.batch_size
            and limit != provenance.configuration.utterance_count
        )
    ):
        raise ValueError(
            "Draft limit must be a complete batch boundary within the configured quota"
        )
    old_texts = validate_neu_provenance(directory, provenance)
    requests = build_neu_requests(provenance.configuration)
    plan_path = directory / "plan.jsonl"
    previous_plan = read_journal(plan_path, NeuDraftRequest)
    if previous_plan != requests[: len(previous_plan)]:
        raise ValueError("Saved Neu plan differs from the current deterministic quota plan")
    for request in requests[len(previous_plan) :]:
        append_record(plan_path, request)
    path = directory / "utterances.jsonl"
    rows = list(read_journal(path, NeuUtterance))
    assignments = tuple(assignment for request in requests for assignment in request.assignments)
    if len(rows) > len(assignments):
        raise ValueError("Saved Neu journal exceeds the quota plan")
    seen = old_texts.copy()
    for assignment, row in zip(assignments[: len(rows)], rows, strict=True):
        validated = accept_neu_batch(
            NeuDraftRequest(
                batch_id="resume", configuration=provenance.configuration, assignments=(assignment,)
            ),
            GeneratedDraftBatch(
                utterances=(GeneratedDraftText(base_id=assignment.base_id, text=row.text),)
            ),
            seen,
        )
        if validated != (row,):
            raise ValueError("Saved Neu utterance differs from its planned assignment")
        seen.add(normalized_utterance(row.text))
    start = 0
    for request in requests:
        end = start + len(request.assignments)
        if limit is not None and end > limit:
            break
        if end <= len(rows):
            start = end
            continue
        transaction_path = directory / "accepted_batches" / f"{request.batch_id}.json"
        if transaction_path.exists():
            transaction = NeuAcceptedBatch.model_validate_json(transaction_path.read_bytes())
        else:
            current = request
            for attempt in range(provenance.configuration.max_acceptance_attempts):
                generated = generate(current)
                try:
                    accepted = accept_neu_batch(
                        request,
                        generated,
                        old_texts | {normalized_utterance(row.text) for row in rows[:start]},
                    )
                    transaction = NeuAcceptedBatch(request=request, utterances=accepted)
                    break
                except ValueError as error:
                    append_record(
                        directory / "acceptance_failures.jsonl",
                        NeuDraftFailure(request=current, generated=generated, error=str(error)),
                    )
                    if attempt + 1 == provenance.configuration.max_acceptance_attempts:
                        raise ValueError(
                            f"Required structural repairs exhausted for {request.batch_id}"
                        ) from error
                    current = NeuDraftRequest(
                        batch_id=f"{request.batch_id}_repair_{attempt + 1}",
                        configuration=request.configuration,
                        assignments=request.assignments,
                        feedback=(
                            str(error),
                            "Previous invalid JSON: " + generated.model_dump_json(),
                        ),
                    )
            else:
                raise AssertionError("Positive repair budget must return or raise")
            write_record(transaction_path, transaction)
        validated = accept_neu_batch(
            request,
            GeneratedDraftBatch(
                utterances=tuple(
                    GeneratedDraftText(base_id=row.assignment.base_id, text=row.text)
                    for row in transaction.utterances
                )
            ),
            old_texts | {normalized_utterance(row.text) for row in rows[:start]},
        )
        if transaction.request != request or transaction.utterances != validated:
            raise ValueError("Accepted transaction differs from its planned provenance")
        if tuple(rows[start:]) != validated[: len(rows) - start]:
            raise ValueError("Accepted transaction conflicts with the already committed prefix")
        for row in validated[len(rows) - start :]:
            append_record(path, row)
            rows.append(row)
        print(f"Accepted fresh Neu utterances: {len(rows)}/{len(assignments)}", flush=True)
        start = end
    return tuple(rows)


def neu_cases(utterances: Sequence[NeuUtterance], seed: int) -> tuple[TtsPilotCase, ...]:
    return tuple(
        TtsPilotCase(
            case_id=f"{row.assignment.base_id}_{emotion.value}",
            utterance_id=row.assignment.base_id,
            text=row.text,
            emotion=emotion,
            seed=seed,
        )
        for row in utterances
        for emotion in row.assignment.emotions
    )


def write_neu_cases(
    directory: Path, utterances: Sequence[NeuUtterance], seed: int
) -> TtsPilotManifest:
    manifest = TtsPilotManifest(
        cases=neu_cases(utterances, seed),
        warmup_text="I have the details ready for our appointment tomorrow morning.",
        warmup_seed=max(seed - 1, 0),
    )
    write_record(directory / "cases.json", manifest)
    return manifest
