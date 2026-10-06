"""Typed quota plan and resumable construction of paired emotional utterances."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from scripts.inventory_results import write_record
from speech_projector.emotion_preview import Delivery
from speech_projector.journal import append_record, read_journal
from speech_projector.models import Record, Split


class Domain(str, Enum):
    HOME = "home"
    WORK = "work"
    TRAVEL = "travel"
    SHOPPING = "shopping"
    FOOD = "food"
    FRIENDS = "friends"
    STUDY = "study"
    TECHNOLOGY = "technology"
    HOBBIES = "hobbies"
    APPOINTMENTS = "appointments"


class Intent(str, Enum):
    REQUEST_HELP = "request_help"
    GIVE_UPDATE = "give_update"
    SEEK_CLARIFICATION = "seek_clarification"
    CORRECT_INFORMATION = "correct_information"
    DECLINE = "decline"
    THANK = "thank"
    ASSESS_RESULT = "assess_result"
    WAIT = "wait"
    SET_BOUNDARY = "set_boundary"
    SHARE_EXPECTATION = "share_expectation"


class EmotionalDatasetConfig(Record):
    utterance_count: int = Field(default=5000, gt=0, le=5000)
    batch_size: int = Field(default=10, gt=0, le=50)
    seed: int = Field(default=42, ge=0)
    min_words: int = Field(default=7, ge=7)
    preferred_max_words: int = Field(default=45, ge=7)
    max_acceptance_attempts: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def validate_length_guidance(self) -> EmotionalDatasetConfig:
        if self.preferred_max_words < self.min_words:
            raise ValueError("Preferred length must be at least the required minimum")
        return self


class DraftAssignment(Record):
    base_id: str
    family_id: str
    split: Split
    domain: Domain
    intent: Intent
    situation: str
    fact_target: str
    deliveries: tuple[Delivery, Delivery]

    @field_validator("deliveries")
    @classmethod
    def validate_deliveries(
        cls, deliveries: tuple[Delivery, Delivery]
    ) -> tuple[Delivery, Delivery]:
        if deliveries[0] == deliveries[1] or Delivery.WORRIED in deliveries:
            raise ValueError("Each assignment needs two distinct approved deliveries")
        return deliveries


class DraftRequest(Record):
    batch_id: str
    configuration: EmotionalDatasetConfig
    assignments: tuple[DraftAssignment, ...] = Field(min_length=1)
    feedback: tuple[str, ...] = ()


class GeneratedDraftText(Record):
    base_id: str
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("Generated utterance must not be empty")
        return text.strip()


class GeneratedDraftBatch(Record):
    utterances: tuple[GeneratedDraftText, ...] = Field(min_length=1)


class DraftAcceptanceFailure(Record):
    request: DraftRequest
    generated: GeneratedDraftBatch
    error: str


class DeliveryAnnotation(Record):
    delivery: Delivery
    instruct: str = Field(min_length=1)


class EmotionalUtterance(Record):
    base_id: str
    family_id: str
    split: Split
    domain: Domain
    intent: Intent
    text: str = Field(min_length=1)
    deliveries: tuple[DeliveryAnnotation, DeliveryAnnotation]

    @model_validator(mode="after")
    def validate_deliveries(self) -> EmotionalUtterance:
        tones = tuple(annotation.delivery for annotation in self.deliveries)
        if tones[0] == tones[1] or Delivery.WORRIED in tones:
            raise ValueError("Each utterance requires exactly two distinct approved deliveries")
        return self


class AcceptedDraftBatch(Record):
    request: DraftRequest
    utterances: tuple[EmotionalUtterance, ...] = Field(min_length=1)


@dataclass(frozen=True)
class DomainScenarios:
    domain: Domain
    situations: tuple[str, str, str, str, str]
    facts: tuple[str, str, str, str, str, str, str, str, str, str]


SCENARIOS = (
    DomainScenarios(
        Domain.HOME,
        (
            "sharing household chores",
            "arranging a repair",
            "moving furniture",
            "organizing storage",
            "coordinating a delivery",
        ),
        (
            "the hallway shelf",
            "a leaking kitchen tap",
            "two spare chairs",
            "the laundry basket",
            "the upstairs window",
            "a missing toolbox",
            "three cardboard boxes",
            "the spare room",
            "a replacement door handle",
            "the Saturday cleaning schedule",
        ),
    ),
    DomainScenarios(
        Domain.WORK,
        (
            "coordinating a small project",
            "reviewing a draft",
            "sharing a task",
            "arranging a meeting",
            "handing over work",
        ),
        (
            "a revised deadline",
            "the final paragraph",
            "two missing attachments",
            "a shared spreadsheet",
            "the Tuesday meeting",
            "a colleague's comments",
            "the client summary",
            "a presentation slide",
            "a printer reservation",
            "the afternoon handover",
        ),
    ),
    DomainScenarios(
        Domain.TRAVEL,
        (
            "planning a short trip",
            "checking a route",
            "waiting for transport",
            "changing a booking",
            "meeting someone on arrival",
        ),
        (
            "the last bus",
            "platform four",
            "a delayed train",
            "the side entrance",
            "a return ticket",
            "the station lockers",
            "a changed departure time",
            "the airport shuttle",
            "a window seat",
            "the walking route through the park",
        ),
    ),
    DomainScenarios(
        Domain.SHOPPING,
        (
            "choosing an everyday item",
            "collecting an order",
            "returning a purchase",
            "comparing available options",
            "finding a replacement",
        ),
        (
            "the blue jacket",
            "a missing receipt",
            "the smaller size",
            "a collection code",
            "the last item on the shelf",
            "a scratched saucepan",
            "a reusable shopping bag",
            "the advertised discount",
            "a different brand",
            "the order confirmation",
        ),
    ),
    DomainScenarios(
        Domain.FOOD,
        (
            "planning dinner",
            "cooking together",
            "ordering a meal",
            "sharing leftovers",
            "preparing food for guests",
        ),
        (
            "the remaining rice",
            "a missing ingredient",
            "the oven timer",
            "a table for three",
            "the vegetarian option",
            "two extra portions",
            "the grocery list",
            "a new soup recipe",
            "the packed lunch",
            "the slightly burnt toast",
        ),
    ),
    DomainScenarios(
        Domain.FRIENDS,
        (
            "arranging a meetup",
            "checking in after an event",
            "changing shared plans",
            "borrowing an item",
            "ending a conversation",
        ),
        (
            "the group message",
            "a borrowed book",
            "the concert tickets",
            "the usual cafe",
            "a last-minute invitation",
            "the weekend picnic",
            "an unanswered message",
            "the evening walk",
            "a board-game night",
            "the birthday gathering",
        ),
    ),
    DomainScenarios(
        Domain.STUDY,
        (
            "working on an assignment",
            "preparing for a class",
            "studying with a partner",
            "reviewing feedback",
            "organizing course materials",
        ),
        (
            "the practice questions",
            "a marked essay",
            "the library desk",
            "the lecture recording",
            "two pages of notes",
            "the project outline",
            "a borrowed textbook",
            "the online quiz",
            "the study-group schedule",
            "the final example in the worksheet",
        ),
    ),
    DomainScenarios(
        Domain.TECHNOLOGY,
        (
            "fixing an everyday device",
            "sharing a digital file",
            "setting up an account",
            "changing an app setting",
            "checking a connection",
        ),
        (
            "the Wi-Fi password",
            "an unsent attachment",
            "a low phone battery",
            "the video-call link",
            "a forgotten login",
            "the notification setting",
            "a stuck software update",
            "the backup folder",
            "a loose charging cable",
            "the shared calendar",
        ),
    ),
    DomainScenarios(
        Domain.HOBBIES,
        (
            "practicing a skill",
            "planning a leisure activity",
            "reviewing something made",
            "sharing equipment",
            "joining a local group",
        ),
        (
            "a half-finished drawing",
            "the guitar strings",
            "a new running route",
            "the garden seedlings",
            "a blurry photograph",
            "the knitting pattern",
            "the club's meeting time",
            "a borrowed tennis racket",
            "the unfinished puzzle",
            "a small pottery bowl",
        ),
    ),
    DomainScenarios(
        Domain.APPOINTMENTS,
        (
            "booking an everyday service",
            "changing a time slot",
            "waiting for a callback",
            "checking directions",
            "confirming arrangements",
        ),
        (
            "the morning slot",
            "a confirmation email",
            "the front desk",
            "a missed callback",
            "the new address",
            "a thirty-minute delay",
            "the reminder message",
            "the Friday booking",
            "a cancellation notice",
            "the queue number",
        ),
    ),
)


def compatible_pairs(intent: Intent) -> tuple[tuple[Delivery, Delivery], ...]:
    match intent:
        case Intent.REQUEST_HELP | Intent.WAIT | Intent.SET_BOUNDARY:
            return ((Delivery.NEUTRAL, Delivery.FRUSTRATED), (Delivery.NEUTRAL, Delivery.SAD))
        case Intent.GIVE_UPDATE | Intent.SHARE_EXPECTATION:
            return ((Delivery.HAPPY, Delivery.SAD), (Delivery.NEUTRAL, Delivery.FRUSTRATED))
        case Intent.SEEK_CLARIFICATION | Intent.CORRECT_INFORMATION:
            return ((Delivery.NEUTRAL, Delivery.FRUSTRATED), (Delivery.NEUTRAL, Delivery.SARCASTIC))
        case Intent.DECLINE:
            return ((Delivery.NEUTRAL, Delivery.SAD), (Delivery.NEUTRAL, Delivery.FRUSTRATED))
        case Intent.THANK | Intent.ASSESS_RESULT:
            return ((Delivery.HAPPY, Delivery.SARCASTIC), (Delivery.HAPPY, Delivery.FRUSTRATED))


def descriptive_delivery(delivery: Delivery) -> str:
    match delivery:
        case Delivery.NEUTRAL:
            return "an even, matter-of-fact conversational tone"
        case Delivery.HAPPY:
            return "clear happiness and pleasure, bright energy and an audible warm smile"
        case Delivery.SAD:
            return "clear sadness and disappointment, subdued energy and a slower, softer voice"
        case Delivery.FRUSTRATED:
            return "clear frustration, audible tension, sharper emphasis and controlled irritation"
        case Delivery.SARCASTIC:
            return "unmistakable dry sarcasm, ironic emphasis and deliberate contrasting intonation"
        case Delivery.WORRIED:
            raise ValueError("Worried delivery is excluded from the initial dataset")


def expressive_instruction(delivery: Delivery) -> str:
    return f"Speak with {descriptive_delivery(delivery)}. " + (
        "Say exactly the supplied words; do not add words, laughter, sighs or stage directions."
    )


def family_split(family_id: str, seed: int) -> Split:
    value = int(hashlib.sha256(f"{seed}:{family_id}".encode()).hexdigest()[:8], 16) % 10000
    if value < 9000:
        return Split.TRAIN
    if value < 9500:
        return Split.VALIDATION
    return Split.TEST


def build_draft_requests(configuration: EmotionalDatasetConfig) -> tuple[DraftRequest, ...]:
    assignments: list[DraftAssignment] = []
    for variation in range(50):
        for domain_index, scenario in enumerate(SCENARIOS):
            family = f"family_{scenario.domain.value}_{variation:02d}"
            for intent_index, intent in enumerate(Intent):
                if len(assignments) == configuration.utterance_count:
                    break
                pairs = compatible_pairs(intent)
                assignments.append(
                    DraftAssignment(
                        base_id=f"utterance_{len(assignments):05d}",
                        family_id=family,
                        split=family_split(family, configuration.seed),
                        domain=scenario.domain,
                        intent=intent,
                        situation=scenario.situations[variation // 10],
                        fact_target=scenario.facts[variation % 10],
                        deliveries=pairs[(variation + domain_index + intent_index) % len(pairs)],
                    )
                )
    return tuple(
        DraftRequest(
            batch_id=f"batch_{index // configuration.batch_size:04d}",
            configuration=configuration,
            assignments=tuple(assignments[index : index + configuration.batch_size]),
        )
        for index in range(0, len(assignments), configuration.batch_size)
    )


def draft_prompt(request: DraftRequest) -> str:
    instructions = (
        "Write one distinct, natural everyday USER utterance for each assigned ID. "
        f"Use 1–3 sentences and at least {request.configuration.min_words} words; "
        f"prefer no more than {request.configuration.preferred_max_words} words. "
        "Each literal utterance must be plausible with BOTH assigned deliveries. "
        "Keep necessary context or facts inside the spoken words: there is no hidden dialogue "
        "history. Use varied phrasing, concrete details and pragmatic intentions. "
        "Do not write delivery explanations, stage directions, role labels, assistant replies "
        "or emotion instructions. Return JSON only matching the supplied schema, in ID order.\n\n"
    )
    prompt = (
        instructions
        + "Assignments:\n"
        + request.model_dump_json()
        + "\n\nJSON schema:\n"
        + (json.dumps(GeneratedDraftBatch.model_json_schema(), ensure_ascii=False))
    )
    if request.feedback:
        prompt += "\n\nRequired structural correction:\n" + "\n".join(request.feedback)
    return prompt


def normalized_utterance(text: str) -> str:
    return " ".join(re.findall(r"\w+(?:'\w+)*", text.casefold().replace("’", "'")))


def sentence_count(text: str) -> int:
    return len(tuple(part for part in re.split(r"[.!?]+(?=\s|$)", text) if part.strip()))


def validate_literal_text(text: str, configuration: EmotionalDatasetConfig) -> str:
    normalized = normalized_utterance(text)
    if len(normalized.split()) < configuration.min_words:
        raise ValueError("Utterance has fewer than the required words")
    if not 1 <= sentence_count(text) <= 3:
        raise ValueError("Utterance must contain 1–3 sentences")
    return normalized


def assemble_utterance(assignment: DraftAssignment, text: str) -> EmotionalUtterance:
    first, second = assignment.deliveries
    return EmotionalUtterance(
        base_id=assignment.base_id,
        family_id=assignment.family_id,
        split=assignment.split,
        domain=assignment.domain,
        intent=assignment.intent,
        text=text,
        deliveries=(
            DeliveryAnnotation(delivery=first, instruct=expressive_instruction(first)),
            DeliveryAnnotation(delivery=second, instruct=expressive_instruction(second)),
        ),
    )


def accept_draft_batch(
    request: DraftRequest, batch: GeneratedDraftBatch, existing: Sequence[EmotionalUtterance]
) -> tuple[EmotionalUtterance, ...]:
    if tuple(item.base_id for item in batch.utterances) != tuple(
        assignment.base_id for assignment in request.assignments
    ):
        raise ValueError("Generated IDs must match the exact ordered assignments")
    unique_texts = {normalized_utterance(item.text) for item in existing}
    accepted: list[EmotionalUtterance] = []
    for assignment, draft in zip(request.assignments, batch.utterances, strict=True):
        text_key = validate_literal_text(draft.text, request.configuration)
        if text_key in unique_texts:
            raise ValueError(f"Exact normalized text duplicate: {draft.base_id}")
        unique_texts.add(text_key)
        accepted.append(assemble_utterance(assignment, draft.text))
    return tuple(accepted)


def load_utterances(path: Path) -> tuple[EmotionalUtterance, ...]:
    return read_journal(path, EmotionalUtterance)


def generate_accepted_batch(
    request: DraftRequest,
    existing: Sequence[EmotionalUtterance],
    start: int,
    generate: Callable[[DraftRequest], GeneratedDraftBatch],
    failure_path: Path,
) -> AcceptedDraftBatch:
    completed = tuple(existing[start:])
    pending = request.assignments[len(completed) :]
    current = request
    for attempt in range(request.configuration.max_acceptance_attempts):
        generated = generate(current)
        if not current.feedback and tuple(item.base_id for item in generated.utterances) == tuple(
            assignment.base_id for assignment in current.assignments
        ):
            for row, draft in zip(completed, generated.utterances[: len(completed)], strict=True):
                if (row.base_id, row.text) != (draft.base_id, draft.text):
                    raise ValueError(
                        "Resumed generated batch differs from its already accepted prefix"
                    )
        try:
            if tuple(draft.base_id for draft in generated.utterances) != tuple(
                assignment.base_id for assignment in current.assignments
            ):
                raise ValueError("Generated IDs must match the exact ordered assignments")
            missing = (
                generated.utterances[len(completed) :]
                if not current.feedback
                else generated.utterances
            )
            accepted = accept_draft_batch(
                DraftRequest(
                    batch_id=current.batch_id,
                    configuration=request.configuration,
                    assignments=pending,
                    feedback=current.feedback,
                ),
                GeneratedDraftBatch(utterances=missing),
                existing,
            )
            return AcceptedDraftBatch(request=current, utterances=completed + accepted)
        except ValueError as error:
            append_record(
                failure_path,
                DraftAcceptanceFailure(
                    request=current,
                    generated=generated,
                    error=str(error),
                ),
            )
            if attempt + 1 == request.configuration.max_acceptance_attempts:
                raise ValueError(
                    f"Required draft corrections exhausted for {request.batch_id}"
                ) from error
            current = DraftRequest(
                batch_id=f"{request.batch_id}_repair_{attempt + 1}",
                configuration=request.configuration,
                assignments=pending,
                feedback=(
                    str(error),
                    "Replace the invalid rows; keep these pending IDs in order.",
                    "Previous invalid JSON: " + generated.model_dump_json(),
                ),
            )
    raise AssertionError("Positive acceptance budget must return or raise")


def run_draft_construction(
    output_directory: Path,
    configuration: EmotionalDatasetConfig,
    generate: Callable[[DraftRequest], GeneratedDraftBatch],
) -> tuple[EmotionalUtterance, ...]:
    output_directory.mkdir(parents=True, exist_ok=True)
    configuration_path = output_directory / "configuration.json"
    if configuration_path.exists():
        previous = EmotionalDatasetConfig.model_validate_json(configuration_path.read_bytes())
        if previous != configuration:
            raise ValueError("Dataset configuration changed; use a new output directory")
    else:
        write_record(configuration_path, configuration)
    requests = build_draft_requests(configuration)
    plan_path = output_directory / "plan.jsonl"
    previous_plan = read_journal(plan_path, DraftRequest)
    if previous_plan != requests[: len(previous_plan)]:
        raise ValueError("Saved quota plan differs from the current construction plan")
    for request in requests[len(previous_plan) :]:
        append_record(plan_path, request)
    path = output_directory / "utterances.jsonl"
    previous_rows = load_utterances(path)
    all_assignments = tuple(
        assignment for request in requests for assignment in request.assignments
    )
    if len(previous_rows) > len(all_assignments):
        raise ValueError("Dataset journal exceeds the configured utterance count")
    unique_texts: set[str] = set()
    for assignment, row in zip(all_assignments[: len(previous_rows)], previous_rows, strict=True):
        normalized = validate_literal_text(row.text, configuration)
        if normalized in unique_texts:
            raise ValueError("Saved dataset contains an exact normalized duplicate")
        unique_texts.add(normalized)
        if assemble_utterance(assignment, row.text) != row:
            raise ValueError("Saved utterance differs from its assigned provenance or deliveries")
    rows = list(previous_rows)
    start = 0
    for request in requests:
        end = start + len(request.assignments)
        if end <= len(rows):
            start = end
            continue
        accepted_path = output_directory / "accepted_batches" / f"{request.batch_id}.json"
        if accepted_path.exists():
            transaction = AcceptedDraftBatch.model_validate_json(accepted_path.read_bytes())
            if (
                transaction.request.configuration != configuration
                or transaction.request.assignments
                != (request.assignments[-len(transaction.request.assignments) :])
            ):
                raise ValueError("Accepted batch transaction differs from its planned assignments")
        else:
            transaction = generate_accepted_batch(
                request,
                rows,
                start,
                generate,
                output_directory / "acceptance_failures.jsonl",
            )
            write_record(accepted_path, transaction)
        validated = accept_draft_batch(
            request,
            GeneratedDraftBatch(
                utterances=tuple(
                    GeneratedDraftText(base_id=row.base_id, text=row.text)
                    for row in transaction.utterances
                )
            ),
            rows[:start],
        )
        if (
            validated != transaction.utterances
            or tuple(rows[start:]) != validated[: len(rows) - start]
        ):
            raise ValueError("Accepted batch transaction conflicts with existing journal rows")
        for row in validated[len(rows) - start :]:
            append_record(path, row)
            rows.append(row)
        start = end
    return tuple(rows)
