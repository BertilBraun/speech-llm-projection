"""Conservative literal-risk screening; missing overlap is never labelled an error."""

import re
from collections.abc import Sequence
from enum import Enum
from pathlib import Path

from scripts.inventory_results import write_record
from speech_projector.followup_evaluation import file_artifact
from speech_projector.followup_quality import QualityResponseSource
from speech_projector.models import FileArtifact, Record, SampleGeneration
from speech_projector.overnight_data import Cohort
from speech_projector.overnight_launcher import FinalEvaluationSelection, load_sources
from speech_projector.overnight_preparation import load_records


class LiteralKind(str, Enum):
    NUMBER = "number"
    CLOCK = "clock"
    WEEKDAY = "weekday"


class ContentLiteral(Record):
    kind: LiteralKind
    normalized: str
    quoted: str


class ContentReviewConfig(Record):
    selection: Path
    sources: Path
    responses: tuple[QualityResponseSource, ...]
    output_directory: Path


class ContentReviewRecord(Record):
    source_name: str
    example_id: str
    cohort: Cohort
    user_text: str
    response: str
    source_literals: tuple[ContentLiteral, ...]
    response_literals: tuple[ContentLiteral, ...]
    novel_response_literals: tuple[ContentLiteral, ...]
    source_negation_markers: tuple[str, ...]
    response_negation_markers: tuple[str, ...]
    first_person_action_overlap: tuple[str, ...]


class ContentReviewCounts(Record):
    source_name: str
    examples: int
    novel_number_time_or_weekday_cases: int
    source_negation_without_response_marker_cases: int
    first_person_action_overlap_cases: int


class ContentReviewReport(Record):
    configuration: ContentReviewConfig
    artifacts: tuple[FileArtifact, ...]
    counts: tuple[ContentReviewCounts, ...]


HOURS = (
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
)
DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
CLOCK = re.compile(
    r"\b(\d{1,2}|" + "|".join(HOURS) + r")(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b", re.IGNORECASE
)
NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
NEGATION = re.compile(
    r"\b(?:not|no|never|without|missing|incorrect|wrong|don['’]t|can['’]t|won['’]t|isn['’]t|didn['’]t)\b",
    re.IGNORECASE,
)
ACTION = re.compile(
    r"\b(?:I['’]ll|I will|I am|I['’]m)\s+"
    r"(send|share|submit|bring|drop|finish|check|update|review|deliver|align|include|make)\b",
    re.IGNORECASE,
)


def extract_literals(text: str) -> tuple[ContentLiteral, ...]:
    result: list[ContentLiteral] = []
    spans: list[tuple[int, int]] = []
    for found in CLOCK.finditer(text):
        hour_word = found.group(1).lower()
        hour = int(hour_word) if hour_word.isdigit() else HOURS.index(hour_word) + 1
        if not 1 <= hour <= 12:
            continue
        minutes = int(found.group(2)) if found.group(2) is not None else 0
        if minutes >= 60:
            continue
        hour = hour % 12 + (12 if found.group(3).lower().startswith("p") else 0)
        result.append(
            ContentLiteral(
                kind=LiteralKind.CLOCK, normalized=f"{hour:02}:{minutes:02}", quoted=found.group()
            )
        )
        spans.append(found.span())
    for found in NUMBER.finditer(text):
        if any(start <= found.start() < end for start, end in spans):
            continue
        result.append(
            ContentLiteral(
                kind=LiteralKind.NUMBER, normalized=str(float(found.group())), quoted=found.group()
            )
        )
    for day in DAYS:
        for found in re.finditer(r"\b" + day + r"\b", text, re.IGNORECASE):
            result.append(
                ContentLiteral(kind=LiteralKind.WEEKDAY, normalized=day, quoted=found.group())
            )
    return tuple(result)


def review_content(
    source_name: str, sample: SampleGeneration, user_text: str, history: str, cohort: Cohort
) -> ContentReviewRecord:
    source = history + "\n" + user_text
    literals = extract_literals(source)
    response_literals = extract_literals(sample.generated_response)
    known = {(row.kind, row.normalized) for row in literals}
    source_actions = {found.group(1).lower() for found in ACTION.finditer(user_text)}
    response_actions = {
        found.group(1).lower() for found in ACTION.finditer(sample.generated_response)
    }
    return ContentReviewRecord(
        source_name=source_name,
        example_id=sample.example_id,
        cohort=cohort,
        user_text=user_text,
        response=sample.generated_response,
        source_literals=literals,
        response_literals=response_literals,
        novel_response_literals=tuple(
            row for row in response_literals if (row.kind, row.normalized) not in known
        ),
        source_negation_markers=tuple(found.group() for found in NEGATION.finditer(user_text)),
        response_negation_markers=tuple(
            found.group() for found in NEGATION.finditer(sample.generated_response)
        ),
        first_person_action_overlap=tuple(sorted(source_actions & response_actions)),
    )


def count_review(source_name: str, records: Sequence[ContentReviewRecord]) -> ContentReviewCounts:
    return ContentReviewCounts(
        source_name=source_name,
        examples=len(records),
        novel_number_time_or_weekday_cases=sum(
            bool(row.novel_response_literals) for row in records
        ),
        source_negation_without_response_marker_cases=sum(
            bool(row.source_negation_markers) and not row.response_negation_markers
            for row in records
        ),
        first_person_action_overlap_cases=sum(
            bool(row.first_person_action_overlap) for row in records
        ),
    )


def run_content_review(configuration: ContentReviewConfig) -> ContentReviewReport:
    selection = FinalEvaluationSelection.model_validate_json(configuration.selection.read_bytes())
    examples = {row.example_id: row for row in selection.examples}
    sources = {row.example_id: row for row in load_sources(configuration.sources)}
    observations: list[ContentReviewRecord] = []
    counts: list[ContentReviewCounts] = []
    for source in configuration.responses:
        samples = tuple(
            row
            for row in load_records(source.generations, SampleGeneration)
            if row.condition == source.condition
        )
        if len({row.example_id for row in samples}) != len(samples):
            raise ValueError("Content review requires unique primary generation identities")
        rows = tuple(
            review_content(
                source.name,
                sample,
                examples[sample.example_id].user_text,
                "\n".join(turn.text for turn in examples[sample.example_id].history),
                sources[sample.example_id].cohort,
            )
            for sample in samples
        )
        observations.extend(rows)
        counts.append(count_review(source.name, rows))
    report = ContentReviewReport(
        configuration=configuration,
        artifacts=tuple(
            file_artifact(path)
            for path in (
                configuration.selection,
                configuration.sources,
                *(source.generations for source in configuration.responses),
            )
        ),
        counts=tuple(counts),
    )
    directory = configuration.output_directory
    write_record(directory / "summary.json", report)
    (directory / "observations.jsonl").write_text(
        "".join(row.model_dump_json() + "\n" for row in observations), encoding="utf-8"
    )
    lines = [
        "# Literal grounding review aids",
        "",
        "These are auditable surface-risk flags, not factual accuracy. New numbers can be valid "
        "suggestions, omitted negation markers can be faithful paraphrases, and an assistant can "
        "legitimately offer a user's action. Absence of repeated user words is never an error. "
        "Read full replies and separately score literal grounding; "
        "teacher targets are not factual gold.",
        "",
    ]
    for row in observations:
        if (
            row.novel_response_literals
            or (row.source_negation_markers and not row.response_negation_markers)
            or row.first_person_action_overlap
        ):
            lines.extend(
                [
                    f"## {row.source_name} / {row.example_id}",
                    "",
                    f"USER: {row.user_text}",
                    "",
                    f"REPLY: {row.response}",
                    "",
                    "New literal values: "
                    f"{tuple(item.quoted for item in row.novel_response_literals)}; "
                    f"user negations: {row.source_negation_markers}; "
                    f"reply negations: {row.response_negation_markers}; "
                    f"shared first-person actions: {row.first_person_action_overlap}",
                    "",
                ]
            )
    (directory / "review.md").write_text("\n".join(lines), encoding="utf-8")
    return report
