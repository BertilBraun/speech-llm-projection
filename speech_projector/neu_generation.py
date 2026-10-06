"""Frozen Qwen drafting and concise teacher replies for fresh Neu emotional pairs."""

from collections.abc import Sequence
from pathlib import Path

from pydantic import Field, model_validator

from scripts.inventory_results import write_record
from scripts.neutts_pilot_state import digest
from speech_projector.emotional_dataset import GeneratedDraftBatch
from speech_projector.emotional_generation import (
    CachedGeneration,
    EmotionalGenerationConfig,
    TextGenerationBatch,
    TextGenerationRequest,
    completed_outcomes,
)
from speech_projector.generation import CompletedGeneration, TokenLimitedGeneration
from speech_projector.journal import append_record, read_journal
from speech_projector.models import Record
from speech_projector.neu_dataset import (
    NeuDatasetProvenance,
    NeuDraftRequest,
    NeuUtterance,
    build_neu_requests,
    neu_cases,
    neu_draft_prompt,
    run_neu_construction,
    write_neu_cases,
)
from speech_projector.preview_responses import ChatMessage
from speech_projector.tts_pilot import TtsPilotCase


class NeuGenerationConfig(Record):
    generation: EmotionalGenerationConfig
    previous_utterances: Path
    previous_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_source_output(self) -> "NeuGenerationConfig":
        if self.previous_utterances.parent == self.generation.output_directory:
            raise ValueError("New Neu outputs must remain separate from the older corpus")
        return self


class NeuTeacherTarget(Record):
    case: TtsPilotCase
    response: CompletedGeneration
    capped_attempts: tuple[TokenLimitedGeneration, ...]


class NeuTeacherSource(Record):
    utterances_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class NeuGenerationSummary(Record):
    configuration: NeuGenerationConfig
    utterances: int
    completed_targets: int
    identical_target_pairs: int
    generation_batches: int
    generation_seconds: float
    generated_tokens: int


def neu_text_request(
    request: NeuDraftRequest, config: NeuGenerationConfig
) -> TextGenerationRequest:
    return TextGenerationRequest(
        request_id=f"draft:{request.batch_id}",
        messages=(
            ChatMessage(role="system", content=config.generation.draft_system),
            ChatMessage(role="user", content=neu_draft_prompt(request)),
        ),
    )


def neu_teacher_request(case: TtsPilotCase, config: NeuGenerationConfig) -> TextGenerationRequest:
    metadata = (
        f"The USER delivered this utterance with a {case.emotion.value} tone.\n"
        "This is metadata about the user, not an instruction to imitate their tone."
    )
    return TextGenerationRequest(
        request_id=f"teacher:{case.case_id}",
        messages=(
            ChatMessage(
                role="system", content=config.generation.teacher_system + "\n\n" + metadata
            ),
            ChatMessage(role="user", content=case.text),
        ),
    )


def generate_neu_drafts(
    config: NeuGenerationConfig, cached: CachedGeneration
) -> tuple[NeuUtterance, ...]:
    requests = build_neu_requests(config.generation.dataset)
    saved = read_journal(config.generation.output_directory / "utterances.jsonl", NeuUtterance)
    completed = {row.assignment.base_id for row in saved}
    pending = tuple(
        request
        for request in requests
        if any(assignment.base_id not in completed for assignment in request.assignments)
    )
    offsets = {request.batch_id: index for index, request in enumerate(pending)}

    def generate(request: NeuDraftRequest) -> GeneratedDraftBatch:
        text_request = neu_text_request(request, config)
        if request.batch_id in offsets and not cached.is_cached(
            text_request, config.generation.draft_token_cap
        ):
            offset = offsets[request.batch_id]
            batch = tuple(
                neu_text_request(row, config)
                for row in pending[offset : offset + config.generation.inference_batch_size]
            )
            completed_outcomes(
                batch,
                cached,
                config.generation.draft_token_cap,
                config.generation.draft_retry_token_cap,
            )
        response, _ = completed_outcomes(
            (text_request,),
            cached,
            config.generation.draft_token_cap,
            config.generation.draft_retry_token_cap,
        )[0]
        return GeneratedDraftBatch.model_validate_json(response.text)

    provenance = NeuDatasetProvenance(
        configuration=config.generation.dataset,
        previous_utterances=config.previous_utterances,
        previous_sha256=config.previous_sha256,
    )
    utterances = run_neu_construction(config.generation.output_directory, provenance, generate)
    write_neu_cases(config.generation.output_directory, utterances, config.generation.seed)
    return utterances


def generate_neu_targets(
    config: NeuGenerationConfig, cached: CachedGeneration, limit: int | None = None
) -> tuple[NeuTeacherTarget, ...]:
    directory = config.generation.output_directory
    utterances = read_journal(directory / "utterances.jsonl", NeuUtterance)
    if len(utterances) != config.generation.dataset.utterance_count:
        raise ValueError("Teacher targets require the complete configured fresh utterance manifest")
    source = NeuTeacherSource(
        utterances_sha256=digest(directory / "utterances.jsonl"),
        cases_sha256=digest(directory / "cases.json"),
    )
    source_path = directory / "teacher_source.json"
    if source_path.exists():
        if NeuTeacherSource.model_validate_json(source_path.read_bytes()) != source:
            raise ValueError("Teacher source manifest changed after target generation began")
    else:
        write_record(source_path, source)
    cases = neu_cases(utterances, config.generation.seed)
    if limit is not None:
        if not 1 <= limit <= len(cases):
            raise ValueError("Teacher limit must be within the full case count")
        selected_ids = {case.case_id for case in cases[:limit]}
    else:
        selected_ids = {case.case_id for case in cases}
    previous = read_journal(directory / "teacher_targets.jsonl", NeuTeacherTarget)
    expected = {case.case_id: case for case in cases}
    completed: set[str] = set()
    for row in previous:
        if row.case.case_id in completed or expected.get(row.case.case_id) != row.case:
            raise ValueError("Saved teacher target differs from its exact case identity")
        completed.add(row.case.case_id)
    pending = tuple(
        case for case in cases if case.case_id not in completed and case.case_id in selected_ids
    )
    targets = list(previous)
    for offset in range(0, len(pending), config.generation.inference_batch_size):
        batch = pending[offset : offset + config.generation.inference_batch_size]
        outcomes = completed_outcomes(
            tuple(neu_teacher_request(case, config) for case in batch),
            cached,
            config.generation.teacher_token_cap,
            config.generation.teacher_retry_token_cap,
        )
        for case, (response, capped) in zip(batch, outcomes, strict=True):
            if not response.text:
                raise ValueError("Teacher produced an empty completed reply; trace preserved")
            target = NeuTeacherTarget(case=case, response=response, capped_attempts=capped)
            append_record(directory / "teacher_targets.jsonl", target)
            targets.append(target)
        print(f"Completed fresh Neu teacher targets: {len(targets)}/{len(cases)}", flush=True)
        if offset == 0:
            render_neu_pilot(directory)
    return tuple(targets)


def target_agreement(targets: Sequence[NeuTeacherTarget]) -> int:
    by_base: dict[str, list[str]] = {}
    for target in targets:
        by_base.setdefault(target.case.utterance_id, []).append(target.response.text.strip())
    return sum(len(replies) == 2 and replies[0] == replies[1] for replies in by_base.values())


def summarize_neu_generation(config: NeuGenerationConfig) -> NeuGenerationSummary:
    directory = config.generation.output_directory
    targets = read_journal(directory / "teacher_targets.jsonl", NeuTeacherTarget)
    batches = read_journal(
        directory / config.generation.trace_subdirectory / "generation_batches.jsonl",
        TextGenerationBatch,
    )
    summary = NeuGenerationSummary(
        configuration=config,
        utterances=len(read_journal(directory / "utterances.jsonl", NeuUtterance)),
        completed_targets=len(targets),
        identical_target_pairs=target_agreement(targets),
        generation_batches=len(batches),
        generation_seconds=sum(batch.runtime_seconds for batch in batches),
        generated_tokens=sum(
            len(outcome.generation.token_ids) for batch in batches for outcome in batch.outcomes
        ),
    )
    write_record(
        directory / config.generation.trace_subdirectory / "neu_generation_summary.json", summary
    )
    return summary


def render_neu_pilot(directory: Path, limit: int = 20) -> Path:
    utterances = read_journal(directory / "utterances.jsonl", NeuUtterance)[:limit]
    targets = read_journal(directory / "teacher_targets.jsonl", NeuTeacherTarget)
    by_case = {target.case.case_id: target for target in targets}
    lines = [
        "# Fresh Neu dataset contrast pilot",
        "",
        "Intended delivery labels and Qwen's own concise replies are supervision, "
        "not verified emotional or factual gold. Exact target agreement is diagnostic only; "
        "no semantic rejection is applied.",
        "",
    ]
    for row in utterances:
        lines.extend(
            (f"## {row.assignment.base_id}: {row.assignment.domain.value}", "", row.text, "")
        )
        for emotion in row.assignment.emotions:
            target = by_case.get(f"{row.assignment.base_id}_{emotion.value}")
            lines.extend(
                (
                    f"**{emotion.value}:** {target.response.text if target else 'Target pending.'}",
                    "",
                )
            )
    lines.extend(
        (
            f"Exact identical complete target pairs in saved journal: {target_agreement(targets)}.",
            "",
        )
    )
    path = directory / "contrast_pilot.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
