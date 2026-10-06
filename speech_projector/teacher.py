"""Resumable transcript-teacher target generation and its manifest boundary."""

import subprocess
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

import torch
from huggingface_hub import snapshot_download
from pydantic import Field

from scripts.package_results import FileArtifact, ModelRevision, file_digest
from speech_projector.generation import (
    INITIAL_TEACHER_TOKEN_CAP,
    RETRY_TEACHER_TOKEN_CAP,
    CompletedGeneration,
    GenerationResult,
    TokenLimitedGeneration,
)
from speech_projector.inputs import TranscriptInput
from speech_projector.journal import append_record, read_journal
from speech_projector.llm import FrozenQwen
from speech_projector.models import Example, Record, RunConfig, Split
from speech_projector.training import weights_digest


class TeacherConfig(Record):
    run: RunConfig
    manifest: Path
    output_directory: Path
    batch_size: int = Field(default=8, gt=0)
    initial_max_new_tokens: int = Field(default=INITIAL_TEACHER_TOKEN_CAP, gt=0)
    retry_max_new_tokens: int = Field(default=RETRY_TEACHER_TOKEN_CAP, gt=0)


class TeacherProvenance(Record):
    config: TeacherConfig
    source_commit: str
    input_manifest: FileArtifact
    model_revision: ModelRevision
    frozen_parameter_sha256: str
    started_at: datetime


class TeacherTarget(Record):
    example: Example
    response: CompletedGeneration
    capped_attempts: tuple[TokenLimitedGeneration, ...]


class TeacherFailureReason(str, Enum):
    TOKEN_LIMIT = "token_limit"
    EMPTY_RESPONSE = "empty_response"


class TeacherFailure(Record):
    example: Example
    reason: TeacherFailureReason
    attempts: tuple[GenerationResult, ...]


class TeacherProgress(Record):
    completed_examples: int
    generated_tokens: int
    elapsed_seconds: float
    examples_per_second: float
    tokens_per_second: float
    peak_vram_gb: float


class SelectionKind(str, Enum):
    FULL = "full"
    BOOTSTRAP = "bootstrap"


class FullTeacherSelection(Record):
    kind: Literal[SelectionKind.FULL] = SelectionKind.FULL
    output_manifest: Path


class BootstrapTeacherSelection(Record):
    kind: Literal[SelectionKind.BOOTSTRAP] = SelectionKind.BOOTSTRAP
    output_manifest: Path
    training_examples: int = Field(default=256, gt=0)
    validation_examples: int = Field(default=32, gt=0)
    test_examples: int = Field(default=32, gt=0)


TeacherSelection: TypeAlias = Annotated[
    FullTeacherSelection | BootstrapTeacherSelection, Field(discriminator="kind")
]


class TeacherExport(Record):
    selection: TeacherSelection
    manifest: FileArtifact
    examples: int


def select_examples(
    examples: tuple[Example, ...], selection: TeacherSelection
) -> tuple[Example, ...]:
    match selection:
        case FullTeacherSelection():
            return examples
        case BootstrapTeacherSelection():
            selected: list[Example] = []
            for split, count in (
                (Split.TRAIN, selection.training_examples),
                (Split.VALIDATION, selection.validation_examples),
                (Split.TEST, selection.test_examples),
            ):
                available = tuple(example for example in examples if example.split == split)[:count]
                if len(available) != count:
                    raise ValueError(f"Teacher bootstrap lacks required {split.value} examples")
                selected.extend(available)
            return tuple(selected)


def measured_progress(
    targets: Sequence[TeacherTarget], elapsed: float, peak_vram_gb: float
) -> TeacherProgress:
    generated = sum(len(target.response.token_ids) for target in targets)
    return TeacherProgress(
        completed_examples=len(targets),
        generated_tokens=generated,
        elapsed_seconds=elapsed,
        examples_per_second=len(targets) / elapsed if elapsed else 0.0,
        tokens_per_second=generated / elapsed if elapsed else 0.0,
        peak_vram_gb=peak_vram_gb,
    )


def target_example(target: TeacherTarget, wrapper: FrozenQwen) -> Example:
    if not target.response.text:
        raise ValueError("Completed teacher response is empty")
    token_count = len(wrapper.tokenizer.encode(target.response.text, add_special_tokens=False)) + 1
    if token_count > wrapper.config.max_target_tokens:
        raise ValueError("Completed teacher response exceeds the student target-token budget")
    return target.example.model_copy(update={"target_text": target.response.text})


def generate_targets(
    wrapper: FrozenQwen, examples: tuple[Example, ...], config: TeacherConfig
) -> tuple[TeacherTarget, ...]:
    initial = wrapper.generate_batch(
        examples,
        tuple(TranscriptInput(example.user_text) for example in examples),
        config.initial_max_new_tokens,
    )
    targets: list[TeacherTarget] = []
    for example, response in zip(examples, initial, strict=True):
        match response:
            case CompletedGeneration():
                completed = response
                capped: tuple[TokenLimitedGeneration, ...] = ()
            case TokenLimitedGeneration():
                retry = wrapper.generate_batch(
                    (example,), (TranscriptInput(example.user_text),), config.retry_max_new_tokens
                )[0]
                match retry:
                    case CompletedGeneration():
                        completed = retry
                        capped = (response,)
                    case TokenLimitedGeneration():
                        append_record(
                            config.output_directory / "failures.jsonl",
                            TeacherFailure(
                                example=example,
                                reason=TeacherFailureReason.TOKEN_LIMIT,
                                attempts=(response, retry),
                            ),
                        )
                        raise ValueError(
                            f"Teacher did not complete {example.example_id} after retry"
                        )
        if not completed.text:
            append_record(
                config.output_directory / "failures.jsonl",
                TeacherFailure(
                    example=example,
                    reason=TeacherFailureReason.EMPTY_RESPONSE,
                    attempts=(*capped, completed),
                ),
            )
            raise ValueError(f"Teacher returned an empty response for {example.example_id}")
        target = TeacherTarget(example=example, response=completed, capped_attempts=capped)
        target_example(target, wrapper)
        append_record(config.output_directory / "targets.jsonl", target)
        targets.append(target)
    return tuple(targets)


def teacher_batch_key(history_turns: int, example: Example) -> tuple[bool, int, str]:
    history = example.history[-history_turns:] if history_turns else ()
    return (
        not bool(history),
        len(example.user_text) + sum(len(turn.text) for turn in history),
        example.example_id,
    )


def run_teacher(config: TeacherConfig, selection: TeacherSelection) -> TeacherProgress:
    if config.retry_max_new_tokens <= config.initial_max_new_tokens:
        raise ValueError("Teacher retry cap must be larger than its initial cap")
    if config.run.max_target_tokens < config.retry_max_new_tokens + 1:
        raise ValueError("Student target budget must preserve every completed teacher reply")
    if subprocess.check_output(["git", "diff", "HEAD", "--"], text=True).strip():
        raise ValueError("Teacher launch requires a committed tracked source snapshot")
    config.output_directory.mkdir(parents=True, exist_ok=True)
    examples = tuple(
        Example.model_validate_json(line) for line in config.manifest.read_text().splitlines()
    )
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("Teacher source manifest must contain unique example identifiers")
    torch.manual_seed(config.run.seed)
    wrapper = FrozenQwen(config.run, torch.device("cuda"))
    snapshot = Path(snapshot_download(config.run.model_name, local_files_only=True))
    manifest_artifact = FileArtifact(
        path=config.manifest,
        source_path=config.manifest,
        bytes=config.manifest.stat().st_size,
        sha256=file_digest(config.manifest),
    )
    provenance_path = config.output_directory / "provenance.json"
    provenance = TeacherProvenance(
        config=config,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        input_manifest=manifest_artifact,
        model_revision=ModelRevision(
            model_name=config.run.model_name,
            snapshot_revisions=(snapshot.name,),
            main_revision=snapshot.name,
        ),
        frozen_parameter_sha256=weights_digest(wrapper.model),
        started_at=datetime.now(timezone.utc),
    )
    if provenance_path.exists():
        previous = TeacherProvenance.model_validate_json(provenance_path.read_bytes())
        if previous.model_copy(update={"started_at": provenance.started_at}) != provenance:
            raise ValueError("Teacher resume provenance differs from the original generation run")
    else:
        provenance_path.write_text(provenance.model_dump_json(indent=2), encoding="utf-8")
    targets = list(read_journal(config.output_directory / "targets.jsonl", TeacherTarget))
    expected = {example.example_id: example for example in examples}
    completed_ids: set[str] = set()
    for target in targets:
        if target.example != expected[target.example.example_id]:
            raise ValueError("Journal example differs from its immutable source manifest")
        if target.example.example_id in completed_ids:
            raise ValueError("Teacher journal contains a duplicate completed identifier")
        completed_ids.add(target.example.example_id)
    selected = select_examples(examples, selection)
    # Separating chat continuations from opening questions limits finished-row padding waste.
    pending = tuple(
        sorted(
            (example for example in selected if example.example_id not in completed_ids),
            key=partial(teacher_batch_key, config.run.history_turns),
        )
    )
    history = read_journal(config.output_directory / "progress.jsonl", TeacherProgress)
    previous_elapsed = history[-1].elapsed_seconds if history else 0.0
    previous_peak = history[-1].peak_vram_gb if history else 0.0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for offset in range(0, len(pending), config.batch_size):
        targets.extend(
            generate_targets(wrapper, pending[offset : offset + config.batch_size], config)
        )
        progress = measured_progress(
            targets,
            previous_elapsed + time.perf_counter() - started,
            max(previous_peak, torch.cuda.max_memory_allocated() / 1e9),
        )
        append_record(config.output_directory / "progress.jsonl", progress)
        print(progress.model_dump_json(), flush=True)
    by_identifier = {target.example.example_id: target for target in targets}
    output = selection.output_manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    pending_output = output.with_suffix(".pending.jsonl")
    with pending_output.open("w", encoding="utf-8") as stream:
        for example in selected:
            stream.write(
                target_example(by_identifier[example.example_id], wrapper).model_dump_json() + "\n"
            )
    pending_output.replace(output)
    progress = measured_progress(
        targets,
        previous_elapsed + time.perf_counter() - started,
        max(previous_peak, torch.cuda.max_memory_allocated() / 1e9),
    )
    (config.output_directory / "summary.json").write_text(
        progress.model_dump_json(indent=2), encoding="utf-8"
    )
    export = TeacherExport(
        selection=selection,
        manifest=FileArtifact(
            path=output, source_path=output, bytes=output.stat().st_size, sha256=file_digest(output)
        ),
        examples=len(selected),
    )
    output.with_suffix(".provenance.json").write_text(
        export.model_dump_json(indent=2), encoding="utf-8"
    )
    return progress
