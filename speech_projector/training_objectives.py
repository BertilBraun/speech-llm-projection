"""Explicit response, transcription and ordinary-only distillation supervision."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from torch import Tensor

from speech_projector.inputs import SpeechInput
from speech_projector.journal import read_journal
from speech_projector.llm import FrozenQwen
from speech_projector.models import (
    Example,
    OrdinaryResponseKLObjective,
    Record,
    ResponseCrossEntropyObjective,
    RunConfig,
    TranscriptMixtureObjective,
)
from speech_projector.overnight_data import (
    NeuEmotionalExampleSource,
    OrdinaryExampleSource,
    QwenEmotionalExampleSource,
    SourceSidecar,
)


class SupervisionTask(str, Enum):
    RESPONSE_CE = "response_ce"
    TRANSCRIPT_CE = "transcript_ce"
    RESPONSE_KL = "response_kl"


@dataclass(frozen=True)
class ObjectiveLoss:
    loss: Tensor
    task: SupervisionTask
    target_tokens: int


@dataclass(frozen=True)
class ObjectiveSmokeExample:
    example: Example
    source: SourceSidecar


class ObjectiveExampleLog(Record):
    example_id: str
    task: SupervisionTask
    target_tokens: int
    loss: float


class ObjectiveStepLog(Record):
    step: int
    epoch: int
    examples: tuple[ObjectiveExampleLog, ...]


def recover_objective_journal(path: Path, checkpoint_step: int) -> None:
    records = read_journal(path, ObjectiveStepLog)
    if any(first.step >= second.step for first, second in zip(records, records[1:], strict=False)):
        raise ValueError("Objective journal steps must be strictly increasing")
    retained = tuple(record for record in records if record.step <= checkpoint_step)
    if retained == records:
        return
    original = path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    archive = path.with_name(f"{path.name}.beyond-checkpoint-{digest}.jsonl")
    if not archive.exists():
        archive.write_bytes(original)
    pending = path.with_name(f"{path.name}.pending")
    pending.write_bytes("".join(record.model_dump_json() + "\n" for record in retained).encode())
    pending.replace(path)


def align_sources(
    configuration: RunConfig, examples: Sequence[Example], sources: Sequence[SourceSidecar]
) -> tuple[SourceSidecar, ...]:
    match configuration.objective:
        case ResponseCrossEntropyObjective():
            return ()
        case TranscriptMixtureObjective() | OrdinaryResponseKLObjective():
            if configuration.microbatch_size != 1:
                raise ValueError("Lexical objectives require verified single-example microbatches")
            indexed = {source.example_id: source for source in sources}
            if len(indexed) != len(sources) or set(indexed) != {
                example.example_id for example in examples
            }:
                raise ValueError("Lexical objectives require exact, unique source-sidecar coverage")
            return tuple(indexed[example.example_id] for example in examples)


def uses_transcription(configuration: RunConfig, epoch: int, example: Example) -> bool:
    match configuration.objective:
        case TranscriptMixtureObjective(transcript_probability=probability):
            digest = hashlib.sha256(
                f"{configuration.seed}:{epoch}:{example.example_id}".encode()
            ).digest()
            return int.from_bytes(digest[:8], "little") / 2**64 < probability
        case ResponseCrossEntropyObjective() | OrdinaryResponseKLObjective():
            return False


def select_objective_smoke(
    configuration: RunConfig,
    examples: Sequence[Example],
    sources: Sequence[SourceSidecar],
    epoch: int,
) -> ObjectiveSmokeExample:
    aligned = align_sources(configuration, examples, sources)
    match configuration.objective:
        case TranscriptMixtureObjective():
            for example, source in zip(examples, aligned, strict=True):
                if uses_transcription(configuration, epoch, example):
                    return ObjectiveSmokeExample(example, source)
            raise ValueError(
                "Transcript mixture has no auxiliary-task example for its gradient gate"
            )
        case OrdinaryResponseKLObjective():
            for example, source in zip(examples, aligned, strict=True):
                match source:
                    case OrdinaryExampleSource():
                        return ObjectiveSmokeExample(example, source)
                    case QwenEmotionalExampleSource() | NeuEmotionalExampleSource():
                        pass
            raise ValueError("Response KL has no ordinary example for its gradient gate")
        case ResponseCrossEntropyObjective():
            raise ValueError("Response CE uses the original gradient gate")


def objective_loss(
    wrapper: FrozenQwen,
    example: Example,
    speech: SpeechInput,
    source: SourceSidecar,
    epoch: int,
) -> ObjectiveLoss:
    match wrapper.config.objective:
        case TranscriptMixtureObjective(transcription_prompt=prompt):
            if uses_transcription(wrapper.config, epoch, example):
                transcription = example.model_copy(
                    update={"history": (), "target_text": example.user_text, "prompt": prompt}
                )
                if (
                    len(wrapper.tokenizer.encode(example.user_text, add_special_tokens=False)) + 1
                    > wrapper.config.max_target_tokens
                ):
                    raise ValueError("Transcription target exceeds the configured token budget")
                return ObjectiveLoss(
                    loss=wrapper.loss(transcription, speech),
                    task=SupervisionTask.TRANSCRIPT_CE,
                    target_tokens=wrapper.target_token_count(transcription),
                )
        case OrdinaryResponseKLObjective():
            match source:
                case OrdinaryExampleSource():
                    return ObjectiveLoss(
                        loss=wrapper.response_kl(example, speech),
                        task=SupervisionTask.RESPONSE_KL,
                        target_tokens=wrapper.target_token_count(example),
                    )
                case QwenEmotionalExampleSource() | NeuEmotionalExampleSource():
                    pass
        case ResponseCrossEntropyObjective():
            pass
    return ObjectiveLoss(
        loss=wrapper.loss(example, speech),
        task=SupervisionTask.RESPONSE_CE,
        target_tokens=wrapper.target_token_count(example),
    )
