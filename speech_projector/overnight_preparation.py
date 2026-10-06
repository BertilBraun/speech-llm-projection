"""Immutable assembly of ordinary and two separately generated emotional corpora."""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeVar

import soundfile
from pydantic import Field

from scripts.inventory_results import stable_digest
from scripts.package_results import FileArtifact
from speech_projector.data import Distribution, distribution
from speech_projector.emotion_preview import PreviewClip
from speech_projector.emotional_generation import EmotionalGenerationConfig, EmotionalTeacherTarget
from speech_projector.models import ChatPromptConfig, Example, Record, Split, SystemPromptConfig
from speech_projector.neu_dataset import NeuUtterance
from speech_projector.neu_generation import NeuGenerationConfig, NeuTeacherTarget
from speech_projector.neutts_batch_benchmark import BatchClipEvidence
from speech_projector.neutts_corpus import NeuCorpusResult
from speech_projector.overnight_data import (
    Cohort,
    NeuEmotionalExampleSource,
    OrdinaryExampleSource,
    QwenEmotionalExampleSource,
    SourceSidecar,
)
from speech_projector.overnight_evaluation import FixedValidationConfig, select_fixed_validation
from speech_projector.tts_pilot import PilotTermination

TRecord = TypeVar("TRecord", bound=Record)


class QwenSourceConfig(Record):
    targets: Path
    generation_configuration: Path
    audio_directory: Path
    expected_pairs: int = Field(default=5000, gt=0)


class NeuSourceConfig(QwenSourceConfig):
    utterances: Path


class OvernightPreparationConfig(Record):
    ordinary_manifest: Path
    qwen: QwenSourceConfig
    neu: NeuSourceConfig
    output_directory: Path
    ordinary_training_examples: int = Field(default=20000, gt=0)
    validation: FixedValidationConfig = FixedValidationConfig()


class ExclusionReason(str, Enum):
    OVER_DURATION = "over_30_seconds"


class PairExclusion(Record):
    cohort: Cohort
    base_id: str
    family_id: str
    split: Split
    source_case_ids: tuple[str, str]
    durations_seconds: tuple[float, float]
    reason: ExclusionReason


class CohortCoverage(Record):
    cohort: Cohort
    split: Split
    examples: int
    duration_seconds: Distribution


class CombinedPreparation(Record):
    configuration: OvernightPreparationConfig
    source_commit: str
    source_artifacts: tuple[FileArtifact, ...]
    manifest: FileArtifact
    sidecar: FileArtifact
    audio_inventory: FileArtifact
    fixed_validation: FileArtifact
    coverage: tuple[CohortCoverage, ...]
    exclusions: tuple[PairExclusion, ...]
    cached_features_reused: int
    missing_features: int
    generation_example_ids: tuple[str, ...]
    boundary_checks: tuple[str, ...]


@dataclass(frozen=True)
class PreparedRows:
    examples: tuple[Example, ...]
    sources: tuple[SourceSidecar, ...]
    audio: tuple[FileArtifact, ...]
    exclusions: tuple[PairExclusion, ...]


def load_records(path: Path, record_type: type[TRecord]) -> tuple[TRecord, ...]:
    contents = path.read_bytes()
    if contents and not contents.endswith(b"\n"):
        raise ValueError(f"Source journal is incomplete; refusing to mutate it: {path}")
    return tuple(record_type.model_validate_json(line) for line in contents.splitlines())


def artifact(path: Path) -> FileArtifact:
    size, digest = stable_digest(path)
    return FileArtifact(path=path, source_path=path.resolve(), bytes=size, sha256=digest)


def write_immutable(path: Path, contents: bytes) -> None:
    if path.exists():
        if path.read_bytes() != contents:
            raise ValueError(f"Refusing to overwrite a different derived dataset: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_bytes(contents)
    pending.replace(path)


def write_records(path: Path, records: Sequence[Record]) -> None:
    write_immutable(path, ("\n".join(row.model_dump_json() for row in records) + "\n").encode())


def verify_audio(path: Path, sha256: str, duration: float) -> FileArtifact:
    record = artifact(path)
    if record.sha256 != sha256:
        raise ValueError(f"Immutable source waveform hash mismatch: {path}")
    metadata = soundfile.info(path)
    if (
        metadata.channels != 1
        or abs(metadata.frames / metadata.samplerate - duration) > 1 / metadata.samplerate
    ):
        raise ValueError(f"Source audio duration or channel metadata mismatch: {path}")
    return record


def feature_path(directory: Path, cohort: Cohort, audio: FileArtifact) -> Path:
    return directory / "features" / cohort.value / f"{audio.sha256}.pt"


def qwen_rows(config: QwenSourceConfig, output: Path) -> PreparedRows:
    configuration = EmotionalGenerationConfig.model_validate_json(
        config.generation_configuration.read_bytes()
    )
    targets = load_records(config.targets, EmotionalTeacherTarget)
    groups: dict[str, list[EmotionalTeacherTarget]] = {}
    for target in targets:
        groups.setdefault(target.request.utterance.base_id, []).append(target)
    if len(groups) != config.expected_pairs or len(targets) != 2 * config.expected_pairs:
        raise ValueError("Older Qwen target journal does not cover the complete paired corpus")
    examples: list[Example] = []
    sources: list[SourceSidecar] = []
    audio: list[FileArtifact] = []
    exclusions: list[PairExclusion] = []
    for base_id, pair in groups.items():
        utterance = pair[0].request.utterance
        if len(pair) != 2 or any(row.request.utterance != utterance for row in pair):
            raise ValueError("Older emotional pair is missing or disagrees on utterance metadata")
        if {row.request.annotation for row in pair} != set(utterance.deliveries):
            raise ValueError("Older target deliveries differ from the original paired assignment")
        clips = tuple(
            PreviewClip.model_validate_json(
                (
                    config.audio_directory
                    / "clips"
                    / f"{base_id}_{row.request.annotation.delivery.value}.json"
                ).read_bytes()
            )
            for row in pair
        )
        verified = tuple(
            verify_audio(
                config.audio_directory / clip.audio.path, clip.audio.sha256, clip.duration_seconds
            )
            for clip in clips
        )
        audio.extend(verified)
        if any(clip.duration_seconds > 30 for clip in clips):
            exclusions.append(
                PairExclusion(
                    cohort=Cohort.QWEN_EMOTIONAL,
                    base_id=base_id,
                    family_id=utterance.family_id,
                    split=utterance.split,
                    source_case_ids=(clips[0].case.case_id, clips[1].case.case_id),
                    durations_seconds=(clips[0].duration_seconds, clips[1].duration_seconds),
                    reason=ExclusionReason.OVER_DURATION,
                )
            )
            continue
        for target, clip, waveform in zip(pair, clips, verified, strict=True):
            if (
                clip.case.text != utterance.text
                or clip.case.delivery != target.request.annotation.delivery
            ):
                raise ValueError("Older audio text or delivery differs from its teacher source")
            identifier = f"qwen:{clip.case.case_id}"
            examples.append(
                Example(
                    example_id=identifier,
                    dialogue_id=f"emotional:{utterance.family_id}",
                    split=utterance.split,
                    history=(),
                    user_text=utterance.text,
                    target_text=target.response.text,
                    audio_path=waveform.path,
                    duration=clip.duration_seconds,
                    domain=utterance.domain.value,
                    emotion=clip.case.delivery.value,
                    feature_path=feature_path(output, Cohort.QWEN_EMOTIONAL, waveform),
                    prompt=SystemPromptConfig(system_text=configuration.teacher_system),
                )
            )
            sources.append(
                QwenEmotionalExampleSource(
                    example_id=identifier,
                    source_manifest=config.targets,
                    source_example_id=clip.case.case_id,
                    base_id=base_id,
                    family_id=utterance.family_id,
                    emotion=clip.case.delivery,
                )
            )
    return PreparedRows(tuple(examples), tuple(sources), tuple(audio), tuple(exclusions))


def neu_rows(config: NeuSourceConfig, output: Path) -> PreparedRows:
    configuration = NeuGenerationConfig.model_validate_json(
        config.generation_configuration.read_bytes()
    )
    utterances = load_records(config.utterances, NeuUtterance)
    targets = load_records(config.targets, NeuTeacherTarget)
    corpus = NeuCorpusResult.model_validate_json(
        (config.audio_directory / "result.json").read_bytes()
    )
    if corpus.verified_clips != 2 * config.expected_pairs or corpus.missing_cases:
        raise ValueError("Neu audio corpus must finish and verify all paired clips before assembly")
    expected = {row.assignment.base_id: row for row in utterances}
    groups: dict[str, list[NeuTeacherTarget]] = {}
    for target in targets:
        groups.setdefault(target.case.utterance_id, []).append(target)
    if (
        len(expected) != config.expected_pairs
        or set(groups) != set(expected)
        or len(targets) != 2 * config.expected_pairs
    ):
        raise ValueError("Neu targets do not cover the complete fresh paired corpus")
    examples: list[Example] = []
    sources: list[SourceSidecar] = []
    audio: list[FileArtifact] = []
    exclusions: list[PairExclusion] = []
    for base_id, pair in groups.items():
        utterance = expected[base_id]
        if len(pair) != 2 or {row.case.emotion for row in pair} != set(
            utterance.assignment.emotions
        ):
            raise ValueError("Neu target pair differs from the original emotional assignment")
        evidence = tuple(
            BatchClipEvidence.model_validate_json(
                (config.audio_directory / "clips" / f"{row.case.case_id}.json").read_bytes()
            )
            for row in pair
        )
        if any(
            not item.generated_token_ids
            or item.generated_token_ids[-1] != corpus.speech_end_token_id
            for item in evidence
        ):
            raise ValueError("Neu source codec record lacks its actual speech-end token")
        clips = tuple(item.clip for item in evidence)
        verified = tuple(
            verify_audio(config.audio_directory / clip.audio_path, clip.sha256, clip.audio_seconds)
            for clip in clips
        )
        audio.extend(verified)
        if any(clip.termination != PilotTermination.STOP for clip in clips):
            raise ValueError("Incomplete Neu codec generation cannot enter the dataset")
        if any(clip.audio_seconds > 30 for clip in clips):
            exclusions.append(
                PairExclusion(
                    cohort=Cohort.NEU_EMOTIONAL,
                    base_id=base_id,
                    family_id=utterance.assignment.family_id,
                    split=utterance.assignment.split,
                    source_case_ids=(clips[0].case.case_id, clips[1].case.case_id),
                    durations_seconds=(clips[0].audio_seconds, clips[1].audio_seconds),
                    reason=ExclusionReason.OVER_DURATION,
                )
            )
            continue
        for target, clip, waveform in zip(pair, clips, verified, strict=True):
            if (
                clip.case.model_copy(update={"seed": target.case.seed}) != target.case
                or target.case.text != utterance.text
            ):
                raise ValueError("Neu waveform case differs from its exact teacher source")
            identifier = f"neu:{clip.case.case_id}"
            examples.append(
                Example(
                    example_id=identifier,
                    dialogue_id=f"emotional:{utterance.assignment.family_id}",
                    split=utterance.assignment.split,
                    history=(),
                    user_text=utterance.text,
                    target_text=target.response.text,
                    audio_path=waveform.path,
                    duration=clip.audio_seconds,
                    domain=utterance.assignment.domain.value,
                    emotion=clip.case.emotion.value,
                    feature_path=feature_path(output, Cohort.NEU_EMOTIONAL, waveform),
                    prompt=SystemPromptConfig(system_text=configuration.generation.teacher_system),
                )
            )
            sources.append(
                NeuEmotionalExampleSource(
                    example_id=identifier,
                    source_manifest=config.targets,
                    source_example_id=clip.case.case_id,
                    base_id=base_id,
                    family_id=utterance.assignment.family_id,
                    emotion=clip.case.emotion,
                )
            )
    return PreparedRows(tuple(examples), tuple(sources), tuple(audio), tuple(exclusions))


def validate_boundaries(examples: Sequence[Example], audio: Sequence[FileArtifact]) -> None:
    if len({row.example_id for row in examples}) != len(examples):
        raise ValueError("Combined example identifiers must be unique")
    dialogue_splits: dict[str, Split] = {}
    prompt_splits: dict[str, Split] = {}
    waveform_splits: dict[str, Split] = {}
    waveforms = {row.path: row.sha256 for row in audio}
    for example in examples:
        previous = dialogue_splits.setdefault(example.dialogue_id, example.split)
        if previous != example.split:
            raise ValueError(f"Conversation/scenario family crosses splits: {example.dialogue_id}")
        key = hashlib.sha256(
            (
                example.prompt.model_dump_json()
                + "\n"
                + "\n".join(f"{turn.role.value}:{turn.text}" for turn in example.history[-2:])
                + "\n"
                + re.sub(r"\s+", " ", example.user_text.strip().casefold())
            ).encode()
        ).hexdigest()
        if prompt_splits.setdefault(key, example.split) != example.split:
            raise ValueError(
                f"An exact normalized student prompt crosses splits: {example.example_id}"
            )
        digest = waveforms[example.audio_path]
        if waveform_splits.setdefault(digest, example.split) != example.split:
            raise ValueError(f"A waveform byte hash crosses splits: {example.example_id}")


def prepare_combined(config: OvernightPreparationConfig, source_commit: str) -> CombinedPreparation:
    existing_path = config.output_directory / "preparation.json"
    if existing_path.exists():
        existing = CombinedPreparation.model_validate_json(existing_path.read_bytes())
        if existing.configuration != config:
            raise ValueError("Cannot reuse a combined dataset under a different preparation config")
        for saved in existing.source_artifacts + (
            existing.manifest,
            existing.sidecar,
            existing.audio_inventory,
            existing.fixed_validation,
        ):
            if stable_digest(saved.path) != (saved.bytes, saved.sha256):
                raise ValueError(f"Immutable preparation artifact changed: {saved.path}")
        return existing
    ordinary = load_records(config.ordinary_manifest, Example)
    training = tuple(row for row in ordinary if row.split == Split.TRAIN)[
        : config.ordinary_training_examples
    ]
    if len(training) != config.ordinary_training_examples:
        raise ValueError("Ordinary source lacks the requested unchanged training prefix")
    ordinary = training + tuple(row for row in ordinary if row.split != Split.TRAIN)
    ordinary = tuple(
        row.model_copy(
            update={"example_id": f"ordinary:{row.example_id}", "prompt": ChatPromptConfig()}
        )
        for row in ordinary
    )
    ordinary_sources = tuple(
        OrdinaryExampleSource(
            example_id=row.example_id,
            source_manifest=config.ordinary_manifest,
            source_example_id=row.example_id.removeprefix("ordinary:"),
        )
        for row in ordinary
    )
    ordinary_audio = tuple(artifact(row.audio_path) for row in ordinary)
    qwen = qwen_rows(config.qwen, config.output_directory)
    neu = neu_rows(config.neu, config.output_directory)
    examples = ordinary + qwen.examples + neu.examples
    sources = ordinary_sources + qwen.sources + neu.sources
    audio = ordinary_audio + qwen.audio + neu.audio
    validate_boundaries(examples, audio)
    selected = select_fixed_validation(examples, sources, config.validation)
    root = config.output_directory
    write_records(root / "examples.jsonl", examples)
    write_records(root / "sources.jsonl", sources)
    write_records(root / "audio_inventory.jsonl", audio)
    write_records(root / "fixed_validation.jsonl", selected.examples)
    write_records(root / "fixed_validation_sources.jsonl", selected.sources)
    write_immutable(root / "validation_selection.json", selected.model_dump_json(indent=2).encode())
    source_map = {row.example_id: row for row in sources}
    coverage = tuple(
        CohortCoverage(
            cohort=cohort,
            split=split,
            examples=len(subset),
            duration_seconds=distribution([row.duration for row in subset]),
        )
        for cohort in Cohort
        for split in Split
        if (
            subset := [
                row
                for row in examples
                if row.split == split and source_map[row.example_id].cohort == cohort
            ]
        )
    )
    inputs = (
        config.ordinary_manifest,
        config.qwen.targets,
        config.qwen.generation_configuration,
        config.neu.targets,
        config.neu.utterances,
        config.neu.generation_configuration,
        config.neu.audio_directory / "result.json",
    )
    result = CombinedPreparation(
        configuration=config,
        source_commit=source_commit,
        source_artifacts=tuple(artifact(path) for path in inputs),
        manifest=artifact(root / "examples.jsonl"),
        sidecar=artifact(root / "sources.jsonl"),
        audio_inventory=artifact(root / "audio_inventory.jsonl"),
        fixed_validation=artifact(root / "fixed_validation.jsonl"),
        coverage=coverage,
        exclusions=qwen.exclusions + neu.exclusions,
        cached_features_reused=sum(row.feature_path.exists() for row in examples),
        missing_features=sum(not row.feature_path.exists() for row in examples),
        generation_example_ids=selected.generation_example_ids,
        boundary_checks=(
            "Conversation and scenario family split identity",
            "Exact normalized current prompt including generic student policy",
            "Complete emotional pairs",
            "Cross-split waveform byte hashes",
            "All source WAV hashes verified; no cropping or rewriting",
            "Current transcript and delivery labels omitted from speech input",
        ),
    )
    write_immutable(root / "preparation.json", result.model_dump_json(indent=2).encode())
    return result
