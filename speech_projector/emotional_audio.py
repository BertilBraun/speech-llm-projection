"""Resumable emotional audio generation records and clip verification."""

from enum import Enum
from pathlib import Path

from pydantic import Field, model_validator

from scripts.inventory_results import stable_digest, write_record
from speech_projector.emotion_preview import PreviewClip, PreviewPlan
from speech_projector.models import Record


class CodecFinish(str, Enum):
    STOP = "stop"
    LENGTH = "length"


class CodecTermination(Record):
    case_id: str
    finish_reason: CodecFinish
    codec_tokens: int = Field(gt=0)
    max_new_tokens: int = Field(ge=4)
    actual_seed: int = Field(ge=0)
    runtime_seconds: float = Field(ge=0)

    @property
    def accepted(self) -> bool:
        return (
            self.finish_reason == CodecFinish.STOP and 0 < self.codec_tokens < self.max_new_tokens
        )


class EmotionalAudioConfig(Record):
    output: Path
    plan: PreviewPlan
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    batch_size: int = Field(default=4, ge=1)
    retry_max_new_tokens: int = Field(default=1024, ge=4)

    @model_validator(mode="after")
    def validate_retry_cap(self) -> "EmotionalAudioConfig":
        if self.retry_max_new_tokens <= self.plan.max_new_tokens:
            raise ValueError("Retry cap must exceed the initial synthesis cap")
        return self


class AudioBatchTiming(Record):
    """Ordered SDK batch seed; individual case seeds apply only to fallback/retry calls."""

    case_ids: tuple[str, ...]
    batch_seed: int
    used_individual_fallback: bool = False
    runtime_seconds: float = Field(ge=0)
    peak_vram_gb: float = Field(ge=0)
    completed: int = Field(ge=0)


class AudioGenerationSummary(Record):
    planned: int = Field(ge=1)
    completed: int = Field(ge=0)
    failed_case_ids: tuple[str, ...]
    session_runtime_seconds: float = Field(ge=0)
    session_audio_seconds: float = Field(ge=0)
    session_completed: int = Field(ge=0)
    peak_vram_gb: float = Field(ge=0)


class UncommittedAudioRecovery(Record):
    source_path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)
    destination: Path


def archive_uncommitted_audio(
    config: EmotionalAudioConfig,
) -> tuple[UncommittedAudioRecovery, ...]:
    recoveries: list[UncommittedAudioRecovery] = []
    for case in config.plan.cases:
        record = config.output / "clips" / f"{case.case_id}.json"
        source = config.output / "audio" / f"{case.case_id}.wav"
        if record.exists() or not source.exists():
            continue
        size, digest = stable_digest(source)
        destination = config.output / "orphaned_audio" / f"{case.case_id}_{digest}.wav"
        recovery = UncommittedAudioRecovery(
            source_path=source.resolve(),
            sha256=digest,
            bytes=size,
            destination=destination.resolve(),
        )
        recovery_path = destination.with_suffix(".json")
        if recovery_path.exists():
            if UncommittedAudioRecovery.model_validate_json(recovery_path.read_bytes()) != recovery:
                raise ValueError(f"Uncommitted audio recovery metadata changed: {recovery_path}")
        else:
            write_record(recovery_path, recovery)
        if destination.exists():
            if stable_digest(destination) != (size, digest):
                raise ValueError(f"Uncommitted audio archive bytes changed: {destination}")
            source.unlink()
        else:
            source.replace(destination)
        recoveries.append(recovery)
    return tuple(recoveries)


def completed_audio(config: EmotionalAudioConfig) -> tuple[PreviewClip, ...]:
    clips: list[PreviewClip] = []
    for case in config.plan.cases:
        record = config.output / "clips" / f"{case.case_id}.json"
        if not record.exists():
            continue
        clip = PreviewClip.model_validate_json(record.read_bytes())
        if clip.case != case:
            raise ValueError(f"Completed audio case changed: {case.case_id}")
        audio = config.output / clip.audio.path
        size, digest = stable_digest(audio)
        if size != clip.audio.bytes or digest != clip.audio.sha256:
            raise ValueError(f"Completed audio bytes changed: {case.case_id}")
        clips.append(clip)
    return tuple(clips)
