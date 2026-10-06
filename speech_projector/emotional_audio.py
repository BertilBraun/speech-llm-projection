"""Resumable emotional audio generation records and clip verification."""

from pathlib import Path

from pydantic import Field, model_validator

from scripts.inventory_results import stable_digest
from speech_projector.emotion_preview import PreviewClip, PreviewPlan
from speech_projector.models import Record


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
    case_ids: tuple[str, ...]
    seed: int
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
