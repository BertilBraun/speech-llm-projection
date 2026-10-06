"""SDK-independent records for the bounded NeuTTS backbone batching benchmark."""

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scripts.neutts_pilot_state import NeuTtsPilotConfig, digest
from speech_projector.tts_pilot import (
    PilotTermination,
    TtsPilotCase,
    TtsPilotClip,
    TtsPilotManifest,
)


class NeuTtsBenchmarkConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pilot: NeuTtsPilotConfig
    source_commit: str
    repeats: int = Field(default=3, gt=0)
    batch_sizes: tuple[int, ...] = (1, 7)

    @model_validator(mode="after")
    def validate_batch_sizes(self) -> "NeuTtsBenchmarkConfig":
        if not self.batch_sizes or any(not 1 <= size <= 28 for size in self.batch_sizes):
            raise ValueError("Batch sizes must be in [1, 28]")
        if len(set(self.batch_sizes)) != len(self.batch_sizes):
            raise ValueError("Benchmark batch sizes must be unique")
        return self


class BatchClipEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    clip: TtsPilotClip
    prompt_tokens: int = Field(gt=0)
    speech_tokens: int = Field(gt=0)
    generated_token_ids: tuple[int, ...]


class BatchMeasurement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    requested_batch_size: int = Field(gt=0)
    repetition: int = Field(ge=0)
    batch_index: int = Field(ge=0)
    shared_batch_seed: int = Field(ge=0)
    padded_prompt_width: int = Field(gt=0)
    generation_token_cap: int = Field(gt=0)
    frontend_seconds: float = Field(ge=0)
    backbone_seconds: float = Field(gt=0)
    codec_watermark_seconds: float = Field(gt=0)
    end_to_end_seconds: float = Field(gt=0)
    clips: tuple[BatchClipEvidence, ...]

    @property
    def audio_seconds(self) -> float:
        return sum(item.clip.audio_seconds for item in self.clips)

    @property
    def aggregate_real_time_factor(self) -> float:
        return self.end_to_end_seconds / self.audio_seconds


class NeuTtsBenchmarkResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    configuration: NeuTtsBenchmarkConfig
    startup_seconds: float = Field(ge=0)
    reference_preparation_seconds: float = Field(ge=0)
    warmup_seconds: float = Field(gt=0)
    torch_version: str
    transformers_version: str
    neucodec_version: str
    helper_sha256: str
    reference_sha256: str
    watermark_active: bool
    effective_max_new_tokens: int = Field(gt=0)
    measurements: tuple[BatchMeasurement, ...]
    timing_scope: str


def completed_tokens(
    generated: tuple[int, ...], end_token: int
) -> tuple[tuple[int, ...], PilotTermination]:
    if end_token in generated:
        index = generated.index(end_token)
        return generated[: index + 1], PilotTermination.STOP
    return generated, PilotTermination.TOKEN_LIMIT


def prompt_emotions(
    cases: tuple[TtsPilotCase, ...], checker: Callable[[str], str | None]
) -> tuple[str | None, ...]:
    return tuple(checker(case.emotion.value) for case in cases)


def validate_benchmark_manifest(
    manifest: TtsPilotManifest, supported_emotions: tuple[str, ...]
) -> None:
    count = len(manifest.cases)
    if not 7 <= count <= 28 or count % 7:
        raise ValueError("Benchmark requires 7, 14, 21 or 28 cases")
    if len({case.text for case in manifest.cases}) != 1:
        raise ValueError("All benchmark emotions must share the same literal sentence")
    if {case.seed for case in manifest.cases} != {42}:
        raise ValueError("Benchmark cases must all use seed42")
    expected = set(supported_emotions)
    observed = {case.emotion.value for case in manifest.cases}
    if observed != expected:
        raise ValueError("Benchmark requires the exact seven NeuTTS emotions")
    families = {case.utterance_id for case in manifest.cases}
    if len(families) != count // 7:
        raise ValueError("Each seven-emotion replica requires a unique utterance ID")
    for family in families:
        cases = tuple(case for case in manifest.cases if case.utterance_id == family)
        if len(cases) != 7 or {case.emotion.value for case in cases} != expected:
            raise ValueError("Each replica must contain each NeuTTS emotion exactly once")


def verify_recorded_audio(result: NeuTtsBenchmarkResult, directory: Path) -> None:
    for measurement in result.measurements:
        for item in measurement.clips:
            if digest(directory / item.clip.audio_path) != item.clip.sha256:
                raise ValueError("Saved benchmark waveform hash differs from record")
