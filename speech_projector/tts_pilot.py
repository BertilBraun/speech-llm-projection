"""Canonical case and measurement records for the small TTS listening comparison."""

import math
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PilotEmotion(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    ANGRY = "angry"
    SAD = "sad"
    AFRAID = "afraid"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    MELANCHOLIC = "melancholic"
    SURPRISED = "surprised"


class PilotTermination(str, Enum):
    STOP = "stop"
    TOKEN_LIMIT = "token_limit"
    NOT_EXPOSED = "not_exposed"


class TtsPilotCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    utterance_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    emotion: PilotEmotion
    seed: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_nonblank_text(self) -> "TtsPilotCase":
        if not all(value.strip() for value in (self.case_id, self.utterance_id, self.text)):
            raise ValueError("Case identifiers and text must not be blank")
        return self


class TtsPilotManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: tuple[TtsPilotCase, ...]
    warmup_text: str
    warmup_seed: int

    @model_validator(mode="after")
    def validate_case_identity(self) -> "TtsPilotManifest":
        if not self.cases or not self.warmup_text.strip():
            raise ValueError("Cases and warmup text must be present")
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Case IDs must be unique")
        utterances: dict[str, str] = {}
        deliveries: set[tuple[str, PilotEmotion]] = set()
        for case in self.cases:
            if case.utterance_id in utterances and utterances[case.utterance_id] != case.text:
                raise ValueError("All deliveries of an utterance must have identical text")
            delivery = (case.utterance_id, case.emotion)
            if delivery in deliveries:
                raise ValueError("An utterance must not repeat an emotion")
            utterances[case.utterance_id] = case.text
            deliveries.add(delivery)
        return self


class TtsPilotClip(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    case: TtsPilotCase
    audio_path: str
    sha256: str
    sample_rate: int = Field(gt=0)
    audio_seconds: float = Field(gt=0)
    generation_seconds: float = Field(gt=0)
    real_time_factor: float = Field(gt=0)
    termination: PilotTermination

    @model_validator(mode="after")
    def validate_real_time_factor(self) -> "TtsPilotClip":
        if not math.isclose(self.real_time_factor, self.generation_seconds / self.audio_seconds):
            raise ValueError("Real-time factor must equal generation seconds / audio seconds")
        return self


class TtsPilotFailure(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case: TtsPilotCase
    error: str


class TtsPilotResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    model_repository: str
    model_revision: str
    source_commit: str
    backend_description: str
    speaker_description: str
    startup_seconds: float = Field(ge=0)
    reference_preparation_seconds: float = Field(ge=0)
    warmup_generation_seconds: float = Field(gt=0)
    clips: tuple[TtsPilotClip, ...]
    failures: tuple[TtsPilotFailure, ...]

    @model_validator(mode="after")
    def validate_result_identity(self) -> "TtsPilotResult":
        successful_ids = [clip.case.case_id for clip in self.clips]
        failed_ids = [failure.case.case_id for failure in self.failures]
        if len(set(successful_ids)) != len(successful_ids):
            raise ValueError("Successful case IDs must be unique")
        if len(set(failed_ids)) != len(failed_ids):
            raise ValueError("Failed case IDs must be unique")
        if set(successful_ids) & set(failed_ids):
            raise ValueError("A case cannot be both successful and failed")
        return self


def load_pilot_manifest(path: Path) -> TtsPilotManifest:
    return TtsPilotManifest.model_validate_json(path.read_text(encoding="utf-8"))
