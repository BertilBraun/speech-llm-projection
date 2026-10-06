"""Configuration and durable records for the bounded IndexTTS comparison."""

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from speech_projector.tts_pilot import (
    PilotEmotion,
    TtsPilotCase,
    TtsPilotClip,
    TtsPilotManifest,
)


class IndexPilotConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_path: Path
    output_directory: Path
    checkpoint_directory: Path
    reference_audio: Path
    reference_sha256: str
    model_revision: str
    vendor_source_commit: str
    source_commit: str
    emotion_intensity: float = Field(default=0.7, gt=0, le=1)
    max_mel_tokens: int = Field(default=1500, gt=0)
    top_p: float = Field(default=0.8, gt=0, le=1)
    top_k: int = Field(default=30, gt=0)
    temperature: float = Field(default=0.8, gt=0)
    num_beams: int = Field(default=3, gt=0)
    repetition_penalty: float = Field(default=10.0, gt=0)


class IndexSession(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    startup_seconds: float = Field(ge=0)
    warmup_generation_seconds: float = Field(gt=0)
    torch_version: str
    transformers_version: str
    reference_sha256: str
    manifest_sha256: str
    warmup_warnings: tuple[str, ...]


class IndexAttemptFailure(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case: TtsPilotCase
    error: str
    generation_seconds: float = Field(ge=0)
    warnings: tuple[str, ...]
    preserved_audio: tuple[str, ...]


class IndexEmotionControl(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case: TtsPilotCase
    vector: tuple[float, ...]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def emotion_vector(emotion: PilotEmotion, intensity: float) -> tuple[float, ...]:
    if not 0 < intensity <= 1:
        raise ValueError("Emotion intensity must be in (0, 1]")
    match emotion:
        case PilotEmotion.HAPPY:
            index = 0
        case PilotEmotion.ANGRY:
            index = 1
        case PilotEmotion.SAD:
            index = 2
        case PilotEmotion.AFRAID:
            index = 3
        case PilotEmotion.DISGUSTED:
            index = 4
        case PilotEmotion.MELANCHOLIC:
            index = 5
        case PilotEmotion.SURPRISED:
            index = 6
        case PilotEmotion.NEUTRAL:
            index = 7
        case PilotEmotion.FEARFUL:
            raise ValueError("IndexTTS calls its fear component afraid; use that label")
    return tuple(intensity if position == index else 0.0 for position in range(8))


def write_record(path: Path, record: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".temporary")
    temporary.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_output(configuration: IndexPilotConfig, manifest: TtsPilotManifest) -> None:
    if digest(configuration.reference_audio) != configuration.reference_sha256:
        raise ValueError("Speaker reference hash differs from configuration")
    directory = configuration.output_directory
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "config.json"
    if config_path.exists():
        saved = IndexPilotConfig.model_validate_json(config_path.read_bytes())
        if saved != configuration:
            raise ValueError("IndexTTS resume configuration changed")
        saved_manifest = TtsPilotManifest.model_validate_json(
            (directory / "cases.json").read_bytes()
        )
        if saved_manifest != manifest:
            raise ValueError("IndexTTS resume case manifest changed")
    else:
        if any(directory.iterdir()):
            raise ValueError("New output directory contains unowned artifacts")
        write_record(config_path, configuration)
        write_record(directory / "cases.json", manifest)


def completed_clip(configuration: IndexPilotConfig, case: TtsPilotCase) -> TtsPilotClip | None:
    path = configuration.output_directory / "clips" / f"{case.case_id}.json"
    if not path.exists():
        return None
    clip = TtsPilotClip.model_validate_json(path.read_bytes())
    if clip.case != case:
        raise ValueError("Saved IndexTTS clip case differs from manifest")
    audio = configuration.output_directory / clip.audio_path
    if digest(audio) != clip.sha256:
        raise ValueError("Saved IndexTTS WAV hash differs from clip")
    return clip
