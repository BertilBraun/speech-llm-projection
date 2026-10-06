"""SDK-independent identity, persistence and resume state for the NeuTTS pilot."""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path

import numpy as np
import soundfile
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from scripts.prepare_neutts_models import NeuTtsPreparation
from speech_projector.tts_pilot import (
    PilotTermination,
    TtsPilotCase,
    TtsPilotClip,
    TtsPilotFailure,
    TtsPilotResult,
)


class PilotDevice(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class NeuTtsPilotConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    manifest: Path
    manifest_sha256: str
    output_directory: Path
    repository_directory: Path
    preparation: NeuTtsPreparation
    device: PilotDevice
    speaker: str = "paul"
    temperature: float = Field(default=1.0, gt=0)
    top_k: int = Field(default=50, gt=0)
    cpu_threads: int = Field(default=4, gt=0)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_record(path: Path, record: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def validate_configuration(configuration: NeuTtsPilotConfig) -> None:
    if digest(configuration.manifest) != configuration.manifest_sha256:
        raise ValueError("Pilot manifest content differs from pinned configuration")
    path = configuration.output_directory / "config.json"
    if path.exists():
        previous = NeuTtsPilotConfig.model_validate_json(path.read_bytes())
        if previous != configuration:
            raise ValueError("Refusing resume with changed configuration or manifest identity")
    else:
        write_record(path, configuration)


def restore_clip(case: TtsPilotCase, directory: Path) -> TtsPilotClip | None:
    record_path = directory / "clips" / f"{case.case_id}.json"
    if not record_path.exists():
        return None
    clip = TtsPilotClip.model_validate_json(record_path.read_bytes())
    if clip.case != case or clip.audio_path != f"audio/{case.case_id}.wav":
        raise ValueError(f"Saved clip identity differs from current case: {case.case_id}")
    path = directory / clip.audio_path
    partial = path.with_suffix(".wav.part")
    candidate = path if path.exists() else partial
    if not candidate.exists() or digest(candidate) != clip.sha256:
        raise ValueError(f"Saved clip audio missing or hash mismatch: {case.case_id}")
    if candidate == partial:
        partial.replace(path)
    return clip


def persist_clip(
    configuration: NeuTtsPilotConfig,
    case: TtsPilotCase,
    waveform: NDArray[np.float32],
    elapsed: float,
    sample_rate: int,
) -> TtsPilotClip:
    relative = Path("audio") / f"{case.case_id}.wav"
    path = configuration.output_directory / relative
    partial = path.with_suffix(".wav.part")
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(partial, waveform, sample_rate, subtype="FLOAT", format="WAV")
    audio_seconds = waveform.size / sample_rate
    clip = TtsPilotClip(
        case=case,
        audio_path=relative.as_posix(),
        sha256=digest(partial),
        sample_rate=sample_rate,
        audio_seconds=audio_seconds,
        generation_seconds=elapsed,
        real_time_factor=elapsed / audio_seconds,
        termination=PilotTermination.NOT_EXPOSED,
    )
    write_record(configuration.output_directory / "clips" / f"{case.case_id}.json", clip)
    partial.replace(path)
    return clip


def completed_result(
    initialization: TtsPilotResult,
    clips: tuple[TtsPilotClip, ...],
    failures: tuple[TtsPilotFailure, ...],
) -> TtsPilotResult:
    return TtsPilotResult(
        model_repository=initialization.model_repository,
        model_revision=initialization.model_revision,
        source_commit=initialization.source_commit,
        backend_description=initialization.backend_description,
        speaker_description=initialization.speaker_description,
        startup_seconds=initialization.startup_seconds,
        reference_preparation_seconds=initialization.reference_preparation_seconds,
        warmup_generation_seconds=initialization.warmup_generation_seconds,
        clips=clips,
        failures=failures,
    )
