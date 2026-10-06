"""Durable exact-EOS audio evidence for the paired NeuTTS corpus."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile
from pydantic import Field, model_validator

from scripts.neutts_pilot_state import NeuTtsPilotConfig, digest, write_record
from speech_projector.models import Record
from speech_projector.neutts_batch_benchmark import BatchClipEvidence, BatchMeasurement
from speech_projector.tts_pilot import PilotTermination, TtsPilotCase, TtsPilotManifest


class NeuCorpusConfig(Record):
    pilot: NeuTtsPilotConfig
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    batch_size: int = Field(default=28, ge=1, le=28)
    attempts: int = Field(default=3, ge=1, le=3)


class NeuCorpusSession(Record):
    session_index: int = Field(ge=0)
    resumed_clips: int = Field(ge=0)
    generated_clips: int = Field(ge=0)
    failed_cases: tuple[str, ...]
    startup_seconds: float = Field(ge=0)
    reference_seconds: float = Field(ge=0)
    warmup_seconds: float = Field(ge=0)
    wall_seconds: float = Field(ge=0)
    measured_batch_seconds: float = Field(ge=0)
    generated_audio_seconds: float = Field(ge=0)
    peak_allocated_gb: float = Field(ge=0)


class NeuCorpusStart(Record):
    session_index: int = Field(ge=0)
    started_at: datetime
    resumed_clips: int = Field(ge=0)
    pending_clips: int = Field(ge=0)
    helper_sha256: str


class NeuCorpusFailure(Record):
    cases: tuple[TtsPilotCase, ...]
    attempt: int = Field(ge=0)
    error: str


@dataclass(frozen=True)
class AudioVerification:
    verified_clips: int
    complete_pairs: int
    missing_cases: tuple[str, ...]
    total_audio_seconds: float
    audio_bytes: int
    over_30_second_cases: tuple[str, ...]
    peak_amplitude: float
    overshoot_samples: int
    waveform_samples: int


class NeuCorpusResult(Record):
    configuration: NeuCorpusConfig
    expected_cases: int = Field(ge=1)
    verified_clips: int = Field(ge=0)
    complete_pairs: int = Field(ge=0)
    missing_cases: tuple[str, ...]
    total_audio_seconds: float = Field(ge=0)
    audio_bytes: int = Field(ge=0)
    over_30_second_cases: tuple[str, ...]
    peak_amplitude: float = Field(ge=0)
    overshoot_samples: int = Field(ge=0)
    waveform_samples: int = Field(ge=0)
    termination_evidence: str
    torch_version: str
    transformers_version: str
    neucodec_version: str
    helper_sha256: str
    reference_sha256: str
    watermark_active: bool
    generation_token_cap: int = Field(gt=0)
    speech_end_token_id: int = Field(ge=0)
    sessions: tuple[NeuCorpusSession, ...]
    interrupted_sessions: tuple[int, ...]

    @model_validator(mode="after")
    def validate_coverage(self) -> "NeuCorpusResult":
        if self.verified_clips + len(self.missing_cases) != self.expected_cases:
            raise ValueError("Verified and missing clips must cover the manifest")
        return self


def validate_paired_manifest(manifest: TtsPilotManifest) -> None:
    families: dict[str, list[TtsPilotCase]] = {}
    for case in manifest.cases:
        families.setdefault(case.utterance_id, []).append(case)
        if case.seed != 42:
            raise ValueError("Production batch cases require the shared seed42")
        if case.emotion.value not in {"happy", "sad", "angry", "fearful"}:
            raise ValueError("Production accepts only the four approved emotional categories")
    if any(len(cases) != 2 for cases in families.values()):
        raise ValueError("Every production utterance must have exactly two deliveries")


def restore_evidence(case: TtsPilotCase, directory: Path) -> BatchClipEvidence | None:
    path = directory / "clips" / f"{case.case_id}.json"
    if not path.exists():
        return None
    item = BatchClipEvidence.model_validate_json(path.read_bytes())
    # Retry seeds are attempt-specific; the canonical corpus case remains seed42.
    if item.clip.case.model_copy(update={"seed": case.seed}) != case:
        raise ValueError(f"Saved case differs from canonical manifest: {case.case_id}")
    if item.clip.termination != PilotTermination.STOP:
        raise ValueError(f"A token-capped attempt was marked complete: {case.case_id}")
    if not item.generated_token_ids:
        raise ValueError(f"Missing true termination token evidence: {case.case_id}")
    audio_path = directory / item.clip.audio_path
    if not audio_path.is_file() or digest(audio_path) != item.clip.sha256:
        raise ValueError(f"Saved audio missing or hash mismatch: {case.case_id}")
    return item


def persist_completed(measurement: BatchMeasurement, directory: Path) -> None:
    for item in measurement.clips:
        if item.clip.termination == PilotTermination.STOP:
            path = directory / "clips" / f"{item.clip.case.case_id}.json"
            if path.exists():
                raise ValueError(f"Refusing to replace completed clip: {item.clip.case.case_id}")
            write_record(path, item)


def verify_audio(
    cases: Iterable[TtsPilotCase], directory: Path, end_token: int
) -> AudioVerification:
    verified: set[str] = set()
    family_members: dict[str, list[str]] = {}
    missing: list[str] = []
    audio_seconds = 0.0
    audio_bytes = 0
    long_cases: list[str] = []
    peak_amplitude = 0.0
    overshoot_samples = 0
    waveform_samples = 0
    for case in cases:
        family_members.setdefault(case.utterance_id, []).append(case.case_id)
        item = restore_evidence(case, directory)
        if item is None:
            missing.append(case.case_id)
            continue
        if item.generated_token_ids[-1] != end_token:
            raise ValueError(f"Saved completion lacks actual speech-end EOS: {case.case_id}")
        path = directory / item.clip.audio_path
        waveform, sample_rate = soundfile.read(path, dtype="float32")
        if sample_rate != 24000 or waveform.ndim != 1 or not np.isfinite(waveform).all():
            raise ValueError(f"Invalid audio format/content: {case.case_id}")
        duration = len(waveform) / sample_rate
        if not np.isclose(duration, item.clip.audio_seconds, rtol=0, atol=1 / sample_rate):
            raise ValueError(f"Recorded audio duration differs: {case.case_id}")
        if duration > 30:
            long_cases.append(case.case_id)
        absolute = np.abs(waveform)
        peak_amplitude = max(peak_amplitude, float(absolute.max()))
        overshoot_samples += int(np.count_nonzero(absolute > 1))
        waveform_samples += len(waveform)
        audio_seconds += duration
        audio_bytes += path.stat().st_size
        verified.add(case.case_id)
    pairs = sum(
        all(identifier in verified for identifier in members) for members in family_members.values()
    )
    return AudioVerification(
        verified_clips=len(verified),
        complete_pairs=pairs,
        missing_cases=tuple(missing),
        total_audio_seconds=audio_seconds,
        audio_bytes=audio_bytes,
        over_30_second_cases=tuple(long_cases),
        peak_amplitude=peak_amplitude,
        overshoot_samples=overshoot_samples,
        waveform_samples=waveform_samples,
    )
