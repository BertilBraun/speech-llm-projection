"""Fixed ten-case emotional speech preview and audio persistence boundaries."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import soundfile
from numpy.typing import NDArray
from pydantic import Field, model_validator

from scripts.inventory_results import stable_digest, write_record
from scripts.package_results import FileArtifact, ModelRevision
from speech_projector.models import Record

TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
TTS_REVISION = "0c0e3051f131929182e2c023b9537f8b1c68adfe"
PREVIEW_TEXTS = ("I'm good.", "That went really well.")


class Delivery(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    FRUSTRATED = "frustrated"
    SARCASTIC = "sarcastic"


class PreviewTermination(str, Enum):
    STOPPED_BEFORE_CAP = "stopped_before_cap"
    CAP_AMBIGUOUS = "cap_ambiguous"


class PreviewCase(Record):
    case_id: str
    text: str
    delivery: Delivery
    instruct: str
    seed: int = Field(ge=0)


class PreviewPlan(Record):
    model_name: str = TTS_MODEL
    revision: str = TTS_REVISION
    speaker: str = "Ryan"
    language: str = "English"
    seed: int = Field(default=42, ge=0)
    max_new_tokens: int = Field(default=256, ge=4)
    temperature: float = Field(default=0.9, gt=0)
    top_k: int = Field(default=50, gt=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    repetition_penalty: float = Field(default=1.05, gt=0)
    cases: tuple[PreviewCase, ...]

    @model_validator(mode="after")
    def validate_matrix(self) -> PreviewPlan:
        expected = tuple((text, delivery) for text in PREVIEW_TEXTS for delivery in Delivery)
        actual = tuple((case.text, case.delivery) for case in self.cases)
        if actual != expected or len({case.case_id for case in self.cases}) != 10:
            raise ValueError("Preview requires the ordered two-text, five-delivery matrix")
        if tuple(case.seed for case in self.cases) != tuple(self.seed + i for i in range(10)):
            raise ValueError("Preview case seeds must be plan seed plus case index")
        return self


class PreviewClip(Record):
    case: PreviewCase
    audio: FileArtifact
    sample_rate: int = Field(gt=0)
    samples: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    runtime_seconds: float = Field(ge=0)
    codec_tokens: int = Field(gt=0)
    peak_amplitude: float = Field(gt=0)


class PreviewCapFailure(Record):
    kind: Literal["cap_ambiguous"] = "cap_ambiguous"
    case: PreviewCase
    codec_tokens: int = Field(ge=0)
    max_new_tokens: int = Field(ge=4)
    runtime_seconds: float = Field(ge=0)


class PreviewAudioFailure(Record):
    kind: Literal["invalid_audio"] = "invalid_audio"
    case: PreviewCase
    error: str
    runtime_seconds: float = Field(ge=0)


class PreviewSynthesisFailure(Record):
    kind: Literal["synthesis_error"] = "synthesis_error"
    case: PreviewCase
    error: str
    runtime_seconds: float = Field(ge=0)


PreviewFailure = Annotated[
    PreviewCapFailure | PreviewAudioFailure | PreviewSynthesisFailure, Field(discriminator="kind")
]


class PreviewProvenance(Record):
    source_commit: str
    model_revision: ModelRevision
    input_sha256: str
    started_at: datetime
    qwen_tts_version: str
    torch_version: str
    transformers_version: str


class PreviewManifest(Record):
    plan: PreviewPlan
    clips: tuple[PreviewClip, ...]
    provenance: PreviewProvenance
    runtime_seconds: float = Field(ge=0)
    peak_vram_gb: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_complete(self) -> PreviewManifest:
        if tuple(clip.case for clip in self.clips) != self.plan.cases:
            raise ValueError("A completed preview manifest must cover its exact ten cases")
        if self.provenance.input_sha256 != plan_digest(self.plan):
            raise ValueError("Preview plan SHA256 differs from the manifest")
        if any(
            codec_termination(clip.codec_tokens, self.plan.max_new_tokens)
            != PreviewTermination.STOPPED_BEFORE_CAP
            for clip in self.clips
        ):
            raise ValueError("A completed preview cannot contain ambiguous capped clips")
        return self


@dataclass(frozen=True)
class SynthesizedAudio:
    waveform: NDArray[np.float32]
    sample_rate: int
    codec_tokens: int
    runtime_seconds: float


@dataclass(frozen=True)
class CappedSynthesis:
    codec_tokens: int
    runtime_seconds: float


def delivery_instruction(delivery: Delivery) -> str:
    match delivery:
        case Delivery.NEUTRAL:
            return "Speak naturally with a calm, matter-of-fact tone."
        case Delivery.HAPPY:
            return "Sound genuinely pleased, with a light warmth and a subtle smile in your voice."
        case Delivery.SAD:
            return "Sound quietly sad and a little subdued, without crying or exaggeration."
        case Delivery.FRUSTRATED:
            return "Sound mildly frustrated and tense, but controlled, without shouting."
        case Delivery.SARCASTIC:
            return (
                "Use dry, understated sarcasm, with subtle ironic emphasis rather than enthusiasm."
            )


def default_preview_plan(max_new_tokens: int = 256, *, revision: str = TTS_REVISION) -> PreviewPlan:
    cases = tuple(
        PreviewCase(
            case_id=f"{prefix}_{delivery.value}",
            text=text,
            delivery=delivery,
            instruct=delivery_instruction(delivery),
            seed=42 + text_index * len(Delivery) + delivery_index,
        )
        for text_index, (prefix, text) in enumerate(
            zip(("good", "well"), PREVIEW_TEXTS, strict=True)
        )
        for delivery_index, delivery in enumerate(Delivery)
    )
    return PreviewPlan(max_new_tokens=max_new_tokens, revision=revision, cases=cases)


def plan_digest(plan: PreviewPlan) -> str:
    return hashlib.sha256(plan.model_dump_json().encode("utf-8")).hexdigest()


def codec_termination(codec_tokens: int, max_new_tokens: int) -> PreviewTermination:
    if codec_tokens < 0 or max_new_tokens < 4:
        raise ValueError("Codec count must be nonnegative and generation cap at least four")
    # The SDK removes EOS and can omit the initial prefill from its codec sequence.
    if codec_tokens >= max_new_tokens - 1:
        return PreviewTermination.CAP_AMBIGUOUS
    return PreviewTermination.STOPPED_BEFORE_CAP


def normalize_waveform(waveform: NDArray[np.float32]) -> NDArray[np.float32]:
    audio = np.ascontiguousarray(waveform, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("TTS must return a nonempty mono waveform")
    if not np.isfinite(audio).all() or not np.any(audio):
        raise ValueError("TTS returned nonfinite or entirely silent audio")
    return audio


def persist_clip(
    directory: Path, plan: PreviewPlan, case: PreviewCase, synthesis: SynthesizedAudio
) -> PreviewClip:
    if case not in plan.cases:
        raise ValueError("Preview case is absent from its plan")
    if codec_termination(synthesis.codec_tokens, plan.max_new_tokens) != (
        PreviewTermination.STOPPED_BEFORE_CAP
    ):
        raise ValueError("Codec generation did not demonstrably stop before its cap")
    audio = normalize_waveform(synthesis.waveform)
    if synthesis.sample_rate <= 0:
        raise ValueError("TTS sample rate must be positive")
    destination = directory / "audio" / f"{case.case_id}.wav"
    if destination.exists():
        raise ValueError(f"Refusing to overwrite existing preview audio: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    soundfile.write(partial, audio, synthesis.sample_rate, format="WAV", subtype="FLOAT")
    partial.replace(destination)
    size, digest = stable_digest(destination)
    clip = PreviewClip(
        case=case,
        audio=FileArtifact(
            path=destination.relative_to(directory),
            source_path=destination.resolve(),
            bytes=size,
            sha256=digest,
        ),
        sample_rate=synthesis.sample_rate,
        samples=audio.size,
        duration_seconds=audio.size / synthesis.sample_rate,
        runtime_seconds=synthesis.runtime_seconds,
        codec_tokens=synthesis.codec_tokens,
        peak_amplitude=float(np.abs(audio).max()),
    )
    write_record(directory / "clips" / f"{case.case_id}.json", clip)
    return clip


def render_preview(manifest: PreviewManifest) -> str:
    lines = [
        "# Ten emotional delivery previews",
        "",
        f"Model: {manifest.plan.model_name}; voice: {manifest.plan.speaker}; "
        f"language: {manifest.plan.language}.",
        "",
        "Delivery labels are synthesis instructions, not independently verified emotions.",
        "The SDK strips EOS. Accepted clips stopped before the conservative codec cap boundary; "
        "raw EOS was not observed.",
        "",
        f"Runtime: {manifest.runtime_seconds:.2f}s; peak PyTorch allocated memory: "
        f"{manifest.peak_vram_gb:.3f} decimal GB.",
        "",
    ]
    for clip in manifest.clips:
        lines.extend(
            (
                f"## {clip.case.case_id}",
                "",
                f'Text: "{clip.case.text}"',
                "",
                f"Delivery: {clip.case.delivery.value}; instruction: {clip.case.instruct}",
                "",
                f"Duration: {clip.duration_seconds:.3f}s; sample rate: {clip.sample_rate}Hz; "
                f"codec frames: {clip.codec_tokens}; synthesis: {clip.runtime_seconds:.3f}s; "
                f"seed: {clip.case.seed}.",
                "",
                f"[Listen]({clip.audio.path.as_posix()})",
                "",
            )
        )
    return "\n".join(lines)
