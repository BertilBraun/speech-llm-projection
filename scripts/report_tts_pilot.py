"""Verify relocated pilot WAVs and report measured batch-one generation speed."""

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict

from speech_projector.tts_pilot import TtsPilotManifest, TtsPilotResult, load_pilot_manifest


@dataclass(frozen=True)
class ModelResults:
    directory: Path
    result: TtsPilotResult


class ModelAudit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_repository: str
    clip_count: int
    generation_seconds: float
    audio_seconds: float
    aggregate_real_time_factor: float
    startup_seconds: float
    reference_preparation_seconds: float
    warmup_generation_seconds: float


class PilotAudit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    models: tuple[ModelAudit, ...]


def verify_results(manifest: TtsPilotManifest, model: ModelResults) -> ModelAudit:
    expected_cases = {case.case_id: case for case in manifest.cases}
    actual_cases = {clip.case.case_id: clip.case for clip in model.result.clips}
    if actual_cases != expected_cases or len(model.result.clips) != len(expected_cases):
        raise ValueError(f"Incomplete or mismatched cases: {model.result.model_repository}")
    if model.result.failures:
        raise ValueError(f"Unresolved pilot failures: {model.result.model_repository}")
    for clip in model.result.clips:
        audio_path = (model.directory / clip.audio_path).resolve()
        if not audio_path.is_relative_to(model.directory.resolve()):
            raise ValueError(f"Non-relocatable audio path: {clip.audio_path}")
        digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
        if digest != clip.sha256:
            raise ValueError(f"WAV hash mismatch: {audio_path}")
        waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        if waveform.shape[1] != 1 or not np.isfinite(waveform).all():
            raise ValueError(f"Invalid mono waveform: {audio_path}")
        if np.max(np.abs(waveform)) > 1.0 or not np.any(waveform):
            raise ValueError(f"Invalid waveform amplitude: {audio_path}")
        duration = len(waveform) / sample_rate
        if sample_rate != clip.sample_rate or not math.isclose(duration, clip.audio_seconds):
            raise ValueError(f"WAV duration mismatch: {audio_path}")
        if not math.isclose(clip.real_time_factor, clip.generation_seconds / duration):
            raise ValueError(f"Real-time factor mismatch: {audio_path}")
    generation_seconds = sum(clip.generation_seconds for clip in model.result.clips)
    audio_seconds = sum(clip.audio_seconds for clip in model.result.clips)
    return ModelAudit(
        model_repository=model.result.model_repository,
        clip_count=len(model.result.clips),
        generation_seconds=generation_seconds,
        audio_seconds=audio_seconds,
        aggregate_real_time_factor=generation_seconds / audio_seconds,
        startup_seconds=model.result.startup_seconds,
        reference_preparation_seconds=model.result.reference_preparation_seconds,
        warmup_generation_seconds=model.result.warmup_generation_seconds,
    )


def render_comparison(manifest: TtsPilotManifest, models: tuple[ModelResults, ...]) -> str:
    lines = [
        "# IndexTTS / NeuTTS listening comparison",
        "",
        "Measured on one RTX 3090, one model at a time, sequential batch one.",
        "RTF = generation seconds / generated audio seconds; lower is faster.",
        "Generation includes the synthesis call and codec, excludes WAV writing,",
        "downloads, initialization and the separately recorded warmup.",
        "",
        "The intended neutral IndexTTS control uses the calm vector. Angry is not",
        "silently relabelled as frustrated. The voice is fixed within each model;",
        "IndexTTS clones the bundled NeuTTS Paul reference, without guaranteed voice parity.",
        "This is a listening pilot, not verified emotion labels or a training dataset.",
        "",
        "| Model | Clips | Generation s | Audio s | Aggregate RTF | Startup s | Warmup s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in models:
        audit = verify_results(manifest, model)
        lines.append(
            f"| {audit.model_repository} | {audit.clip_count} | "
            f"{audit.generation_seconds:.3f} | {audit.audio_seconds:.3f} | "
            f"{audit.aggregate_real_time_factor:.3f} | {audit.startup_seconds:.3f} | "
            f"{audit.warmup_generation_seconds:.3f} |"
        )
    for case in manifest.cases:
        lines.extend(["", f"## {case.utterance_id}: {case.emotion.value}", "", f"> {case.text}"])
        for model in models:
            clip = next(clip for clip in model.result.clips if clip.case == case)
            audio_path = (model.directory / clip.audio_path).resolve().as_posix()
            lines.extend(
                [
                    "",
                    f"**{model.result.model_repository}**",
                    "",
                    f"![{case.case_id} {model.result.model_repository}](<{audio_path}>)",
                    "",
                    f"{clip.generation_seconds:.3f}s generation / {clip.audio_seconds:.3f}s "
                    f"audio = RTF {clip.real_time_factor:.3f}; "
                    f"termination {clip.termination.value}.",
                ]
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = load_pilot_manifest(arguments.manifest)
    models = tuple(
        ModelResults(
            directory=directory,
            result=TtsPilotResult.model_validate_json(
                (directory / "result.json").read_text(encoding="utf-8")
            ),
        )
        for directory in arguments.results
    )
    audit = PilotAudit(models=tuple(verify_results(manifest, model) for model in models))
    comparison = render_comparison(manifest, models)
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "verification.json").write_text(
        audit.model_dump_json(indent=2), encoding="utf-8"
    )
    (arguments.output / "listening_comparison.md").write_text(comparison, encoding="utf-8")
    print(audit.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
