"""Official NeuTTS-2E eight-case pilot using pinned offline models and fixed Paul voice."""

from __future__ import annotations

import argparse
import random
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from neutts import NeuTTS2E
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from scripts.neutts_pilot_state import (
    NeuTtsPilotConfig,
    PilotDevice,
    completed_result,
    digest,
    persist_clip,
    restore_clip,
    validate_configuration,
    write_record,
)
from scripts.prepare_neutts_models import NeuTtsPreparation
from speech_projector.tts_pilot import (
    PilotEmotion,
    TtsPilotCase,
    TtsPilotClip,
    TtsPilotFailure,
    TtsPilotResult,
    load_pilot_manifest,
)


class NeuTtsRuntime(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    torch_version: str
    transformers_version: str
    neucodec_version: str
    sdk_source_commit: str
    helper_sha256: str
    manifest_sha256: str
    reference_wav_sha256: str
    reference_codes_sha256: str
    reference_text_sha256: str
    watermark_active: bool
    sdk_max_context: int
    sample_rate: int
    timing_scope: str


def synchronize(device: PilotDevice) -> None:
    if device == PilotDevice.CUDA:
        torch.cuda.synchronize()


def generate(
    model: NeuTTS2E, configuration: NeuTtsPilotConfig, case: TtsPilotCase
) -> tuple[NDArray[np.float32], float]:
    # The official infer API has constructor-only seed; this known SDK field pins each call.
    model._seed = case.seed
    random.seed(case.seed)
    np.random.seed(case.seed)
    synchronize(configuration.device)
    started = perf_counter()
    waveform = np.asarray(
        model.infer(
            case.text,
            speaker=configuration.speaker,
            emotion=case.emotion.value,
            temperature=configuration.temperature,
            top_k=configuration.top_k,
        ),
        dtype=np.float32,
    )
    synchronize(configuration.device)
    elapsed = perf_counter() - started
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"Official SDK returned invalid mono waveform: {case.case_id}")
    return waveform, elapsed


def run(configuration: NeuTtsPilotConfig) -> TtsPilotResult:
    manifest = load_pilot_manifest(configuration.manifest)
    validate_configuration(configuration)
    if configuration.speaker not in NeuTTS2E.SPEAKERS:
        raise ValueError(f"Unsupported official NeuTTS-2E speaker: {configuration.speaker}")
    supported_emotions = {"neutral", "happy", "angry", "sad", "fearful", "disgusted", "surprised"}
    for case in manifest.cases:
        if case.emotion.value not in supported_emotions:
            raise ValueError(f"Unsupported NeuTTS-2E emotion: {case.emotion.value}")
    backbone, codec = configuration.preparation.repositories
    if backbone.repository != "neuphonic/neutts-2e" or codec.repository != "neuphonic/neucodec":
        raise ValueError("Preparation must pin the official NeuTTS-2E backbone and NeuCodec")
    for repository in configuration.preparation.repositories:
        cache_reference = repository.snapshot.parent.parent / "refs/main"
        if cache_reference.read_text().strip() != repository.revision:
            raise ValueError(f"Isolated offline model ref changed: {repository.repository}")
    torch.set_num_threads(configuration.cpu_threads)
    restored = tuple(
        clip
        for case in manifest.cases
        if (clip := restore_clip(case, configuration.output_directory)) is not None
    )
    initialization_path = configuration.output_directory / "initialization.json"
    runtime_path = configuration.output_directory / "runtime.json"
    previous: TtsPilotResult | None = None
    if initialization_path.exists():
        previous = TtsPilotResult.model_validate_json(initialization_path.read_bytes())
        runtime = NeuTtsRuntime.model_validate_json(runtime_path.read_bytes())
        if runtime.helper_sha256 != digest(Path(__file__)):
            raise ValueError("Resume requires the exact original pilot helper source")
        if len(restored) == len(manifest.cases):
            result = completed_result(previous, restored, ())
            write_record(configuration.output_directory / "result.json", result)
            return result
    startup_started = perf_counter()
    model = NeuTTS2E(
        backbone_repo=str(backbone.snapshot),
        backbone_device=configuration.device.value,
        codec_repo=codec.repository,
        codec_device=configuration.device.value,
        seed=manifest.warmup_seed,
    )
    synchronize(configuration.device)
    startup_seconds = perf_counter() - startup_started
    synchronize(configuration.device)
    reference_started = perf_counter()
    model._speaker(configuration.speaker)
    synchronize(configuration.device)
    reference_seconds = perf_counter() - reference_started
    runtime = NeuTtsRuntime(
        torch_version=torch.__version__,
        transformers_version=version("transformers"),
        neucodec_version=version("neucodec"),
        sdk_source_commit=configuration.preparation.source_commit,
        helper_sha256=digest(Path(__file__)),
        manifest_sha256=digest(configuration.manifest),
        reference_wav_sha256=digest(
            configuration.repository_directory / "samples" / f"{configuration.speaker}.wav"
        ),
        reference_codes_sha256=digest(
            configuration.repository_directory / "samples" / f"{configuration.speaker}.pt"
        ),
        reference_text_sha256=digest(
            configuration.repository_directory / "samples" / f"{configuration.speaker}.txt"
        ),
        watermark_active=model.watermarker is not None,
        sdk_max_context=model.max_context,
        sample_rate=model.sample_rate,
        timing_scope="Startup: official SDK constructor and final CUDA synchronization; "
        "reference: loading bundled pre-encoded Paul codes/text; warmup excluded from timed cases; "
        "each clip: sequential batch1 infer including backbone, codec and optional watermark, "
        "CUDA synchronization before/after; disk writes and hashing excluded. "
        "No waveform normalization, trimming or resampling; float32 WAV preserves SDK samples. "
        "Public SDK exposes waveform only; EOS versus context-cap termination is unavailable.",
    )
    if previous is None:
        write_record(runtime_path, runtime)
    warmup = TtsPilotCase(
        case_id="warmup",
        utterance_id="warmup",
        text=manifest.warmup_text,
        emotion=PilotEmotion.NEUTRAL,
        seed=manifest.warmup_seed,
    )
    _, warmup_seconds = generate(model, configuration, warmup)
    initialization = TtsPilotResult(
        model_repository=backbone.repository,
        model_revision=backbone.revision,
        source_commit=configuration.preparation.source_commit,
        backend_description=f"Official NeuTTS SDK PyTorch BF16 backbone and NeuCodec on "
        f"{configuration.device.value}; temperature={configuration.temperature}, "
        f"top_k={configuration.top_k}; waveform-only API, termination NOT_EXPOSED; "
        "result.source_commit denotes vendor SDK source; "
        "runtime.helper_sha256 denotes pilot helper",
        speaker_description="Official bundled fixed Paul speaker; original paul.pt/paul.txt "
        "references; unaltered paul.wav shared with IndexTTS",
        startup_seconds=startup_seconds,
        reference_preparation_seconds=reference_seconds,
        warmup_generation_seconds=warmup_seconds,
        clips=(),
        failures=(),
    )
    attempt_directory = configuration.output_directory / "initialization_attempts"
    attempt_directory.mkdir(exist_ok=True)
    attempt_index = len(tuple(attempt_directory.glob("*.json")))
    write_record(attempt_directory / f"attempt_{attempt_index:03d}.json", initialization)
    if previous is None:
        write_record(initialization_path, initialization)
    clips: list[TtsPilotClip] = []
    failures: list[TtsPilotFailure] = []
    audio_directory = configuration.output_directory / "audio"
    audio_directory.mkdir(exist_ok=True)
    for case in manifest.cases:
        saved = next((clip for clip in restored if clip.case == case), None)
        if saved is not None:
            clips.append(saved)
            print(f"Reuse hash-verified clip: {case.case_id}", flush=True)
            continue
        try:
            waveform, elapsed = generate(model, configuration, case)
            clips.append(persist_clip(configuration, case, waveform, elapsed, model.sample_rate))
            print(clips[-1].model_dump_json(), flush=True)
        except (RuntimeError, ValueError) as error:
            failure = TtsPilotFailure(case=case, error=str(error))
            failures.append(failure)
            failure_path = configuration.output_directory / "failures" / case.case_id
            failure_path.mkdir(parents=True, exist_ok=True)
            write_record(
                failure_path / f"attempt_{len(tuple(failure_path.glob('*.json'))):03d}.json",
                failure,
            )
            print(failures[-1].model_dump_json(), flush=True)
    result = completed_result(previous or initialization, tuple(clips), tuple(failures))
    write_record(configuration.output_directory / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument(
        "--device", choices=tuple(item.value for item in PilotDevice), required=True
    )
    arguments = parser.parse_args()
    configuration = NeuTtsPilotConfig(
        manifest=arguments.manifest,
        manifest_sha256=digest(arguments.manifest),
        output_directory=arguments.output,
        repository_directory=arguments.repository,
        preparation=NeuTtsPreparation.model_validate_json(arguments.preparation.read_bytes()),
        device=PilotDevice(arguments.device),
    )
    result = run(configuration)
    print(result.model_dump_json(indent=2), flush=True)
    if result.failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
