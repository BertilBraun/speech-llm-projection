"""Compare genuine NeuTTS batched backbone generation with exact-length codec decoding."""

import argparse
import random
import re
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile
import torch
from neutts import NeuTTS2E
from numpy.typing import NDArray
from torch import Tensor
from transformers import PreTrainedTokenizerBase, Qwen3ForCausalLM

from scripts.neutts_pilot_state import NeuTtsPilotConfig, PilotDevice, digest, write_record
from scripts.prepare_neutts_models import NeuTtsPreparation
from speech_projector.neutts_batch_benchmark import (
    BatchClipEvidence,
    BatchMeasurement,
    NeuTtsBenchmarkConfig,
    NeuTtsBenchmarkResult,
    completed_tokens,
    prompt_emotions,
    verify_recorded_audio,
)
from speech_projector.tts_pilot import (
    TtsPilotCase,
    TtsPilotClip,
    load_pilot_manifest,
)


@dataclass(frozen=True)
class BatchAudio:
    waveform: NDArray[np.float32]
    token_ids: tuple[int, ...]
    prompt_tokens: int
    speech_tokens: int


@dataclass(frozen=True)
class GeneratedBatch:
    audio: tuple[BatchAudio, ...]
    padded_prompt_width: int
    generation_token_cap: int
    frontend_seconds: float
    backbone_seconds: float
    codec_watermark_seconds: float
    end_to_end_seconds: float


@torch.inference_mode()
def synthesize_batch(
    model: NeuTTS2E,
    configuration: NeuTtsPilotConfig,
    cases: tuple[TtsPilotCase, ...],
    reference_codes: Tensor,
    reference_text: str,
) -> GeneratedBatch:
    if len({case.seed for case in cases}) != 1:
        raise ValueError("This benchmark requires one shared seed per batch")
    tokenizer: PreTrainedTokenizerBase = model.tokenizer
    backbone: Qwen3ForCausalLM = model.backbone
    if backbone.generation_config.max_new_tokens is None:
        raise ValueError("Pinned NeuTTS generation config lacks max_new_tokens")
    generation_token_cap = backbone.generation_config.max_new_tokens
    torch.cuda.synchronize()
    started = perf_counter()
    emotions = prompt_emotions(cases, model._check_emotion)
    prompts: tuple[list[int], ...] = tuple(
        model._apply_chat_template(reference_codes, reference_text, case.text, emotion)
        for case, emotion in zip(cases, emotions, strict=True)
    )
    width = max(len(prompt) for prompt in prompts)
    if width + 50 >= model.max_context:
        raise ValueError("Prompt leaves insufficient space for the SDK minimum generation length")
    pad_token = tokenizer.pad_token_id
    if pad_token is None:
        raise ValueError("NeuTTS tokenizer has no required batch padding token")
    inputs = torch.full((len(cases), width), pad_token, dtype=torch.long, device=backbone.device)
    mask = torch.zeros_like(inputs)
    for index, prompt in enumerate(prompts):
        inputs[index, -len(prompt) :] = torch.tensor(prompt, device=backbone.device)
        mask[index, -len(prompt) :] = 1
    end_token: int = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
    torch.cuda.synchronize()
    frontend_seconds = perf_counter() - started
    generation_started = perf_counter()
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(cases[0].seed)
        random.seed(cases[0].seed)
        np.random.seed(cases[0].seed)
        output: Tensor = backbone.generate(
            input_ids=inputs,
            attention_mask=mask,
            max_length=model.max_context,
            eos_token_id=end_token,
            pad_token_id=pad_token,
            do_sample=True,
            temperature=configuration.temperature,
            top_k=configuration.top_k,
            use_cache=True,
            min_new_tokens=50,
            return_dict_in_generate=False,
        )
    torch.cuda.synchronize()
    backbone_seconds = perf_counter() - generation_started
    codec_started = perf_counter()
    rows: list[BatchAudio] = []
    for index, case in enumerate(cases):
        token_ids: tuple[int, ...] = tuple(output[index, width:].tolist())
        retained, _ = completed_tokens(token_ids, end_token)
        output_text = tokenizer.decode(retained, add_special_tokens=False)
        waveform: NDArray[np.float32] = np.asarray(model._decode(output_text), dtype=np.float32)
        if model.watermarker is not None:
            waveform = np.asarray(
                model.watermarker.apply_watermark(waveform, sample_rate=24000), dtype=np.float32
            )
        if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
            raise ValueError(f"Invalid decoded waveform for {case.case_id}")
        rows.append(
            BatchAudio(
                waveform=waveform,
                token_ids=retained,
                prompt_tokens=len(prompts[index]),
                speech_tokens=len(re.findall(r"<\|speech_(\d+)\|>", output_text)),
            )
        )
    torch.cuda.synchronize()
    return GeneratedBatch(
        audio=tuple(rows),
        padded_prompt_width=width,
        generation_token_cap=generation_token_cap,
        frontend_seconds=frontend_seconds,
        backbone_seconds=backbone_seconds,
        codec_watermark_seconds=perf_counter() - codec_started,
        end_to_end_seconds=perf_counter() - started,
    )


def persist_batch(
    configuration: NeuTtsBenchmarkConfig,
    cases: tuple[TtsPilotCase, ...],
    generated: GeneratedBatch,
    batch_size: int,
    repetition: int,
    batch_index: int,
    end_token: int,
    sample_rate: int,
) -> BatchMeasurement:
    directory = configuration.pilot.output_directory
    clips: list[BatchClipEvidence] = []
    for case, audio in zip(cases, generated.audio, strict=True):
        relative = Path("audio") / f"batch{batch_size}/repeat{repetition}" / f"{case.case_id}.wav"
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".temporary.wav")
        soundfile.write(partial, audio.waveform, sample_rate, subtype="FLOAT")
        partial.replace(path)
        duration = audio.waveform.size / sample_rate
        _, termination = completed_tokens(audio.token_ids, end_token)
        clip = TtsPilotClip(
            case=case,
            audio_path=relative.as_posix(),
            sha256=digest(path),
            sample_rate=sample_rate,
            audio_seconds=duration,
            generation_seconds=generated.end_to_end_seconds,
            real_time_factor=generated.end_to_end_seconds / duration,
            termination=termination,
        )
        clips.append(
            BatchClipEvidence(
                clip=clip,
                prompt_tokens=audio.prompt_tokens,
                speech_tokens=audio.speech_tokens,
                generated_token_ids=audio.token_ids,
            )
        )
    measurement = BatchMeasurement(
        requested_batch_size=batch_size,
        repetition=repetition,
        batch_index=batch_index,
        shared_batch_seed=cases[0].seed,
        padded_prompt_width=generated.padded_prompt_width,
        generation_token_cap=generated.generation_token_cap,
        frontend_seconds=generated.frontend_seconds,
        backbone_seconds=generated.backbone_seconds,
        codec_watermark_seconds=generated.codec_watermark_seconds,
        end_to_end_seconds=generated.end_to_end_seconds,
        clips=tuple(clips),
    )
    write_record(
        directory / "measurements" / f"batch{batch_size}_{repetition}_{batch_index}.json",
        measurement,
    )
    return measurement


def benchmark(configuration: NeuTtsBenchmarkConfig) -> NeuTtsBenchmarkResult:
    pilot = configuration.pilot
    if digest(pilot.manifest) != pilot.manifest_sha256:
        raise ValueError("Benchmark manifest hash differs from configuration")
    manifest = load_pilot_manifest(pilot.manifest)
    if len(manifest.cases) != 7 or {case.emotion.value for case in manifest.cases} != set(
        NeuTTS2E.EMOTIONS
    ):
        raise ValueError("Benchmark requires the exact seven NeuTTS emotions")
    if len({case.text for case in manifest.cases}) != 1:
        raise ValueError("All benchmark emotions must share the same literal sentence")
    directory = pilot.output_directory
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "config.json").exists():
        saved = NeuTtsBenchmarkConfig.model_validate_json((directory / "config.json").read_bytes())
        if saved != configuration:
            raise ValueError("Refusing changed benchmark configuration")
        if (directory / "result.json").exists():
            result = NeuTtsBenchmarkResult.model_validate_json(
                (directory / "result.json").read_bytes()
            )
            verify_recorded_audio(result, directory)
            return result
        raise ValueError(
            "Partial benchmark preserved; use a new output directory for a complete timed replay"
        )
    if any(directory.iterdir()):
        raise ValueError("New benchmark directory contains unowned files")
    write_record(directory / "config.json", configuration)
    write_record(directory / "cases.json", manifest)
    torch.set_num_threads(pilot.cpu_threads)
    backbone, codec = pilot.preparation.repositories
    for repository in pilot.preparation.repositories:
        if (
            repository.snapshot.parent.parent / "refs/main"
        ).read_text().strip() != repository.revision:
            raise ValueError("Offline model reference differs from pinned benchmark revision")
    torch.cuda.synchronize()
    started = perf_counter()
    model = NeuTTS2E(
        backbone_repo=str(backbone.snapshot),
        backbone_device="cuda",
        codec_repo=codec.repository,
        codec_device="cuda",
        seed=manifest.warmup_seed,
    )
    torch.cuda.synchronize()
    startup_seconds = perf_counter() - started
    if model._is_quantized_model or model._is_onnx_codec:
        raise ValueError("Benchmark requires the fixed PyTorch backbone and standard NeuCodec")
    if model.backbone.generation_config.max_new_tokens is None:
        raise ValueError("Pinned NeuTTS generation config lacks max_new_tokens")
    (directory / "vendor_generation_config.json").write_text(
        model.backbone.generation_config.to_json_string(use_diff=False), encoding="utf-8"
    )
    reference_started = perf_counter()
    reference_codes, reference_text = model._speaker(pilot.speaker)
    torch.cuda.synchronize()
    reference_seconds = perf_counter() - reference_started
    warmup_seconds = 0.0
    for batch_size in configuration.batch_sizes:
        for offset in range(0, len(manifest.cases), batch_size):
            warmed = synthesize_batch(
                model,
                pilot,
                manifest.cases[offset : offset + batch_size],
                reference_codes,
                reference_text,
            )
            warmup_seconds += warmed.end_to_end_seconds
    measurements: list[BatchMeasurement] = []
    end_token: int = model.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
    for repetition in range(configuration.repeats):
        order = (
            configuration.batch_sizes
            if repetition % 2 == 0
            else tuple(reversed(configuration.batch_sizes))
        )
        for batch_size in order:
            for offset in range(0, len(manifest.cases), batch_size):
                cases = manifest.cases[offset : offset + batch_size]
                generated = synthesize_batch(model, pilot, cases, reference_codes, reference_text)
                measurement = persist_batch(
                    configuration,
                    cases,
                    generated,
                    batch_size,
                    repetition,
                    offset // batch_size,
                    end_token,
                    model.sample_rate,
                )
                measurements.append(measurement)
                print(measurement.model_dump_json(), flush=True)
    result = NeuTtsBenchmarkResult(
        configuration=configuration,
        startup_seconds=startup_seconds,
        reference_preparation_seconds=reference_seconds,
        warmup_seconds=warmup_seconds,
        torch_version=torch.__version__,
        transformers_version=version("transformers"),
        neucodec_version=version("neucodec"),
        helper_sha256=digest(Path(__file__)),
        reference_sha256=digest(pilot.repository_directory / "samples" / f"{pilot.speaker}.wav"),
        watermark_active=model.watermarker is not None,
        effective_max_new_tokens=model.backbone.generation_config.max_new_tokens,
        measurements=tuple(measurements),
        timing_scope=(
            "Three measured passes per batch size after warming both shapes. True batched "
            "Qwen3 backbone with left padding and mask; individual exact-length standard "
            "NeuCodec decode plus official watermark. Synchronized E2E includes frontend, "
            "generation, codec, watermark and transfers; excludes WAV writes/hashing. "
            "Clip generation_seconds denotes completion latency of its whole batch; "
            "aggregate RTF divides batch E2E by sum audio duration. Each batch uses shared "
            "seed42; batching can change stochastic samples and is not exact greedy parity."
            " Vendor max_new_tokens=2000 takes precedence over SDK max_length=2048; "
            "the actual per-request token cap is recorded, with EOS retained in evidence."
        ),
    )
    write_record(directory / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    arguments = parser.parse_args()
    pilot = NeuTtsPilotConfig(
        manifest=arguments.manifest,
        manifest_sha256=digest(arguments.manifest),
        output_directory=arguments.output,
        repository_directory=arguments.repository,
        preparation=NeuTtsPreparation.model_validate_json(arguments.preparation.read_bytes()),
        device=PilotDevice.CUDA,
    )
    result = benchmark(NeuTtsBenchmarkConfig(pilot=pilot, source_commit=arguments.source_commit))
    print(result.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
