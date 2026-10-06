"""Run eight serial, synchronized IndexTTS-2.5 listening comparisons."""

import argparse
import random
import warnings
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile
import torch
from indextts.infer_v2_5 import IndexTTS2
from numpy.typing import NDArray

from speech_projector.index_tts_pilot import (
    IndexAttemptFailure,
    IndexEmotionControl,
    IndexPilotConfig,
    IndexSession,
    completed_clip,
    digest,
    emotion_vector,
    prepare_output,
    write_record,
)
from speech_projector.tts_pilot import (
    PilotEmotion,
    PilotTermination,
    TtsPilotClip,
    TtsPilotFailure,
    TtsPilotResult,
    load_pilot_manifest,
)


@dataclass(frozen=True)
class IndexAudio:
    waveform: NDArray[np.int16]
    sample_rate: int
    generation_seconds: float
    warning_messages: tuple[str, ...]


@torch.inference_mode()
def synthesize(
    model: IndexTTS2,
    configuration: IndexPilotConfig,
    text: str,
    emotion: PilotEmotion,
    seed: int,
) -> IndexAudio:
    torch.cuda.synchronize()
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            result: tuple[int, NDArray[np.int16]] | None = model.infer(
                spk_audio_prompt=str(configuration.reference_audio),
                text=text,
                output_path=None,
                lang="EN",
                emo_vector=list(emotion_vector(emotion, configuration.emotion_intensity)),
                emo_alpha=1.0,
                use_random=False,
                use_emo_text=False,
                stream_return=False,
                max_text_tokens_per_segment=120,
                duration_factor=1.0,
                text_normalization=True,
                do_sample=True,
                top_p=configuration.top_p,
                top_k=configuration.top_k,
                temperature=configuration.temperature,
                num_beams=configuration.num_beams,
                repetition_penalty=configuration.repetition_penalty,
                max_mel_tokens=configuration.max_mel_tokens,
            )
        torch.cuda.synchronize()
        elapsed = perf_counter() - started
    if result is None:
        raise ValueError("IndexTTS returned no waveform")
    sample_rate, waveform = result
    if waveform.dtype != np.int16 or waveform.ndim != 2 or waveform.shape[1] != 1:
        raise ValueError("IndexTTS returned unexpected PCM16 waveform layout")
    if waveform.shape[0] == 0 or not np.any(waveform):
        raise ValueError("IndexTTS returned empty or silent waveform")
    return IndexAudio(
        waveform=np.ascontiguousarray(waveform[:, 0]),
        sample_rate=sample_rate,
        generation_seconds=elapsed,
        warning_messages=tuple(str(item.message) for item in caught),
    )


def save_audio(path: Path, audio: IndexAudio) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".temporary.wav")
    soundfile.write(temporary, audio.waveform, audio.sample_rate, subtype="PCM_16")
    temporary.replace(path)


def has_cap_warning(audio: IndexAudio) -> bool:
    return any("exceeding `max_mel_tokens`" in message for message in audio.warning_messages)


def generate(configuration: IndexPilotConfig) -> TtsPilotResult:
    manifest = load_pilot_manifest(configuration.manifest_path)
    prepare_output(configuration, manifest)
    output = configuration.output_directory
    clips: list[TtsPilotClip] = []
    failures: list[TtsPilotFailure] = []
    if (output / "result.json").exists():
        completed = TtsPilotResult.model_validate_json((output / "result.json").read_bytes())
        if not completed.failures and len(completed.clips) == len(manifest.cases):
            verified = tuple(completed_clip(configuration, case) for case in manifest.cases)
            if completed.clips != verified:
                raise ValueError("Saved IndexTTS result differs from verified clip records")
            if (
                completed.model_revision != configuration.model_revision
                or completed.source_commit != configuration.source_commit
            ):
                raise ValueError("Saved IndexTTS result provenance differs from configuration")
            return completed
    torch.cuda.synchronize()
    started = perf_counter()
    model = IndexTTS2(
        cfg_path=str(configuration.checkpoint_directory / "config.yaml"),
        model_dir=str(configuration.checkpoint_directory),
        use_bf16=True,
        device="cuda:0",
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_accel=False,
        use_torch_compile=False,
        use_qwen_emo=False,
    )
    torch.cuda.synchronize()
    startup_seconds = perf_counter() - started
    warmup = synthesize(
        model, configuration, manifest.warmup_text, PilotEmotion.NEUTRAL, manifest.warmup_seed
    )
    save_audio(output / "warmup.wav", warmup)
    if has_cap_warning(warmup):
        raise ValueError("IndexTTS warmup hit max_mel_tokens; saved warmup WAV retained")
    session = IndexSession(
        startup_seconds=startup_seconds,
        warmup_generation_seconds=warmup.generation_seconds,
        torch_version=torch.__version__,
        transformers_version=version("transformers"),
        reference_sha256=digest(configuration.reference_audio),
        manifest_sha256=digest(configuration.manifest_path),
        warmup_warnings=warmup.warning_messages,
    )
    session_directory = output / "sessions"
    session_directory.mkdir(exist_ok=True)
    write_record(
        session_directory / f"{len(tuple(session_directory.glob('*.json'))):03d}.json", session
    )
    for case in manifest.cases:
        saved = completed_clip(configuration, case)
        if saved is not None:
            clips.append(saved)
            continue
        attempt_directory = output / "attempts" / case.case_id
        attempt_directory.mkdir(parents=True, exist_ok=True)
        attempt = len(tuple(attempt_directory.glob("*.json")))
        started = perf_counter()
        try:
            audio = synthesize(model, configuration, case.text, case.emotion, case.seed)
            if has_cap_warning(audio):
                failed_path = attempt_directory / f"{attempt:03d}.wav"
                save_audio(failed_path, audio)
                evidence = IndexAttemptFailure(
                    case=case,
                    error="IndexTTS emitted max_mel_tokens cap warning",
                    generation_seconds=audio.generation_seconds,
                    warnings=audio.warning_messages,
                    preserved_audio=(failed_path.relative_to(output).as_posix(),),
                )
                write_record(attempt_directory / f"{attempt:03d}.json", evidence)
                failures.append(TtsPilotFailure(case=case, error=evidence.error))
                continue
            audio_path = output / "audio" / f"{case.case_id}.wav"
            save_audio(audio_path, audio)
            duration = audio.waveform.shape[0] / audio.sample_rate
            clip = TtsPilotClip(
                case=case,
                audio_path=audio_path.relative_to(output).as_posix(),
                sha256=digest(audio_path),
                sample_rate=audio.sample_rate,
                audio_seconds=duration,
                generation_seconds=audio.generation_seconds,
                real_time_factor=audio.generation_seconds / duration,
                termination=PilotTermination.NOT_EXPOSED,
            )
            write_record(output / "clips" / f"{case.case_id}.json", clip)
            write_record(
                output / "emotion_controls" / f"{case.case_id}.json",
                IndexEmotionControl(
                    case=case,
                    vector=emotion_vector(case.emotion, configuration.emotion_intensity),
                    effective_vector=tuple(
                        model.normalize_emo_vec(
                            list(emotion_vector(case.emotion, configuration.emotion_intensity))
                        )
                    ),
                ),
            )
            clips.append(clip)
            print(clip.model_dump_json(), flush=True)
        except (ValueError, RuntimeError, OSError) as error:
            evidence = IndexAttemptFailure(
                case=case,
                error=f"{type(error).__name__}: {error}",
                generation_seconds=perf_counter() - started,
                warnings=(),
                preserved_audio=(),
            )
            write_record(attempt_directory / f"{attempt:03d}.json", evidence)
            failures.append(TtsPilotFailure(case=case, error=evidence.error))
    result = TtsPilotResult(
        model_repository="IndexTeam/IndexTTS-2.5",
        model_revision=configuration.model_revision,
        source_commit=configuration.source_commit,
        backend_description=(
            "Official IndexTTS-2.5 BF16, serial CUDA; no DeepSpeed/compile/custom kernel. "
            "GPU-synchronized generation includes text frontend, codec and waveform decoding; "
            "excludes WAV write. Reference conditioning is included in neutral warmup, "
            "not independently timed. SDK exposes cap warnings but not normal-stop token IDs."
        ),
        speaker_description=f"Unaltered Paul reference, SHA256 {configuration.reference_sha256}",
        startup_seconds=startup_seconds,
        reference_preparation_seconds=0,
        warmup_generation_seconds=warmup.generation_seconds,
        clips=tuple(clips),
        failures=tuple(failures),
    )
    write_record(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = IndexPilotConfig.model_validate_json(arguments.config.read_bytes())
    result = generate(configuration)
    print(result.model_dump_json(indent=2), flush=True)
    if result.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
