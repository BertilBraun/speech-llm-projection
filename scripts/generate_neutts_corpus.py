"""Resume only missing paired NeuTTS audio using genuine native batch generation."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import torch
from neutts import NeuTTS2E

from scripts.benchmark_neutts_batch import persist_batch, synthesize_batch
from scripts.neutts_pilot_state import NeuTtsPilotConfig, PilotDevice, digest, write_record
from scripts.prepare_neutts_models import NeuTtsPreparation
from speech_projector.journal import append_record, read_journal
from speech_projector.neutts_corpus import (
    NeuCorpusConfig,
    NeuCorpusFailure,
    NeuCorpusResult,
    NeuCorpusSession,
    NeuCorpusStart,
    persist_completed,
    restore_evidence,
    validate_paired_manifest,
    verify_audio,
)
from speech_projector.tts_pilot import PilotTermination, TtsPilotCase, load_pilot_manifest


def run(configuration: NeuCorpusConfig, limit: int) -> NeuCorpusResult:
    pilot = configuration.pilot
    directory = pilot.output_directory
    directory.mkdir(parents=True, exist_ok=True)
    saved = directory / "config.json"
    if saved.exists():
        if NeuCorpusConfig.model_validate_json(saved.read_bytes()) != configuration:
            raise ValueError("Refusing changed corpus configuration on resume")
    else:
        write_record(saved, configuration)
    if digest(pilot.manifest) != pilot.manifest_sha256:
        raise ValueError("Corpus manifest differs from saved hash")
    manifest = load_pilot_manifest(pilot.manifest)
    validate_paired_manifest(manifest)
    result_path = directory / "result.json"
    if result_path.exists():
        previous = NeuCorpusResult.model_validate_json(result_path.read_bytes())
        if not previous.missing_cases:
            verify_audio(manifest.cases, directory, previous.speech_end_token_id)
            return previous
    write_record(directory / "cases.json", manifest)
    restored = tuple(
        case for case in manifest.cases if restore_evidence(case, directory) is not None
    )
    completed = {case.case_id for case in restored}
    pending = [case for case in manifest.cases if case.case_id not in completed]
    if limit:
        pending = pending[:limit]
    pending.sort(key=lambda case: (len(case.text), case.utterance_id, case.case_id))
    sessions = read_journal(directory / "sessions.jsonl", NeuCorpusSession)
    starts_directory = directory / "session_starts"
    starts_directory.mkdir(exist_ok=True)
    session_index = len(tuple(starts_directory.glob("*.json")))
    write_record(
        starts_directory / f"{session_index:06d}.json",
        NeuCorpusStart(
            session_index=session_index,
            started_at=datetime.now(timezone.utc),
            resumed_clips=len(restored),
            pending_clips=len(pending),
            helper_sha256=digest(Path(__file__)),
        ),
    )
    started = perf_counter()
    torch.set_num_threads(pilot.cpu_threads)
    backbone, codec = pilot.preparation.repositories
    if backbone.repository != "neuphonic/neutts-2e" or codec.repository != "neuphonic/neucodec":
        raise ValueError("Corpus requires the authorized NeuTTS-2E and standard NeuCodec")
    for repository in pilot.preparation.repositories:
        if (
            repository.snapshot.parent.parent / "refs/main"
        ).read_text().strip() != repository.revision:
            raise ValueError("Offline model reference differs from pinned corpus configuration")
    startup = perf_counter()
    model = NeuTTS2E(
        backbone_repo=str(backbone.snapshot),
        backbone_device="cuda",
        codec_repo=codec.repository,
        codec_device="cuda",
        seed=manifest.warmup_seed,
    )
    torch.cuda.synchronize()
    startup_seconds = perf_counter() - startup
    if model._is_quantized_model or model._is_onnx_codec:
        raise ValueError("Corpus uses the benchmarked PyTorch backbone and standard codec")
    reference_started = perf_counter()
    reference_codes, reference_text = model._speaker(pilot.speaker)
    torch.cuda.synchronize()
    reference_seconds = perf_counter() - reference_started
    warmup_seconds = 0.0
    if pending:
        warmed = synthesize_batch(model, pilot, tuple(pending[:4]), reference_codes, reference_text)
        warmup_seconds = warmed.end_to_end_seconds
    torch.cuda.reset_peak_memory_stats()
    end_token: int = model.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
    batch_index = session_index * 100000
    measured_seconds = 0.0
    generated_seconds = 0.0
    generated_count = 0
    for attempt in range(configuration.attempts):
        remaining: list[TtsPilotCase] = []
        for offset in range(0, len(pending), configuration.batch_size):
            canonical_cases = tuple(pending[offset : offset + configuration.batch_size])
            cases = tuple(
                case.model_copy(update={"seed": case.seed + attempt}) for case in canonical_cases
            )
            try:
                generated = synthesize_batch(model, pilot, cases, reference_codes, reference_text)
                measurement = persist_batch(
                    pilot,
                    cases,
                    generated,
                    configuration.batch_size,
                    attempt,
                    batch_index,
                    end_token,
                    model.sample_rate,
                )
                persist_completed(measurement, directory)
            except (RuntimeError, ValueError) as error:
                append_record(
                    directory / "failures.jsonl",
                    NeuCorpusFailure(
                        cases=cases,
                        attempt=attempt,
                        error=f"{type(error).__name__}: {error}",
                    ),
                )
                remaining.extend(canonical_cases)
                torch.cuda.empty_cache()
                batch_index += 1
                continue
            batch_index += 1
            measured_seconds += measurement.end_to_end_seconds
            generated_seconds += measurement.audio_seconds
            for case, item in zip(canonical_cases, measurement.clips, strict=True):
                if item.clip.termination == PilotTermination.STOP:
                    generated_count += 1
                else:
                    remaining.append(case)
            print(
                f"session={session_index} attempt={attempt}"
                f" completed={len(restored) + generated_count}"
                f"/{len(manifest.cases)} batch_seconds={measurement.end_to_end_seconds:.3f}"
                f" measured_rtf={measured_seconds / generated_seconds:.5f}"
                f" wall_seconds={perf_counter() - started:.3f}",
                flush=True,
            )
        pending = remaining
        if not pending:
            break
    session = NeuCorpusSession(
        session_index=session_index,
        resumed_clips=len(restored),
        generated_clips=generated_count,
        failed_cases=tuple(case.case_id for case in pending),
        startup_seconds=startup_seconds,
        reference_seconds=reference_seconds,
        warmup_seconds=warmup_seconds,
        wall_seconds=perf_counter() - started,
        measured_batch_seconds=measured_seconds,
        generated_audio_seconds=generated_seconds,
        peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
    )
    append_record(directory / "sessions.jsonl", session)
    verification = verify_audio(manifest.cases, directory, end_token)
    if model.backbone.generation_config.max_new_tokens is None:
        raise ValueError("NeuTTS generation token cap is missing")
    result = NeuCorpusResult(
        configuration=configuration,
        expected_cases=len(manifest.cases),
        verified_clips=verification.verified_clips,
        complete_pairs=verification.complete_pairs,
        missing_cases=verification.missing_cases,
        total_audio_seconds=verification.total_audio_seconds,
        audio_bytes=verification.audio_bytes,
        over_30_second_cases=verification.over_30_second_cases,
        peak_amplitude=verification.peak_amplitude,
        overshoot_samples=verification.overshoot_samples,
        waveform_samples=verification.waveform_samples,
        termination_evidence=(
            "Every accepted waveform retains the actual speech-generation-end token ID."
        ),
        torch_version=torch.__version__,
        transformers_version=version("transformers"),
        neucodec_version=version("neucodec"),
        helper_sha256=digest(Path(__file__)),
        reference_sha256=digest(pilot.repository_directory / "samples" / f"{pilot.speaker}.wav"),
        watermark_active=model.watermarker is not None,
        generation_token_cap=model.backbone.generation_config.max_new_tokens,
        speech_end_token_id=end_token,
        sessions=tuple(sessions) + (session,),
        interrupted_sessions=tuple(
            index
            for index in range(session_index)
            if index not in {entry.session_index for entry in sessions}
        ),
    )
    write_record(directory / "result.json", result)
    if pending:
        raise ValueError(f"Corpus has {len(pending)} failed cases after saved retries")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--limit", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.limit < 0:
        raise ValueError("Pilot limit must be nonnegative")
    configuration = NeuCorpusConfig(
        pilot=NeuTtsPilotConfig(
            manifest=arguments.manifest,
            manifest_sha256=digest(arguments.manifest),
            output_directory=arguments.output,
            repository_directory=arguments.repository,
            preparation=NeuTtsPreparation.model_validate_json(arguments.preparation.read_bytes()),
            device=PilotDevice.CUDA,
        ),
        source_commit=arguments.source_commit,
    )
    result = run(configuration, arguments.limit)
    print(
        f"Verified clips={result.verified_clips}/{result.expected_cases};"
        f" pairs={result.complete_pairs};"
        f" audio_hours={result.total_audio_seconds / 3600:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
