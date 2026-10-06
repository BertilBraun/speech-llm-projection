"""Batch and resume the Qwen CustomVoice emotional audio dataset."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from huggingface_hub import snapshot_download
from numpy.typing import NDArray
from qwen_tts import Qwen3TTSModel
from torch import Tensor

from scripts.generate_emotion_preview import synthesize
from scripts.inventory_results import write_record
from scripts.package_results import ModelRevision
from speech_projector.emotion_preview import (
    CappedSynthesis,
    PreviewCase,
    PreviewClip,
    PreviewPlan,
    PreviewProvenance,
    PreviewSynthesisFailure,
    PreviewTermination,
    SynthesizedAudio,
    codec_termination,
    persist_clip,
    plan_digest,
)
from speech_projector.emotional_audio import (
    AudioBatchTiming,
    AudioGenerationSummary,
    EmotionalAudioConfig,
    completed_audio,
)


@torch.inference_mode()
def synthesize_batch(
    model: Qwen3TTSModel,
    plan: PreviewPlan,
    cases: tuple[PreviewCase, ...],
    seed: int,
) -> tuple[SynthesizedAudio | CappedSynthesis, ...]:
    input_ids: list[Tensor] = model._tokenize_texts(
        [model._build_assistant_text(case.text) for case in cases]
    )
    instruct_ids: list[Tensor] = model._tokenize_texts(
        [model._build_instruct_text(case.instruct) for case in cases]
    )
    torch.cuda.synchronize()
    started = perf_counter()
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        codec_output: tuple[list[Tensor], list[Tensor]] = model.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=[plan.language] * len(cases),
            speakers=[plan.speaker] * len(cases),
            non_streaming_mode=True,
            max_new_tokens=plan.max_new_tokens,
            do_sample=True,
            top_k=plan.top_k,
            top_p=plan.top_p,
            temperature=plan.temperature,
            repetition_penalty=plan.repetition_penalty,
            subtalker_dosample=True,
            subtalker_top_k=plan.top_k,
            subtalker_top_p=plan.top_p,
            subtalker_temperature=plan.temperature,
        )
    codes, _ = codec_output
    assert len(codes) == len(cases)
    outcomes: list[SynthesizedAudio | CappedSynthesis] = []
    for code in codes:
        codec_tokens = code.shape[0]
        if codec_termination(codec_tokens, plan.max_new_tokens) == PreviewTermination.CAP_AMBIGUOUS:
            outcomes.append(CappedSynthesis(codec_tokens=codec_tokens, runtime_seconds=0.0))
            continue
        decoded: tuple[list[NDArray[np.float32]], int] = model.model.speech_tokenizer.decode(
            [{"audio_codes": code}]
        )
        waveforms, sample_rate = decoded
        assert len(waveforms) == 1
        outcomes.append(
            SynthesizedAudio(
                waveform=waveforms[0],
                sample_rate=sample_rate,
                codec_tokens=codec_tokens,
                runtime_seconds=0.0,
            )
        )
    torch.cuda.synchronize()
    elapsed_per_case = (perf_counter() - started) / len(cases)
    return tuple(timed_outcome(item, elapsed_per_case) for item in outcomes)


def timed_outcome(
    outcome: SynthesizedAudio | CappedSynthesis, runtime_seconds: float
) -> SynthesizedAudio | CappedSynthesis:
    match outcome:
        case CappedSynthesis():
            return CappedSynthesis(
                codec_tokens=outcome.codec_tokens, runtime_seconds=runtime_seconds
            )
        case SynthesizedAudio():
            return SynthesizedAudio(
                waveform=outcome.waveform,
                sample_rate=outcome.sample_rate,
                codec_tokens=outcome.codec_tokens,
                runtime_seconds=runtime_seconds,
            )


def store_outcome(
    model: Qwen3TTSModel,
    config: EmotionalAudioConfig,
    case: PreviewCase,
    outcome: SynthesizedAudio | CappedSynthesis,
) -> PreviewClip:
    plan = config.plan
    match outcome:
        case CappedSynthesis():
            plan = plan.model_copy(update={"max_new_tokens": config.retry_max_new_tokens})
            outcome = synthesize(model, plan, case)
        case SynthesizedAudio():
            pass
    match outcome:
        case CappedSynthesis():
            raise ValueError(f"Audio {case.case_id} reached the retry codec cap")
        case SynthesizedAudio():
            return persist_clip(config.output, plan, case, outcome)


def generate_audio(config: EmotionalAudioConfig) -> AudioGenerationSummary:
    config.output.mkdir(parents=True, exist_ok=True)
    configuration_path = config.output / "config.json"
    if configuration_path.exists():
        saved = EmotionalAudioConfig.model_validate_json(configuration_path.read_bytes())
        if saved != config:
            raise ValueError("Audio resume configuration differs from the original run")
    else:
        write_record(configuration_path, config)
        write_record(config.output / "plan.json", config.plan)
    existing = completed_audio(config)
    existing_ids = {clip.case.case_id for clip in existing}
    pending = tuple(case for case in config.plan.cases if case.case_id not in existing_ids)
    if not pending:
        summary = AudioGenerationSummary(
            planned=len(config.plan.cases),
            completed=len(existing),
            failed_case_ids=(),
            session_runtime_seconds=0.0,
            session_audio_seconds=0.0,
            session_completed=0,
            peak_vram_gb=0.0,
        )
        write_record(config.output / "summary.json", summary)
        return summary
    snapshot = Path(
        snapshot_download(
            config.plan.model_name, revision=config.plan.revision, local_files_only=True
        )
    )
    provenance = PreviewProvenance(
        source_commit=config.source_commit,
        model_revision=ModelRevision(
            model_name=config.plan.model_name,
            snapshot_revisions=(snapshot.name,),
            main_revision=snapshot.name,
        ),
        input_sha256=plan_digest(config.plan),
        started_at=datetime.now(timezone.utc),
        qwen_tts_version=version("qwen-tts"),
        torch_version=version("torch"),
        transformers_version=version("transformers"),
    )
    if not (config.output / "provenance.json").exists():
        write_record(config.output / "provenance.json", provenance)
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        str(snapshot), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    completed: list[PreviewClip] = []
    failed: list[str] = []
    for offset in range(0, len(pending), config.batch_size):
        cases = pending[offset : offset + config.batch_size]
        batch_started = perf_counter()
        seed = cases[0].seed
        try:
            outcomes = synthesize_batch(model, config.plan, cases, seed)
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            torch.cuda.empty_cache()
            outcomes = tuple(synthesize(model, config.plan, case) for case in cases)
        batch_completed = 0
        for case, outcome in zip(cases, outcomes, strict=True):
            try:
                clip = store_outcome(model, config, case, outcome)
            except (ValueError, RuntimeError) as error:
                write_record(
                    config.output / "failures" / f"{case.case_id}.json",
                    PreviewSynthesisFailure(
                        case=case,
                        error=str(error),
                        runtime_seconds=perf_counter() - batch_started,
                    ),
                )
                failed.append(case.case_id)
                print(f"FAILED {case.case_id}: {error}", flush=True)
                continue
            completed.append(clip)
            batch_completed += 1
        timing = AudioBatchTiming(
            case_ids=tuple(case.case_id for case in cases),
            seed=seed,
            runtime_seconds=perf_counter() - batch_started,
            peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
            completed=batch_completed,
        )
        with (config.output / "batches.jsonl").open("a", encoding="utf-8") as journal:
            journal.write(timing.model_dump_json() + "\n")
        summary = AudioGenerationSummary(
            planned=len(config.plan.cases),
            completed=len(existing) + len(completed),
            failed_case_ids=tuple(failed),
            session_runtime_seconds=perf_counter() - started,
            session_audio_seconds=sum(clip.duration_seconds for clip in completed),
            session_completed=len(completed),
            peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
        )
        write_record(config.output / "summary.json", summary)
        print(
            f"audio {summary.completed}/{summary.planned}; "
            f"batch {timing.runtime_seconds:.2f}s; failed {len(failed)}",
            flush=True,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    summary = generate_audio(
        EmotionalAudioConfig.model_validate_json(arguments.config.read_bytes())
    )
    if summary.failed_case_ids:
        raise ValueError(f"{len(summary.failed_case_ids)} audio cases require retry")


if __name__ == "__main__":
    main()
