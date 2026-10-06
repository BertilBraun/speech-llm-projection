"""Generate only the fixed ten CustomVoice previews in the isolated Qwen-TTS environment."""

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

from scripts.inventory_results import write_record
from scripts.package_results import ModelRevision
from speech_projector.emotion_preview import (
    CappedSynthesis,
    PreviewAudioFailure,
    PreviewCapFailure,
    PreviewCase,
    PreviewClip,
    PreviewManifest,
    PreviewPlan,
    PreviewProvenance,
    PreviewSynthesisFailure,
    PreviewTermination,
    SynthesizedAudio,
    codec_termination,
    default_preview_plan,
    persist_clip,
    plan_digest,
    render_preview,
)


@torch.inference_mode()
def synthesize(
    model: Qwen3TTSModel, plan: PreviewPlan, case: PreviewCase
) -> SynthesizedAudio | CappedSynthesis:
    input_ids: list[Tensor] = model._tokenize_texts([model._build_assistant_text(case.text)])
    instruct_ids: list[Tensor] = model._tokenize_texts([model._build_instruct_text(case.instruct)])
    torch.cuda.synchronize()
    started = perf_counter()
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(case.seed)
        torch.cuda.manual_seed_all(case.seed)
        codec_output: tuple[list[Tensor], list[Tensor]] = model.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=[plan.language],
            speakers=[plan.speaker],
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
    assert len(codes) == 1
    codec_tokens = codes[0].shape[0]
    if codec_termination(codec_tokens, plan.max_new_tokens) == PreviewTermination.CAP_AMBIGUOUS:
        torch.cuda.synchronize()
        return CappedSynthesis(
            codec_tokens=codec_tokens,
            runtime_seconds=perf_counter() - started,
        )
    decoded: tuple[list[NDArray[np.float32]], int] = model.model.speech_tokenizer.decode(
        [{"audio_codes": codes[0]}]
    )
    waveforms, sample_rate = decoded
    assert len(waveforms) == 1
    waveform: NDArray[np.float32] = waveforms[0]
    torch.cuda.synchronize()
    return SynthesizedAudio(
        waveform=waveform,
        sample_rate=sample_rate,
        codec_tokens=codec_tokens,
        runtime_seconds=perf_counter() - started,
    )


def generate_preview(output: Path, plan: PreviewPlan, source_commit: str) -> PreviewManifest:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preview output must be new or empty; preserve prior attempts separately")
    output.mkdir(parents=True, exist_ok=True)
    write_record(output / "plan.json", plan)
    snapshot = Path(
        snapshot_download(plan.model_name, revision=plan.revision, local_files_only=True)
    )
    started_at = datetime.now(timezone.utc)
    provenance = PreviewProvenance(
        source_commit=source_commit,
        model_revision=ModelRevision(
            model_name=plan.model_name,
            snapshot_revisions=(snapshot.name,),
            main_revision=snapshot.name,
        ),
        input_sha256=plan_digest(plan),
        started_at=started_at,
        qwen_tts_version=version("qwen-tts"),
        torch_version=version("torch"),
        transformers_version=version("transformers"),
    )
    write_record(output / "provenance.json", provenance)
    started = perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = Qwen3TTSModel.from_pretrained(
        str(snapshot), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    clips: list[PreviewClip] = []
    for case in plan.cases:
        case_started = perf_counter()
        try:
            outcome = synthesize(model, plan, case)
        except RuntimeError as error:
            write_record(
                output / "failure.json",
                PreviewSynthesisFailure(
                    case=case, error=str(error), runtime_seconds=perf_counter() - case_started
                ),
            )
            raise
        match outcome:
            case CappedSynthesis():
                write_record(
                    output / "failure.json",
                    PreviewCapFailure(
                        case=case,
                        codec_tokens=outcome.codec_tokens,
                        max_new_tokens=plan.max_new_tokens,
                        runtime_seconds=outcome.runtime_seconds,
                    ),
                )
                raise ValueError(f"Preview {case.case_id} reached an ambiguous codec cap")
            case SynthesizedAudio():
                synthesis = outcome
        try:
            clip = persist_clip(output, plan, case, synthesis)
        except ValueError as error:
            write_record(
                output / "failure.json",
                PreviewAudioFailure(
                    case=case, error=str(error), runtime_seconds=synthesis.runtime_seconds
                ),
            )
            raise
        clips.append(clip)
        print(f"{case.case_id}: {clip.duration_seconds:.3f}s / {clip.codec_tokens} codec frames")
    manifest = PreviewManifest(
        plan=plan,
        clips=tuple(clips),
        provenance=provenance,
        runtime_seconds=perf_counter() - started,
        peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
    )
    write_record(output / "manifest.json", manifest)
    (output / "index.md").write_text(render_preview(manifest), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-commit", required=True)
    arguments = parser.parse_args()
    generate_preview(
        arguments.output,
        default_preview_plan(arguments.max_new_tokens, revision=arguments.revision),
        arguments.source_commit,
    )


if __name__ == "__main__":
    main()
