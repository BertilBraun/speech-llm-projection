"""Separate compile time from warm batching and quantify padding/roundoff effects."""

import argparse
import subprocess
import time
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as functional
from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available

from speech_projector.data import load_examples
from speech_projector.generation import GenerationResult
from speech_projector.inputs import TranscriptInput
from speech_projector.llm import FrozenQwen, GenerationBatch
from speech_projector.models import Example, Record, Split
from speech_projector.teacher_configuration import teacher_compression_runs


class LogitComparison(Record):
    example_id: str
    maximum_absolute_error: float
    mean_absolute_error: float
    reference_top_two_margin: float
    batch_top_two_margin: float
    first_token_matches: bool


class WarmBatchProfile(Record):
    batch_size: int
    cold_seconds: float
    warm_seconds: float
    warm_examples_per_second: float
    warm_generated_tokens_per_second: float
    peak_vram_gb: float
    responses: tuple[GenerationResult, ...]


class WarmTeacherProfile(Record):
    source_commit: str
    examples: tuple[Example, ...]
    profiles: tuple[WarmBatchProfile, ...]
    unpadded_to_padded_single: tuple[LogitComparison, ...]
    unpadded_to_batched: tuple[LogitComparison, ...]


def next_logits(wrapper: FrozenQwen, batch: GenerationBatch) -> Tensor:
    positions = (batch.attention_mask.cumsum(-1) - 1).clamp_min(0)
    return (
        wrapper.model(
            inputs_embeds=batch.embeddings,
            attention_mask=batch.attention_mask,
            position_ids=positions,
            logits_to_keep=1,
            use_cache=False,
        )
        .logits[:, 0]
        .float()
    )


def compare_logits(example: Example, reference: Tensor, candidate: Tensor) -> LogitComparison:
    reference_top = reference.topk(2)
    candidate_top = candidate.topk(2)
    errors = (reference - candidate).abs()
    return LogitComparison(
        example_id=example.example_id,
        maximum_absolute_error=float(errors.max()),
        mean_absolute_error=float(errors.mean()),
        reference_top_two_margin=float(reference_top.values[0] - reference_top.values[1]),
        batch_top_two_margin=float(candidate_top.values[0] - candidate_top.values[1]),
        first_token_matches=int(reference_top.indices[0]) == int(candidate_top.indices[0]),
    )


@torch.no_grad()
def profile(manifest: Path, output: Path) -> WarmTeacherProfile:
    assert is_fast_path_available
    config = teacher_compression_runs()[0]
    examples = tuple(load_examples(manifest, Split.VALIDATION, 64))
    if len(examples) != 64:
        raise ValueError("Warm teacher profile requires64 held-out examples")
    wrapper = FrozenQwen(config, torch.device("cuda"))
    utterances = tuple(TranscriptInput(example.user_text) for example in examples)
    batched = next_logits(wrapper, wrapper.prepare_generation_batch(examples, utterances))
    padded_comparisons: list[LogitComparison] = []
    batch_comparisons: list[LogitComparison] = []
    for index, example in enumerate(examples[:16]):
        prompt = wrapper._prompt(example, utterances[index]).unsqueeze(0)
        original = GenerationBatch(
            prompt, torch.ones(prompt.shape[:2], device=wrapper.device, dtype=torch.long)
        )
        reference = next_logits(wrapper, original)[0]
        padded = next_logits(
            wrapper, wrapper.prepare_generation_batch((example,), (utterances[index],))
        )[0]
        padded_comparisons.append(compare_logits(example, reference, padded))
        batch_comparisons.append(compare_logits(example, reference, batched[index]))
    profiles: list[WarmBatchProfile] = []
    for size in (16, 32, 64):
        selected = examples[:size]
        inputs = utterances[:size]
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        wrapper.generate_batch(selected, inputs, 128)
        torch.cuda.synchronize()
        cold = time.perf_counter() - started
        started = time.perf_counter()
        responses = wrapper.generate_batch(selected, inputs, 128)
        torch.cuda.synchronize()
        warm = time.perf_counter() - started
        result = WarmBatchProfile(
            batch_size=size,
            cold_seconds=cold,
            warm_seconds=warm,
            warm_examples_per_second=size / warm,
            warm_generated_tokens_per_second=sum(len(response.token_ids) for response in responses)
            / warm,
            peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
            responses=responses,
        )
        profiles.append(result)
        print(result.model_dump_json(), flush=True)
    report = WarmTeacherProfile(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        examples=examples,
        profiles=tuple(profiles),
        unpadded_to_padded_single=tuple(padded_comparisons),
        unpadded_to_batched=tuple(batch_comparisons),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    profile(arguments.manifest, arguments.output)


if __name__ == "__main__":
    main()
