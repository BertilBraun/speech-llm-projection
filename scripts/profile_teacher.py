"""Bounded teacher quality, batching parity, throughput and GPU-memory profile."""

import argparse
import subprocess
import time
from pathlib import Path

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available

from scripts.package_results import FileArtifact, file_digest
from speech_projector.data import load_examples
from speech_projector.generation import GenerationResult
from speech_projector.inputs import TranscriptInput
from speech_projector.llm import FrozenQwen
from speech_projector.models import Example, Record, RunConfig, Split
from speech_projector.teacher_configuration import teacher_compression_runs


class BatchProfile(Record):
    batch_size: int
    elapsed_seconds: float
    examples_per_second: float
    generated_tokens_per_second: float
    peak_vram_gb: float
    exact_single_token_matches: tuple[bool, ...]
    responses: tuple[GenerationResult, ...]


class TeacherProfile(Record):
    config: RunConfig
    source_commit: str
    source_files: tuple[FileArtifact, ...]
    examples: tuple[Example, ...]
    profiles: tuple[BatchProfile, ...]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    assert is_fast_path_available
    config = teacher_compression_runs()[0]
    examples = tuple(load_examples(arguments.manifest, Split.VALIDATION, 16))
    if len(examples) != 16:
        raise ValueError("Teacher profile needs exactly16 held-out examples")
    torch.manual_seed(config.seed)
    wrapper = FrozenQwen(config, torch.device("cuda"))
    wrapper.generate_batch((examples[0],), (TranscriptInput(examples[0].user_text),), 8)
    profiles: list[BatchProfile] = []
    reference: tuple[GenerationResult, ...] = ()
    for size in (1, 4, 8, 16):
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        responses: list[GenerationResult] = []
        for offset in range(0, len(examples), size):
            selected = examples[offset : offset + size]
            responses.extend(
                wrapper.generate_batch(
                    selected,
                    tuple(TranscriptInput(example.user_text) for example in selected),
                    128,
                )
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if size == 1:
            reference = tuple(responses)
        profile = BatchProfile(
            batch_size=size,
            elapsed_seconds=elapsed,
            examples_per_second=len(examples) / elapsed,
            generated_tokens_per_second=sum(len(response.token_ids) for response in responses)
            / elapsed,
            peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
            exact_single_token_matches=tuple(
                response.token_ids == original.token_ids
                for response, original in zip(responses, reference, strict=True)
            ),
            responses=tuple(responses),
        )
        profiles.append(profile)
        print(profile.model_dump_json(), flush=True)
    source_files = tuple(
        FileArtifact(
            path=path, source_path=path, bytes=path.stat().st_size, sha256=file_digest(path)
        )
        for path in (
            Path("speech_projector/llm.py"),
            Path("speech_projector/models.py"),
            Path("speech_projector/prompts.py"),
            Path("speech_projector/generation.py"),
            Path("scripts/profile_teacher.py"),
        )
    )
    result = TeacherProfile(
        config=config,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_files=source_files,
        examples=examples,
        profiles=tuple(profiles),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
