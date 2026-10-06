"""Check sampled EOS completion on real examples that defeated greedy decoding."""

import argparse
import subprocess
import time
from pathlib import Path

import torch

from scripts.package_results import FileArtifact, file_digest
from speech_projector.generation import CompletedGeneration, GenerationResult
from speech_projector.inputs import TranscriptInput
from speech_projector.llm import FrozenQwen
from speech_projector.models import Example, Record, RunConfig, SamplingDecodingConfig


class SamplingCompletionCheck(Record):
    config: RunConfig
    source_commit: str
    source_files: tuple[FileArtifact, ...]
    examples: tuple[Example, ...]
    token_cap: int
    responses: tuple[GenerationResult, ...]
    repeated_responses: tuple[GenerationResult, ...]
    exact_repeat: bool
    generation_seconds: float
    peak_vram_gb: float


def run_check(
    manifest: Path, run_config: Path, identifiers: tuple[str, ...], output: Path, token_cap: int
) -> SamplingCompletionCheck:
    config = RunConfig.model_validate_json(run_config.read_bytes())
    match config.decoding:
        case SamplingDecodingConfig():
            pass
        case _:
            raise ValueError("This completion check requires the explicit sampling configuration")
    source_examples = tuple(
        Example.model_validate_json(line) for line in manifest.read_bytes().splitlines()
    )
    examples: list[Example] = []
    for identifier in identifiers:
        selected = tuple(example for example in source_examples if example.example_id == identifier)
        if len(selected) != 1:
            raise ValueError(f"Sampling check needs one actual source example for {identifier}")
        examples.append(selected[0])
    wrapper = FrozenQwen(config, torch.device("cuda"))
    utterances = tuple(TranscriptInput(example.user_text) for example in examples)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    responses = wrapper.generate_batch(tuple(examples), utterances, token_cap)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    repeated = wrapper.generate_batch(tuple(examples), utterances, token_cap)
    files = tuple(
        FileArtifact(
            path=path, source_path=path, bytes=path.stat().st_size, sha256=file_digest(path)
        )
        for path in (
            Path("speech_projector/llm.py"),
            Path("speech_projector/decoding.py"),
            Path("speech_projector/models.py"),
            Path(__file__),
        )
    )
    result = SamplingCompletionCheck(
        config=config,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_files=files,
        examples=tuple(examples),
        token_cap=token_cap,
        responses=responses,
        repeated_responses=repeated,
        exact_repeat=responses == repeated,
        generation_seconds=elapsed,
        peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    if not all(isinstance(response, CompletedGeneration) for response in responses):
        raise ValueError(f"Sampled completion check hit the cap; full evidence saved to {output}")
    print(
        f"Completed {len(responses)} sampled replies in {elapsed:.2f}s; "
        f"exact_repeat={result.exact_repeat}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--example-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-cap", type=int, required=True)
    arguments = parser.parse_args()
    run_check(
        arguments.manifest,
        arguments.run_config,
        tuple(arguments.example_id),
        arguments.output,
        arguments.token_cap,
    )


if __name__ == "__main__":
    main()
