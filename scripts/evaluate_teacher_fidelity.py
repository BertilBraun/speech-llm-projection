"""Resume same-prefix teacher fidelity evaluation for a saved projector checkpoint."""

import argparse
import subprocess
import time
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file

from scripts.package_results import FileArtifact, file_digest
from speech_projector.data import load_examples
from speech_projector.llm import FrozenQwen
from speech_projector.models import EvaluationCondition, Record, RunConfig, Split
from speech_projector.projectors import Projector
from speech_projector.teacher_evaluation import (
    FidelitySummary,
    evaluate_teacher_fidelity,
    save_teacher_fidelity,
    summarize_fidelity,
)


class FidelityJob(Record):
    config_path: Path
    manifest: Path
    checkpoint: Path
    output_directory: Path
    device: Literal["cpu", "cuda"] = "cuda"


class FidelityRunInputs(Record):
    run: RunConfig
    artifacts: tuple[FileArtifact, ...]


class FidelitySplitSummary(Record):
    split: Split
    metrics: FidelitySummary


class FidelityRunSummary(Record):
    inputs: FidelityRunInputs
    evaluation_source_commit: str
    metrics: tuple[FidelitySplitSummary, ...]
    process_seconds: float
    peak_vram_gb: float


def evaluate_checkpoint(job: FidelityJob) -> FidelityRunSummary:
    config = RunConfig.model_validate_json(job.config_path.read_bytes())
    inputs = FidelityRunInputs(
        run=config,
        artifacts=tuple(
            FileArtifact(
                path=path, source_path=path, bytes=path.stat().st_size, sha256=file_digest(path)
            )
            for path in (job.config_path, job.manifest, job.checkpoint)
        ),
    )
    job.output_directory.mkdir(parents=True, exist_ok=True)
    inputs_path = job.output_directory / "inputs.json"
    if inputs_path.exists():
        if FidelityRunInputs.model_validate_json(inputs_path.read_bytes()) != inputs:
            raise ValueError("Fidelity output belongs to different input artifacts/configuration")
    else:
        inputs_path.write_text(inputs.model_dump_json(indent=2) + "\n", encoding="utf-8")
    summary_path = job.output_directory / "summary.json"
    if summary_path.exists():
        return FidelityRunSummary.model_validate_json(summary_path.read_bytes())
    started = time.perf_counter()
    device = torch.device(job.device)
    wrapper = FrozenQwen(config, device)
    projector = Projector(config.projector).to(device)
    projector.load_state_dict(load_file(str(job.checkpoint), device=job.device), strict=True)
    summaries: list[FidelitySplitSummary] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for split, count in (
        (Split.VALIDATION, config.validation_examples),
        (Split.TEST, config.test_examples),
    ):
        examples = load_examples(job.manifest, split, count)
        if len(examples) != count:
            raise ValueError(f"Fidelity manifest has insufficient {split.value} examples")
        observations = evaluate_teacher_fidelity(
            wrapper, projector, examples, job.output_directory / split.value
        )
        save_teacher_fidelity(observations, job.output_directory / split.value)
        for condition in (
            EvaluationCondition.SPEECH,
            EvaluationCondition.SHUFFLED_SPEECH,
            EvaluationCondition.ZERO_SPEECH,
        ):
            summaries.append(
                FidelitySplitSummary(
                    split=split, metrics=summarize_fidelity(observations, condition)
                )
            )
    summary = FidelityRunSummary(
        inputs=inputs,
        evaluation_source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        metrics=tuple(summaries),
        process_seconds=time.perf_counter() - started,
        peak_vram_gb=torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0,
    )
    partial = summary_path.with_suffix(".partial")
    partial.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    partial.replace(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", dest="config_path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", dest="output_directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    job = FidelityJob.model_validate(vars(parser.parse_args()))
    print(evaluate_checkpoint(job).model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
