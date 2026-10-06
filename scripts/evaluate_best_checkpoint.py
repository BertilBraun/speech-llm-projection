"""Evaluate saved validation-selected projectors after the controlled suite completes."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from speech_projector.data import load_examples
from speech_projector.evaluation import (
    SemanticEvaluator,
    evaluate,
    save_evaluation,
)
from speech_projector.llm import FrozenQwen
from speech_projector.models import (
    EvaluationCondition,
    EvaluationMetrics,
    Record,
    RunConfig,
    RunResult,
    Split,
)
from speech_projector.projectors import Projector
from speech_projector.training import ValidationCheckpointRecord


class BestCheckpointSummary(Record):
    config: RunConfig
    training_git_commit: str
    evaluation_git_commit: str
    checkpoint_path: Path
    checkpoint_sha256: str
    selected_step: int
    selection_validation_cross_entropy: float
    final_checkpoint_validation_cross_entropy: float
    validation: EvaluationMetrics
    test: EvaluationMetrics
    evaluation_seconds: float
    peak_vram_gb: float


def evaluate_best_checkpoint(
    run_directory: Path, manifest: Path, device: torch.device
) -> BestCheckpointSummary:
    result = RunResult.model_validate_json((run_directory / "result.json").read_bytes())
    selection = ValidationCheckpointRecord.model_validate_json(
        (run_directory / "best_validation.json").read_bytes()
    )
    checkpoint = run_directory / "best_projector.safetensors"
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output_directory = run_directory / "best_checkpoint"
    summary_path = output_directory / "summary.json"
    if summary_path.exists():
        saved = BestCheckpointSummary.model_validate_json(summary_path.read_bytes())
        if saved.checkpoint_sha256 != checkpoint_digest or saved.config != result.config:
            raise ValueError("Supplementary evaluation does not match the current best checkpoint")
        return saved
    started = time.monotonic()
    configuration = result.config
    wrapper = FrozenQwen(configuration, device)
    projector = Projector(configuration.projector).to(device)
    projector.load_state_dict(load_file(str(checkpoint), device=str(device)), strict=True)
    semantic_evaluator = SemanticEvaluator()
    validation_examples = load_examples(manifest, Split.VALIDATION, result.validation_examples)
    test_examples = load_examples(manifest, Split.TEST, result.test_examples)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    validation = evaluate(
        wrapper,
        projector,
        validation_examples,
        configuration,
        EvaluationCondition.SPEECH,
        semantic_evaluator=semantic_evaluator,
    )
    save_evaluation(validation, output_directory / "validation")
    tested = evaluate(
        wrapper,
        projector,
        test_examples,
        configuration,
        EvaluationCondition.SPEECH,
        semantic_evaluator=semantic_evaluator,
    )
    save_evaluation(tested, output_directory / "test")
    summary = BestCheckpointSummary(
        config=configuration,
        training_git_commit=result.git_commit,
        evaluation_git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_digest,
        selected_step=selection.step,
        selection_validation_cross_entropy=selection.cross_entropy,
        final_checkpoint_validation_cross_entropy=result.validation.cross_entropy,
        validation=validation.metrics,
        test=tested.metrics,
        evaluation_seconds=time.monotonic() - started,
        peak_vram_gb=torch.cuda.max_memory_allocated(device) / 1e9
        if device.type == "cuda"
        else 0.0,
    )
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    result = evaluate_best_checkpoint(
        arguments.run, arguments.manifest, torch.device(arguments.device)
    )
    print(result.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
