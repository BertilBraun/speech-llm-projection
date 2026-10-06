"""Evaluate a preserved 20k first epoch against the 10k two-epoch budget."""

import argparse
import subprocess
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from scripts.package_results import FileArtifact, ModelRevision, file_digest, write_record
from scripts.snapshot_first_epoch import SnapshotProvenance
from speech_projector.data import load_examples
from speech_projector.evaluation import SemanticEvaluator, evaluate, save_evaluation
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
from speech_projector.training import TrainingState, weights_digest


class MatchedBudgetEvaluation(Record):
    snapshot: SnapshotProvenance
    reference: RunResult
    evaluation_git_commit: str
    manifest: FileArtifact
    model_revisions: tuple[ModelRevision, ...]
    frozen_llm_weights_sha256: str
    validation: EvaluationMetrics
    test: EvaluationMetrics
    evaluation_seconds: float
    peak_vram_gb: float


def validate_budget(snapshot: SnapshotProvenance, reference: RunResult, config: RunConfig) -> None:
    if snapshot.run_config != config:
        raise ValueError("Snapshot configuration differs from the run's canonical configuration")
    state = snapshot.training_state
    if (state.step, state.examples_seen, state.epoch, state.offset) != (2500, 20000, 1, 0):
        raise ValueError("Snapshot must preserve update 2500 / 20000 exposures at the first epoch")
    if config.train_examples != 20000 or config.epochs != 2:
        raise ValueError("The preserved larger-data comparison requires 20000 pairs and two epochs")
    if (
        reference.steps != state.step
        or reference.train_examples * reference.config.epochs != state.examples_seen
    ):
        raise ValueError(
            "Reference and snapshot must have the same update and example-exposure budget"
        )
    normalized = config.model_copy(
        update={"name": reference.config.name, "train_examples": reference.config.train_examples}
    )
    if normalized != reference.config:
        raise ValueError(
            "Matched-budget runs must share all configuration except name and subset size"
        )
    if (config.validation_examples, config.test_examples, config.semantic_examples) != (
        128,
        128,
        48,
    ):
        raise ValueError(
            "Matched-budget evaluation requires the fixed 128/128 splits and 48 generations"
        )


def read_verified_snapshot(run_directory: Path, reference: RunResult) -> SnapshotProvenance:
    directory = run_directory / "matched_budget_step2500"
    snapshot = SnapshotProvenance.model_validate_json((directory / "provenance.json").read_bytes())
    config = RunConfig.model_validate_json((run_directory / "config.json").read_bytes())
    validate_budget(snapshot, reference, config)
    expected_paths = {Path("projector.safetensors"), Path("state.json")}
    if {artifact.path for artifact in snapshot.artifacts} != expected_paths:
        raise ValueError(
            "Snapshot provenance must contain exactly the projector and training state"
        )
    for artifact in snapshot.artifacts:
        path = directory / artifact.path
        if path.stat().st_size != artifact.bytes or file_digest(path) != artifact.sha256:
            raise ValueError(f"Preserved snapshot integrity failed: {artifact.path}")
    state = TrainingState.model_validate_json((directory / "state.json").read_bytes())
    if state != snapshot.training_state:
        raise ValueError("Preserved training state differs from its snapshot provenance")
    return snapshot


def read_model_revisions(path: Path, config: RunConfig) -> tuple[ModelRevision, ...]:
    revisions = tuple(
        ModelRevision.model_validate_json(line) for line in path.read_bytes().splitlines()
    )
    available_names = {revision.model_name for revision in revisions if revision.snapshot_revisions}
    if not {config.model_name, config.speech_model_name}.issubset(available_names):
        raise ValueError(
            "Model revision inventory must identify the frozen Qwen and Whisper snapshots"
        )
    return revisions


def evaluate_matched_budget(
    run_directory: Path,
    reference_directory: Path,
    manifest: Path,
    model_revisions_path: Path,
    device: torch.device,
) -> MatchedBudgetEvaluation:
    reference = RunResult.model_validate_json((reference_directory / "result.json").read_bytes())
    snapshot = read_verified_snapshot(run_directory, reference)
    config = snapshot.run_config
    revisions = read_model_revisions(model_revisions_path, config)
    manifest_artifact = FileArtifact(
        path=Path(manifest.name),
        source_path=manifest,
        bytes=manifest.stat().st_size,
        sha256=file_digest(manifest),
    )
    directory = run_directory / "matched_budget"
    summary_path = directory / "summary.json"
    if summary_path.exists():
        saved = MatchedBudgetEvaluation.model_validate_json(summary_path.read_bytes())
        if (saved.snapshot, saved.reference, saved.manifest, saved.model_revisions) != (
            snapshot,
            reference,
            manifest_artifact,
            revisions,
        ):
            raise ValueError("Existing matched-budget evaluation has different input provenance")
        return saved
    started = time.monotonic()
    wrapper = FrozenQwen(config, device)
    projector = Projector(config.projector).to(device)
    checkpoint = run_directory / "matched_budget_step2500" / "projector.safetensors"
    projector.load_state_dict(load_file(str(checkpoint), device=str(device)), strict=True)
    semantic_evaluator = SemanticEvaluator()
    llm_sha256 = weights_digest(wrapper.model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    validation = evaluate(
        wrapper,
        projector,
        load_examples(manifest, Split.VALIDATION, config.validation_examples),
        config,
        EvaluationCondition.SPEECH,
        semantic_evaluator=semantic_evaluator,
    )
    save_evaluation(validation, directory / "validation")
    tested = evaluate(
        wrapper,
        projector,
        load_examples(manifest, Split.TEST, config.test_examples),
        config,
        EvaluationCondition.SPEECH,
        semantic_evaluator=semantic_evaluator,
    )
    save_evaluation(tested, directory / "test")
    summary = MatchedBudgetEvaluation(
        snapshot=snapshot,
        reference=reference,
        evaluation_git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        manifest=manifest_artifact,
        model_revisions=revisions,
        frozen_llm_weights_sha256=llm_sha256,
        validation=validation.metrics,
        test=tested.metrics,
        evaluation_seconds=time.monotonic() - started,
        peak_vram_gb=torch.cuda.max_memory_allocated(device) / 1e9
        if device.type == "cuda"
        else 0.0,
    )
    write_record(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-revisions", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    print(
        evaluate_matched_budget(
            arguments.run,
            arguments.reference,
            arguments.manifest,
            arguments.model_revisions,
            torch.device(arguments.device),
        ).model_dump_json(indent=2)
    )


if __name__ == "__main__":
    main()
