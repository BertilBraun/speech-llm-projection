"""Continue a completed lexical branch without changing its training objective or trajectory."""

import torch

from speech_projector.evaluation import SemanticEvaluator
from speech_projector.followup_evaluation import bind_provenance
from speech_projector.llm import FrozenQwen
from speech_projector.models import Example, FileArtifact, Record, RunConfig, RunResult
from speech_projector.objective_branch import (
    ObjectiveBranchData,
    ObjectiveBranchJob,
    ObjectiveGradientReport,
    check_branch_gradients,
    load_branch_data,
)
from speech_projector.overnight_continuation import ContinuationProvenance, prepare_continuation
from speech_projector.overnight_evaluation import SweepCandidate
from speech_projector.overnight_launcher import run_candidate
from speech_projector.overnight_preparation import artifact, load_records
from speech_projector.training import TrainingState


class ObjectiveContinuationProvenance(Record):
    job: ObjectiveBranchJob
    continuation: ContinuationProvenance
    data_artifacts: tuple[FileArtifact, ...]


def prepare_objective_continuation(
    job: ObjectiveBranchJob, data: ObjectiveBranchData
) -> ObjectiveContinuationProvenance:
    source = job.source_run
    original = RunConfig.model_validate_json((source / "config.json").read_bytes())
    state = TrainingState.model_validate_json((source / "checkpoint" / "state.json").read_bytes())
    candidate = SweepCandidate.model_validate_json((source / "candidate.json").read_bytes())
    result = RunResult.model_validate_json((source / "result.json").read_bytes())
    if (
        state.final_validation_loss is None
        or state.final_fixed_training_loss is None
        or state.step != original.max_optimizer_updates
        or state.step != result.steps
        or result.config != original
        or candidate.configuration != original
    ):
        raise ValueError("Objective continuation requires a completed, consistent selected branch")
    effective_batch = original.microbatch_size * original.gradient_accumulation
    updates_per_epoch = (original.train_examples + effective_batch - 1) // effective_batch
    if job.configuration.max_optimizer_updates != 2 * updates_per_epoch or original.epochs < 2:
        raise ValueError("Selected continuation must end at exactly the second complete data pass")
    if (
        state.epoch != 1
        or state.offset < 0
        or state.offset >= original.train_examples
        or state.offset % effective_batch != 0
        or state.step != updates_per_epoch + state.offset // effective_batch
        or state.examples_seen != original.train_examples + state.offset
    ):
        raise ValueError("Selected branch checkpoint has an inconsistent second-pass cursor")
    subset_path = source / "subset.jsonl"
    if load_records(subset_path, Example) != tuple(data.training):
        raise ValueError("Continuation training examples differ from the parent's saved subset")
    paths = (
        subset_path,
        source / "candidate.json",
        source / "result.json",
        job.data_root / "examples.jsonl",
        job.data_root / "sources.jsonl",
        job.data_root / "fixed_validation.jsonl",
        job.data_root / "fixed_validation_sources.jsonl",
    )
    records = tuple(artifact(path) for path in paths)
    destination = job.output_root / job.configuration.name
    continuation = prepare_continuation(source, destination, job.configuration)
    if tuple(artifact(path) for path in paths) != records:
        raise ValueError("Parent or dataset changed during continuation preparation")
    provenance = ObjectiveContinuationProvenance(
        job=job, continuation=continuation, data_artifacts=records
    )
    bind_provenance(destination / "continuation_data.json", provenance)
    return provenance


def run_objective_continuation(
    job: ObjectiveBranchJob, *, smoke_only: bool
) -> SweepCandidate | ObjectiveGradientReport:
    data = load_branch_data(job)
    prepare_objective_continuation(job, data)
    wrapper = FrozenQwen(job.configuration, torch.device("cuda"))
    checked = check_branch_gradients(job, data, wrapper)
    if smoke_only:
        return checked
    return run_candidate(
        job.configuration,
        data.training,
        data.validation,
        data.validation_sources,
        wrapper,
        SemanticEvaluator(),
        job.output_root,
        training_sources=data.training_sources,
    )
