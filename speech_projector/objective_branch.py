"""One controlled objective continuation with parent-weight gradient evidence."""

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from scripts.inventory_results import write_record
from scripts.package_results import FileArtifact
from speech_projector.evaluation import SemanticEvaluator
from speech_projector.launcher import initialize_projector
from speech_projector.llm import FrozenQwen
from speech_projector.models import (
    Example,
    GradientCheck,
    OrdinaryResponseKLObjective,
    Record,
    ResponseCrossEntropyObjective,
    RunConfig,
    Split,
    TranscriptMixtureObjective,
)
from speech_projector.overnight_continuation import prepare_objective_branch
from speech_projector.overnight_data import Cohort, SourceSidecar
from speech_projector.overnight_evaluation import SweepCandidate
from speech_projector.overnight_launcher import load_sources, run_candidate
from speech_projector.overnight_preparation import artifact, load_records
from speech_projector.training import TrainingState, objective_gradient_sanity
from speech_projector.training_objectives import (
    ObjectiveSmokeExample,
    align_sources,
    select_objective_smoke,
)


class ObjectiveBranchJob(Record):
    source_run: Path
    data_root: Path
    output_root: Path
    configuration: RunConfig


@dataclass(frozen=True)
class ObjectiveBranchData:
    training: list[Example]
    training_sources: tuple[SourceSidecar, ...]
    validation: tuple[Example, ...]
    validation_sources: tuple[SourceSidecar, ...]


class ObjectiveGradientEvidence(Record):
    example: Example
    source: SourceSidecar
    epoch: int
    check: GradientCheck


class ObjectiveGradientReport(Record):
    job: ObjectiveBranchJob
    parent_projector: FileArtifact
    evidence: tuple[ObjectiveGradientEvidence, ...]


def load_branch_data(job: ObjectiveBranchJob) -> ObjectiveBranchData:
    examples = load_records(job.data_root / "examples.jsonl", Example)
    training = [example for example in examples if example.split == Split.TRAIN]
    identifiers = {example.example_id for example in training}
    sources = load_sources(job.data_root / "sources.jsonl")
    training_sources = tuple(source for source in sources if source.example_id in identifiers)
    if (
        len(training) != job.configuration.train_examples
        or len(identifiers) != len(training)
        or len(training_sources) != len(training)
        or {source.example_id for source in training_sources} != identifiers
    ):
        raise ValueError("Branch requires exact unique training examples and source coverage")
    align_sources(job.configuration, training, training_sources)
    validation = load_records(job.data_root / "fixed_validation.jsonl", Example)
    validation_sources = load_sources(job.data_root / "fixed_validation_sources.jsonl")
    if any(not example.feature_path.is_file() for example in (*training, *validation)):
        raise ValueError("All branch training and validation features must already be cached")
    return ObjectiveBranchData(training, training_sources, validation, validation_sources)


def gradient_examples(
    wrapper: FrozenQwen, data: ObjectiveBranchData, epoch: int
) -> tuple[ObjectiveSmokeExample, ...]:
    indexed = {source.example_id: source for source in data.training_sources}
    ordinary = tuple(
        ObjectiveSmokeExample(example, indexed[example.example_id])
        for example in data.training
        if indexed[example.example_id].cohort == Cohort.ORDINARY
    )
    if not ordinary:
        raise ValueError(
            "Objective branch requires ordinary examples for its lexical gradient gate"
        )
    match wrapper.config.objective:
        case ResponseCrossEntropyObjective():
            first = ordinary[0]
        case TranscriptMixtureObjective() | OrdinaryResponseKLObjective():
            first = select_objective_smoke(
                wrapper.config, data.training, data.training_sources, epoch
            )
    selected = [first]
    longest = max(ordinary, key=lambda item: wrapper.target_token_count(item.example))
    if longest.example.example_id != first.example.example_id:
        selected.append(longest)
    emotional = next(
        (
            ObjectiveSmokeExample(example, indexed[example.example_id])
            for example in data.training
            if indexed[example.example_id].cohort != ordinary[0].source.cohort
        ),
        None,
    )
    if emotional is not None and all(
        chosen.example.example_id != emotional.example.example_id for chosen in selected
    ):
        selected.append(emotional)
    return tuple(selected)


def check_branch_gradients(
    job: ObjectiveBranchJob, data: ObjectiveBranchData, wrapper: FrozenQwen
) -> ObjectiveGradientReport:
    directory = job.output_root / "smoke" / job.configuration.name
    checkpoint = job.source_run / "checkpoint" / "projector.safetensors"
    state = TrainingState.model_validate_json(
        (job.source_run / "checkpoint" / "state.json").read_bytes()
    )
    parent_projector = artifact(checkpoint)
    selected_examples = gradient_examples(wrapper, data, state.epoch)
    report_path = directory / "objective_gradient_report.json"
    evidence: list[ObjectiveGradientEvidence] = []
    if report_path.exists():
        previous = ObjectiveGradientReport.model_validate_json(report_path.read_bytes())
        if previous.job != job or previous.parent_projector != parent_projector:
            raise ValueError("Cached objective gradient report differs from its pinned branch")
        if len(previous.evidence) > len(selected_examples):
            raise ValueError("Cached gradient report contains unexpected examples")
        for checked, selected in zip(previous.evidence, selected_examples, strict=False):
            if (
                checked.example != selected.example
                or checked.source != selected.source
                or checked.epoch != state.epoch
                or checked.check.projector_gradient_norm <= 0
                or not checked.check.projector_changed
                or not checked.check.llm_weights_unchanged
                or checked.check.llm_has_gradients
            ):
                raise ValueError("Cached objective gradient evidence is invalid or misaligned")
        evidence.extend(previous.evidence)
    for selected in selected_examples[len(evidence) :]:
        projector = initialize_projector(job.configuration, wrapper.device)
        projector.load_state_dict(load_file(checkpoint))
        check = objective_gradient_sanity(
            wrapper, projector, selected.example, selected.source, state.epoch
        )
        evidence.append(
            ObjectiveGradientEvidence(
                example=selected.example, source=selected.source, epoch=state.epoch, check=check
            )
        )
        report = ObjectiveGradientReport(
            job=job, parent_projector=parent_projector, evidence=tuple(evidence)
        )
        write_record(report_path, report)
    assert evidence
    report = ObjectiveGradientReport(
        job=job, parent_projector=parent_projector, evidence=tuple(evidence)
    )
    if artifact(checkpoint) != parent_projector:
        raise ValueError("Parent checkpoint changed during the objective gradient gate")
    write_record(directory / "gradient_check.json", evidence[0].check)
    return report


def run_objective_branch(
    job: ObjectiveBranchJob, *, smoke_only: bool
) -> SweepCandidate | ObjectiveGradientReport:
    data = load_branch_data(job)
    prepare_objective_branch(
        job.source_run, job.output_root / job.configuration.name, job.configuration
    )
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
