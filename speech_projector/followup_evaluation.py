"""Fixed held-out checkpoint comparisons for the lexical-alignment follow-up."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, model_validator
from safetensors.torch import load_file

from scripts.inventory_results import stable_digest, write_record
from speech_projector.evaluation import SemanticEvaluator, evaluate, save_evaluation
from speech_projector.journal import read_journal
from speech_projector.launcher import initialize_projector
from speech_projector.llm import FrozenQwen
from speech_projector.models import (
    AsrTranscript,
    EvaluationCondition,
    FileArtifact,
    Record,
    RunResult,
    Split,
)
from speech_projector.overnight_data import Cohort, SourceSidecar
from speech_projector.overnight_evaluation import (
    ValidationCohorts,
    build_emotion_pairs,
    evaluate_emotion_pairs,
    summarize_cohort_validation,
    summarize_preferences,
)
from speech_projector.overnight_launcher import (
    FinalEvaluationSelection,
    final_evaluation_config,
    load_evaluation,
    load_sources,
)
from speech_projector.teacher_evaluation import evaluate_teacher_fidelity, save_teacher_fidelity


class FollowupEvaluationConfig(Record):
    run_result: Path
    selection: Path
    sources: Path
    asr_transcripts: Path
    output_directory: Path
    source_git_commit: str = Field(min_length=7)
    fidelity: bool = True
    paired_preferences: bool = True
    conditions: tuple[
        Literal[EvaluationCondition.SPEECH, EvaluationCondition.TEXT, EvaluationCondition.ASR], ...
    ] = (EvaluationCondition.SPEECH,)

    @model_validator(mode="after")
    def validate_conditions(self) -> FollowupEvaluationConfig:
        if not self.conditions or len(set(self.conditions)) != len(self.conditions):
            raise ValueError("Evaluation conditions must be nonempty and unique")
        return self


class FollowupEvaluationProvenance(Record):
    configuration: FollowupEvaluationConfig
    inputs: tuple[FileArtifact, ...]


class ConditionCohorts(Record):
    condition: EvaluationCondition
    cohorts: ValidationCohorts


class FollowupEvaluationSummary(Record):
    provenance: FollowupEvaluationProvenance
    conditions: tuple[ConditionCohorts, ...]


def file_artifact(path: Path) -> FileArtifact:
    size, digest = stable_digest(path)
    return FileArtifact(path=path, source_path=path.resolve(), bytes=size, sha256=digest)


def bind_provenance(path: Path, provenance: Record) -> None:
    """A saved computation may resume only against exactly the same inputs."""
    if path.exists():
        if type(provenance).model_validate_json(path.read_bytes()) != provenance:
            raise ValueError(f"Evaluation provenance changed: {path}")
    else:
        write_record(path, provenance)


def selected_sources(
    selection: FinalEvaluationSelection, sources: Sequence[SourceSidecar]
) -> tuple[SourceSidecar, ...]:
    identifiers = tuple(example.example_id for example in selection.examples)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Fixed evaluation selection must contain unique examples")
    source_identifiers = tuple(source.example_id for source in sources)
    if len(set(source_identifiers)) != len(source_identifiers):
        raise ValueError("Cohort source records contain duplicate examples")
    if set(identifiers) - set(source_identifiers):
        raise ValueError("Fixed evaluation examples lack cohort source records")
    splits = {example.split for example in selection.examples}
    if len(splits) != 1 or Split.TRAIN in splits:
        raise ValueError("Fixed evaluation must contain one held-out split")
    if selection.generation_example_ids != identifiers[: len(selection.generation_example_ids)]:
        raise ValueError("Generation examples must be the fixed selection prefix")
    selected_identifiers = set(identifiers)
    return tuple(source for source in sources if source.example_id in selected_identifiers)


def evaluate_followup(configuration: FollowupEvaluationConfig) -> FollowupEvaluationSummary:
    result = RunResult.model_validate_json(configuration.run_result.read_bytes())
    selection = FinalEvaluationSelection.model_validate_json(configuration.selection.read_bytes())
    sources = selected_sources(selection, load_sources(configuration.sources))
    provenance = FollowupEvaluationProvenance(
        configuration=configuration,
        inputs=tuple(
            file_artifact(path)
            for path in (
                configuration.run_result,
                result.checkpoint_path,
                configuration.selection,
                configuration.sources,
                configuration.asr_transcripts,
            )
        ),
    )
    bind_provenance(configuration.output_directory / "provenance.json", provenance)
    config = final_evaluation_config(result.config, selection)
    if config.generation_batch_size < 2:
        config = config.model_copy(update={"generation_batch_size": 4})
    wrapper = FrozenQwen(config, torch.device("cuda"))
    projector = initialize_projector(config, wrapper.device)
    projector.load_state_dict(load_file(str(result.checkpoint_path), device=str(wrapper.device)))
    transcripts = read_journal(configuration.asr_transcripts, AsrTranscript)
    semantic = SemanticEvaluator()
    summaries: list[ConditionCohorts] = []
    for condition in configuration.conditions:
        directory = configuration.output_directory / condition.value
        complete = directory / "evaluation_qualitative.md"
        if complete.exists():
            outcome = load_evaluation(directory)
        else:
            outcome = evaluate(
                wrapper,
                projector if condition == EvaluationCondition.SPEECH else None,
                selection.examples,
                config,
                condition,
                asr_transcripts=transcripts,
                semantic_evaluator=semantic,
            )
            save_evaluation(outcome, directory)
        summaries.append(
            ConditionCohorts(
                condition=condition,
                cohorts=summarize_cohort_validation(outcome, sources, condition),
            )
        )
    if configuration.paired_preferences:
        for cohort in (Cohort.QWEN_EMOTIONAL, Cohort.NEU_EMOTIONAL):
            pairs = build_emotion_pairs(
                selection.examples, sources, cohort, selection.examples[0].split
            )
            observations = evaluate_emotion_pairs(
                wrapper, projector, pairs, configuration.output_directory / cohort.value
            )
            write_record(
                configuration.output_directory / cohort.value / "summary.json",
                summarize_preferences(observations),
            )
    if configuration.fidelity:
        observations = evaluate_teacher_fidelity(
            wrapper, projector, selection.examples, configuration.output_directory / "fidelity"
        )
        save_teacher_fidelity(observations, configuration.output_directory / "fidelity")
    summary = FollowupEvaluationSummary(provenance=provenance, conditions=tuple(summaries))
    write_record(configuration.output_directory / "summary.json", summary)
    return summary
