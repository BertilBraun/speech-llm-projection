"""Compare stored loss and generation records on exactly the same fixed IDs."""

from collections.abc import Sequence
from pathlib import Path
from statistics import mean

from scripts.inventory_results import write_record
from speech_projector.evaluation import EvaluationOutcome, LossObservation, summarize_losses
from speech_projector.followup_evaluation import file_artifact, selected_sources
from speech_projector.followup_report import ComparisonInput
from speech_projector.models import (
    EvaluationCondition,
    EvaluationMetrics,
    FileArtifact,
    GenerationKind,
    Record,
)
from speech_projector.overnight_data import Cohort
from speech_projector.overnight_launcher import (
    FinalEvaluationSelection,
    load_evaluation,
    load_sources,
)


class MatchedEvaluationSource(ComparisonInput):
    condition: EvaluationCondition


class MatchedMetricsConfig(Record):
    selection: Path
    sources: Path
    cohort: Cohort
    systems: tuple[MatchedEvaluationSource, ...]
    output_directory: Path


class MatchedMeasurement(Record):
    name: str
    condition: EvaluationCondition
    metrics: EvaluationMetrics
    artifacts: tuple[FileArtifact, ...]


class MatchedMetricsReport(Record):
    configuration: MatchedMetricsConfig
    example_ids: tuple[str, ...]
    measurements: tuple[MatchedMeasurement, ...]


def subset_metrics(
    outcome: EvaluationOutcome, condition: EvaluationCondition, identifiers: Sequence[str]
) -> EvaluationMetrics:
    selected = set(identifiers)
    if not selected or len(selected) != len(identifiers):
        raise ValueError("Matched metrics require unique nonempty example IDs")
    losses = tuple(
        row
        for row in outcome.example_losses
        if row.condition == condition and row.example_id in selected
    )
    generations = tuple(
        row for row in outcome.samples if row.condition == condition and row.example_id in selected
    )
    if (
        len(losses) != len(selected)
        or {row.example_id for row in losses} != selected
        or len(generations) != len(selected)
        or {row.example_id for row in generations} != selected
    ):
        raise ValueError("Matched records must contain one loss and one generation per selected ID")
    cross_entropy, perplexity, tokens = summarize_losses(
        tuple(
            LossObservation(row.cross_entropy * row.target_tokens, row.target_tokens)
            for row in losses
        )
    )
    metadata = tuple(row.generation for row in generations)
    complete = all(row is not None for row in metadata)
    similarities = tuple(
        row.semantic_similarity for row in generations if row.semantic_similarity is not None
    )
    return EvaluationMetrics(
        examples=len(losses),
        target_tokens=tokens,
        cross_entropy=cross_entropy,
        perplexity=perplexity,
        semantic_similarity=mean(similarities) if similarities else None,
        generated_examples=len(generations),
        generated_tokens=sum(len(row.token_ids) for row in metadata if row is not None),
        completed_generations=sum(
            row.kind == GenerationKind.COMPLETED for row in metadata if row is not None
        )
        if complete
        else None,
        token_limited_generations=sum(
            row.kind == GenerationKind.TOKEN_LIMIT for row in metadata if row is not None
        )
        if complete
        else None,
    )


def compare_matched_metrics(configuration: MatchedMetricsConfig) -> MatchedMetricsReport:
    selection = FinalEvaluationSelection.model_validate_json(configuration.selection.read_bytes())
    sources = selected_sources(selection, load_sources(configuration.sources))
    by_source = {row.example_id: row for row in sources}
    by_example = {row.example_id: row for row in selection.examples}
    identifiers = tuple(
        identifier
        for identifier in selection.generation_example_ids
        if by_source[identifier].cohort == configuration.cohort
    )
    if not configuration.systems or len({row.name for row in configuration.systems}) != len(
        configuration.systems
    ):
        raise ValueError("Matched comparison requires unique named systems")
    measurements: list[MatchedMeasurement] = []
    token_counts: tuple[tuple[str, int], ...] | None = None
    for system in configuration.systems:
        outcome = load_evaluation(system.evaluation_directory)
        metrics = subset_metrics(outcome, system.condition, identifiers)
        losses = {
            row.example_id: row
            for row in outcome.example_losses
            if row.condition == system.condition and row.example_id in identifiers
        }
        actual_counts = tuple(
            (identifier, losses[identifier].target_tokens) for identifier in identifiers
        )
        if token_counts is not None and token_counts != actual_counts:
            raise ValueError("Matched comparison must use identical per-example target tokens")
        token_counts = actual_counts
        for row in outcome.samples:
            if row.condition == system.condition and row.example_id in identifiers:
                if row.gold_response != by_example[row.example_id].target_text:
                    raise ValueError("Matched generation references differ from the fixed panel")
        measurements.append(
            MatchedMeasurement(
                name=system.name,
                condition=system.condition,
                metrics=metrics,
                artifacts=tuple(
                    file_artifact(system.evaluation_directory / name)
                    for name in (
                        "evaluation.json",
                        "evaluation_losses.jsonl",
                        "evaluation_generations.jsonl",
                        "evaluation_conditioning.json",
                    )
                ),
            )
        )
    report = MatchedMetricsReport(
        configuration=configuration, example_ids=identifiers, measurements=tuple(measurements)
    )
    directory = configuration.output_directory
    write_record(directory / "matched_metrics.json", report)
    lines = [
        f"# Exactly matched {configuration.cohort.value} generation panel",
        "",
        "CE, semantics and completion below use identical IDs. CE is token-weighted, "
        "not a mean of per-example losses. This subset has no separately measured clock. "
        "Semantics are reference similarity, not correctness or tone accuracy.",
        "",
        "| System | IDs | Target tokens | CE | Semantic | EOS | Cap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in measurements:
        metrics = row.metrics
        lines.append(
            f"| {row.name} | {metrics.examples} | {metrics.target_tokens} | "
            f"{metrics.cross_entropy:.6f} | {metrics.semantic_similarity} | "
            f"{metrics.completed_generations} | {metrics.token_limited_generations} |"
        )
    lines += [
        "",
        "The larger cohort CE table uses different coverage and is reported separately. "
        "All intended-tone labels describe unverified synthetic settings. Predicted-tone "
        "and oracle-label provenance remain distinct even if their actual input labels coincide.",
    ]
    (directory / "matched_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
