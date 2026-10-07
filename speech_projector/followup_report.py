"""Measured checkpoint and baseline comparisons without inference or test-led selection."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

import matplotlib
from pydantic import Field

from scripts.inventory_results import write_record
from speech_projector.evaluation import LossObservation, summarize_losses
from speech_projector.followup_evaluation import file_artifact, selected_sources
from speech_projector.followup_tone_baselines import ToneBaselineSummary
from speech_projector.models import (
    EvaluationCondition,
    EvaluationMetrics,
    FileArtifact,
    Record,
    RunResult,
)
from speech_projector.overnight_data import Cohort
from speech_projector.overnight_evaluation import (
    PairPreferenceSummary,
    SweepCandidate,
    ValidationCohorts,
    summarize_cohort_validation,
)
from speech_projector.overnight_launcher import (
    FinalEvaluationSelection,
    load_evaluation,
    load_sources,
)
from speech_projector.overnight_preparation import load_records
from speech_projector.teacher_evaluation import FidelitySummary

matplotlib.use("Agg")
from matplotlib import pyplot as plotting  # noqa: E402


class ComparisonInput(Record):
    name: str = Field(min_length=1)
    evaluation_directory: Path


class CheckpointComparisonInput(ComparisonInput):
    kind: Literal["checkpoint"] = "checkpoint"
    run_result: Path
    validation_candidate: Path
    fidelity_summary: Path
    neu_preference: Path


class BaselineComparisonInput(ComparisonInput):
    kind: Literal["baseline"] = "baseline"
    condition: Literal[EvaluationCondition.TEXT, EvaluationCondition.ASR]


ReportInput: TypeAlias = Annotated[
    CheckpointComparisonInput | BaselineComparisonInput, Field(discriminator="kind")
]


class FollowupReportConfig(Record):
    selection: Path
    sources: Path
    inputs: tuple[ReportInput, ...]
    output_directory: Path
    tone_baseline_summaries: tuple[Path, ...] = ()


class ConditionMeasurement(Record):
    name: str
    condition: EvaluationCondition
    pooled: EvaluationMetrics
    cohorts: ValidationCohorts
    artifacts: tuple[FileArtifact, ...]


class CheckpointMeasurement(ConditionMeasurement):
    kind: Literal["checkpoint"] = "checkpoint"
    run: RunResult
    candidate: SweepCandidate
    fidelity: tuple[FidelitySummary, ...]
    neu_preference: PairPreferenceSummary


class BaselineMeasurement(ConditionMeasurement):
    kind: Literal["baseline"] = "baseline"


ReportedMeasurement: TypeAlias = Annotated[
    CheckpointMeasurement | BaselineMeasurement, Field(discriminator="kind")
]


class FollowupMeasuredReport(Record):
    configuration: FollowupReportConfig
    selection: FileArtifact
    sources: FileArtifact
    measurements: tuple[ReportedMeasurement, ...]
    tone_baselines: tuple[ToneBaselineSummary, ...]
    tone_baseline_files: tuple[FileArtifact, ...]


def measured_condition(
    item: ReportInput, configuration: FollowupReportConfig
) -> ReportedMeasurement:
    selection = FinalEvaluationSelection.model_validate_json(configuration.selection.read_bytes())
    sources = selected_sources(selection, load_sources(configuration.sources))
    match item:
        case CheckpointComparisonInput():
            condition = EvaluationCondition.SPEECH
        case BaselineComparisonInput():
            condition = item.condition
    outcome = load_evaluation(item.evaluation_directory)
    primary_losses = tuple(row for row in outcome.example_losses if row.condition == condition)
    cross_entropy, _, tokens = summarize_losses(
        tuple(
            LossObservation(row.cross_entropy * row.target_tokens, row.target_tokens)
            for row in primary_losses
        )
    )
    if (
        outcome.metrics.examples != len(primary_losses)
        or outcome.metrics.target_tokens != tokens
        or not math.isclose(
            outcome.metrics.cross_entropy, cross_entropy, rel_tol=1e-7, abs_tol=1e-7
        )
    ):
        raise ValueError("Pooled metrics differ from saved primary loss records")
    primary = tuple(row for row in outcome.samples if row.condition == condition)
    if tuple(row.example_id for row in primary) != selection.generation_example_ids:
        raise ValueError("Comparison generations do not match the fixed ordered panel")
    examples = {row.example_id: row for row in selection.examples}
    for generation in primary:
        if generation.gold_response != examples[generation.example_id].target_text:
            raise ValueError("Comparison generations use different reference targets")
    artifacts = tuple(
        file_artifact(item.evaluation_directory / name)
        for name in (
            "evaluation.json",
            "evaluation_losses.jsonl",
            "evaluation_generations.jsonl",
            "evaluation_conditioning.json",
        )
    )
    cohorts = summarize_cohort_validation(outcome, sources, condition)
    match item:
        case BaselineComparisonInput():
            return BaselineMeasurement(
                name=item.name,
                condition=condition,
                pooled=outcome.metrics,
                cohorts=cohorts,
                artifacts=artifacts,
            )
        case CheckpointComparisonInput():
            result = RunResult.model_validate_json(item.run_result.read_bytes())
            candidate = SweepCandidate.model_validate_json(item.validation_candidate.read_bytes())
            if result.config != candidate.configuration:
                raise ValueError("Validation candidate and checkpoint configuration differ")
            return CheckpointMeasurement(
                name=item.name,
                condition=condition,
                pooled=outcome.metrics,
                cohorts=cohorts,
                artifacts=artifacts
                + tuple(
                    file_artifact(path)
                    for path in (
                        item.run_result,
                        item.validation_candidate,
                        item.fidelity_summary,
                        item.neu_preference,
                    )
                ),
                run=result,
                candidate=candidate,
                fidelity=load_records(item.fidelity_summary, FidelitySummary),
                neu_preference=PairPreferenceSummary.model_validate_json(
                    item.neu_preference.read_bytes()
                ),
            )


def cohort_rows(cohorts: ValidationCohorts) -> tuple[tuple[Cohort, EvaluationMetrics], ...]:
    return (
        (Cohort.ORDINARY, cohorts.old_ordinary),
        (Cohort.QWEN_EMOTIONAL, cohorts.old_emotional),
        (Cohort.NEU_EMOTIONAL, cohorts.new_neu_emotional),
    )


def render_measured_report(report: FollowupMeasuredReport) -> str:
    lines = [
        "# Lexical-alignment follow-up: measured comparisons",
        "",
        "TEST results are descriptive; branch selection uses the precommitted VAL rule. "
        "CE is token-weighted within each cohort; macro CE averages the three cohorts. "
        "Semantic similarity measures reference resemblance, not response correctness.",
        "",
        "| System | Cohort | CE examples | Target tokens | CE | Semantic | Generated | EOS | Cap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.measurements:
        for cohort, metrics in cohort_rows(row.cohorts):
            semantic = (
                f"{metrics.semantic_similarity:.4f}"
                if metrics.semantic_similarity is not None
                else "unmeasured"
            )
            lines.append(
                f"| {row.name} | {cohort.value} | {metrics.examples} | {metrics.target_tokens} | "
                f"{metrics.cross_entropy:.5f} | {semantic} | {metrics.generated_examples} | "
                f"{metrics.completed_generations} | {metrics.token_limited_generations} |"
            )
    lines += ["", "| System | Pooled CE | Macro CE |", "|---|---:|---:|"]
    for row in report.measurements:
        lines.append(
            f"| {row.name} | {row.pooled.cross_entropy:.5f} | "
            f"{row.cohorts.macro_cross_entropy:.5f} |"
        )
    lines += [
        "",
        "| Checkpoint | Updates | VAL ordinary CE | VAL macro CE | "
        "Recorded cumulative training seconds | Peak PyTorch allocated decimal GB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report.measurements:
        if isinstance(row, CheckpointMeasurement):
            lines.append(
                f"| {row.name} | {row.run.steps} | "
                f"{row.candidate.validation.old_ordinary.cross_entropy:.5f} | "
                f"{row.candidate.validation.macro_cross_entropy:.5f} | "
                f"{row.run.runtime_seconds:.1f} | {row.run.peak_vram_gb:.3f} |"
            )
    lines += [
        "",
        "Training elapsed counters include inherited parent checkpoint time and interval "
        "evaluation. Shared parents must be deduplicated for resource totals; incremental "
        "branch time requires subtracting the source checkpoint elapsed counter. Setup, "
        "final held-out evaluation and fidelity are separate from this training counter. "
        "Reused frozen TEXT/ASR baselines retain original timing and add no new GPU work. "
        "Cohort clocks are unmeasured.",
        "",
        "| Checkpoint | TEST Neu pairs/families | Raw matching margin [95% CI] | "
        "Resized matching margin [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for row in report.measurements:
        if isinstance(row, CheckpointMeasurement):
            preference = row.neu_preference
            raw = preference.matching_margin
            resized = preference.resized_matching_margin
            lines.append(
                f"| {row.name} | {preference.pairs}/{preference.family_clusters} | "
                f"{raw.estimate:.5f} [{raw.lower:.5f}, {raw.upper:.5f}] | "
                f"{resized.estimate:.5f} [{resized.lower:.5f}, {resized.upper:.5f}] |"
            )
    lines += [
        "",
        "Margins compare matched and swapped same-word audio against both saved teacher "
        "targets. The resized control linearly resamples wrong-audio encoder states to hold "
        "pseudo-token count, changing feature statistics. These are supporting conditioning "
        "diagnostics, not emotion classification accuracy or standalone semantic proof. "
        "Intended synthetic tone labels are not human-verified audible emotions.",
        "",
        "| Checkpoint | Fidelity condition | Examples | First-token agreement | "
        "First-8 agreement | Full-prefix agreement |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report.measurements:
        if isinstance(row, CheckpointMeasurement):
            for fidelity in row.fidelity:
                lines.append(
                    f"| {row.name} | {fidelity.condition.value} | {fidelity.examples} | "
                    f"{fidelity.first_token_agreement:.4f} | "
                    f"{fidelity.first_8_token_agreement:.4f} | {fidelity.top1_agreement:.4f} |"
                )
    lines += [
        "",
        "All fidelity positions use the same teacher target prefix; late-prefix agreement "
        "is easier and does not establish free-running response quality. Blinded 0–2 "
        "tone/grounding reviews remain separate model-assisted ordinal judgments.",
    ]
    if report.tone_baselines:
        lines += [
            "",
            "| Tone-cue method | Neu examples | CE | Semantic | Generated | EOS | Cap |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for baseline in report.tone_baselines:
            metrics = baseline.metrics
            lines.append(
                f"| {baseline.provenance.configuration.kind} | {metrics.examples} | "
                f"{metrics.cross_entropy:.5f} | {metrics.semantic_similarity} | "
                f"{metrics.generated_examples} | {metrics.completed_generations} | "
                f"{metrics.token_limited_generations} |"
            )
        lines += [
            "",
            "This is the fixed Neu generation panel, not the larger CE panel above. "
            "Predicted cues use a TRAIN-only acoustic classifier. Privileged intended-label "
            "cues are an information-complete reference for these synthetic annotations. "
            "When predicted and intended labels match exactly, the oracle output is "
            "explicitly reused; it adds no inference runtime. Plain ASR remains a words-only "
            "comparator. Cue labels are not verified natural emotion perception.",
        ]
    return "\n".join(lines) + "\n"


def plot_comparison(report: FollowupMeasuredReport, destination: Path) -> None:
    figure, axes = plotting.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    names = tuple(row.name for row in report.measurements)
    positions = tuple(range(len(names)))
    for index, cohort in enumerate(Cohort):
        metrics = tuple(cohort_rows(row.cohorts)[index][1] for row in report.measurements)
        axes[0].plot(positions, [row.cross_entropy for row in metrics], "o-", label=cohort.value)
        axes[1].plot(
            positions,
            [
                float("nan") if row.semantic_similarity is None else row.semantic_similarity
                for row in metrics
            ],
            "o-",
            label=cohort.value,
        )
    for axis in axes:
        axis.set_xticks(positions, names, rotation=30, ha="right")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Held-out target CE (token weighted within cohort)")
    axes[1].set_ylabel("Reference semantic similarity (secondary proxy)")
    figure.suptitle(
        "Fixed TEST panel; one-seed point estimates\n"
        "No test-led selection; semantic similarity is a secondary proxy",
        fontsize=12,
    )
    for extension in ("png", "pdf"):
        figure.savefig(destination.with_suffix("." + extension), dpi=180)
    plotting.close(figure)


def write_comparison(configuration: FollowupReportConfig) -> FollowupMeasuredReport:
    names = tuple(item.name for item in configuration.inputs)
    if not names or len(set(names)) != len(names):
        raise ValueError("Report requires unique nonempty comparison systems")
    report = FollowupMeasuredReport(
        configuration=configuration,
        selection=file_artifact(configuration.selection),
        sources=file_artifact(configuration.sources),
        measurements=tuple(
            measured_condition(item, configuration) for item in configuration.inputs
        ),
        tone_baselines=tuple(
            ToneBaselineSummary.model_validate_json(path.read_bytes())
            for path in configuration.tone_baseline_summaries
        ),
        tone_baseline_files=tuple(
            file_artifact(path) for path in configuration.tone_baseline_summaries
        ),
    )
    directory = configuration.output_directory
    write_record(directory / "measured_comparison.json", report)
    (directory / "measured_comparison.md").write_text(
        render_measured_report(report), encoding="utf-8"
    )
    with (directory / "cohort_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "system",
                "cohort",
                "examples",
                "target_tokens",
                "ce",
                "semantic",
                "generated",
                "completed",
                "capped",
            )
        )
        for row in report.measurements:
            for cohort, metrics in cohort_rows(row.cohorts):
                writer.writerow(
                    (
                        row.name,
                        cohort.value,
                        metrics.examples,
                        metrics.target_tokens,
                        metrics.cross_entropy,
                        metrics.semantic_similarity,
                        metrics.generated_examples,
                        metrics.completed_generations,
                        metrics.token_limited_generations,
                    )
                )
    plot_comparison(report, directory / "heldout_comparison")
    return report
