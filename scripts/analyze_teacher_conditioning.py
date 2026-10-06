"""CPU-only history strata and paired input-sensitivity diagnostics from saved records."""

import argparse
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, TypeVar

from pydantic import Field, TypeAdapter

from scripts.package_results import FileArtifact
from scripts.summarize_teacher_targets import HistoryGroup
from speech_projector.evaluation import (
    ConditioningDiagnostic,
    ExampleLoss,
    LossObservation,
    conditioning_diagnostic,
    summarize_losses,
)
from speech_projector.judge import (
    BootstrapInterval,
    PairedMetricObservation,
    paired_dialogue_bootstrap,
)
from speech_projector.models import (
    EvaluationCondition,
    EvaluationMetrics,
    Example,
    Record,
    RunResult,
    Split,
    SuiteState,
)
from speech_projector.teacher_evaluation import (
    FidelityProvenance,
    FidelitySummary,
    TeacherFidelity,
    summarize_fidelity,
)

TRecord = TypeVar("TRecord", bound=Record)
TObservation = TypeVar("TObservation", ExampleLoss, TeacherFidelity)


@dataclass(frozen=True)
class SerializedExample:
    example: Example
    line: bytes


class EstimatedMargin(Record):
    kind: Literal["estimated"] = "estimated"
    interval: BootstrapInterval


class InsufficientDialogues(Record):
    kind: Literal["insufficient_dialogues"] = "insufficient_dialogues"
    examples: int
    dialogues: int


MarginEstimate = Annotated[EstimatedMargin | InsufficientDialogues, Field(discriminator="kind")]


class ConditionLoss(Record):
    condition: EvaluationCondition
    metrics: EvaluationMetrics


class LossMargin(Record):
    diagnostic: ConditioningDiagnostic
    example_weighted: MarginEstimate
    token_weighted: MarginEstimate


class FidelityMetric(str, Enum):
    CROSS_ENTROPY = "control_minus_correct_cross_entropy"
    KL = "control_minus_correct_kl"
    AGREEMENT = "correct_minus_control_top1_agreement"
    TARGET_ACCURACY = "correct_minus_control_target_accuracy"
    FIRST_TOKEN_AGREEMENT = "correct_minus_control_first_token_agreement"
    FIRST_TOKEN_ACCURACY = "correct_minus_control_first_token_target_accuracy"
    FIRST_8_AGREEMENT = "correct_minus_control_first_8_token_agreement"
    FIRST_8_ACCURACY = "correct_minus_control_first_8_token_target_accuracy"


class FidelityMargin(Record):
    correct_condition: EvaluationCondition
    control_condition: EvaluationCondition
    metric: FidelityMetric
    estimate: MarginEstimate


class StratumDiagnostics(Record):
    examples: int
    losses: tuple[ConditionLoss, ...]
    fidelity: tuple[FidelitySummary, ...]
    loss_margins: tuple[LossMargin, ...]
    fidelity_margins: tuple[FidelityMargin, ...]


class HistoryStratum(Record):
    history: HistoryGroup
    diagnostics: StratumDiagnostics


class SplitDiagnostics(Record):
    split: Split
    provenance: FidelityProvenance
    inputs: tuple[FileArtifact, ...]
    overall: StratumDiagnostics
    history: tuple[HistoryStratum, ...]


class TeacherConditioningReport(Record):
    run: RunResult
    manifest: FileArtifact
    splits: tuple[SplitDiagnostics, ...]


CONTROL_PAIRS = (
    (EvaluationCondition.SPEECH, EvaluationCondition.SHUFFLED_SPEECH),
    (EvaluationCondition.SPEECH, EvaluationCondition.ZERO_SPEECH),
    (EvaluationCondition.SPEECH_NO_HISTORY, EvaluationCondition.SHUFFLED_SPEECH_NO_HISTORY),
)


def completed_lines(path: Path) -> tuple[bytes, ...]:
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        raise ValueError(f"Read-only analysis requires a completed journal: {path}")
    return tuple(content.splitlines())


def read_records(path: Path, record_type: type[TRecord]) -> tuple[TRecord, ...]:
    return tuple(record_type.model_validate_json(line) for line in completed_lines(path))


def artifact(path: Path) -> FileArtifact:
    content = path.read_bytes()
    return FileArtifact(
        path=path, source_path=path, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()
    )


def interval(observations: Sequence[PairedMetricObservation]) -> MarginEstimate:
    dialogues = len({item.dialogue_id for item in observations})
    if dialogues < 2:
        return InsufficientDialogues(examples=len(observations), dialogues=dialogues)
    return EstimatedMargin(interval=paired_dialogue_bootstrap(observations))


def pair_records(
    records: Sequence[TObservation],
    correct: EvaluationCondition,
    control: EvaluationCondition,
) -> tuple[tuple[TObservation, TObservation], ...]:
    controls = tuple(item for item in records if item.condition == control)
    correct_by_id = {item.example_id: item for item in records if item.condition == correct}
    pairs: list[tuple[TObservation, TObservation]] = []
    for wrong in controls:
        if wrong.example_id not in correct_by_id:
            raise ValueError("Control record has no paired correct input")
        right = correct_by_id[wrong.example_id]
        if (right.dialogue_id, right.target_tokens) != (wrong.dialogue_id, wrong.target_tokens):
            raise ValueError("Control records differ in dialogue or teacher target length")
        pairs.append((right, wrong))
    return tuple(pairs)


def loss_margin(
    records: Sequence[ExampleLoss], correct: EvaluationCondition, control: EvaluationCondition
) -> LossMargin:
    pairs = pair_records(records, correct, control)
    ordered: list[ExampleLoss] = []
    observations: list[PairedMetricObservation] = []
    weighted: list[PairedMetricObservation] = []
    for right, wrong in pairs:
        ordered.extend((right, wrong))
        difference = wrong.cross_entropy - right.cross_entropy
        observations.append(
            PairedMetricObservation(right.example_id, right.dialogue_id, difference)
        )
        weighted.append(
            PairedMetricObservation(
                right.example_id, right.dialogue_id, difference, right.target_tokens
            )
        )
    return LossMargin(
        diagnostic=conditioning_diagnostic(ordered, correct, control),
        example_weighted=interval(observations),
        token_weighted=interval(weighted),
    )


def fidelity_value(item: TeacherFidelity, metric: FidelityMetric) -> tuple[float, int]:
    match metric:
        case FidelityMetric.CROSS_ENTROPY:
            return -item.input_cross_entropy, item.target_tokens
        case FidelityMetric.KL:
            return -item.teacher_to_input_kl, item.target_tokens
        case FidelityMetric.AGREEMENT:
            return item.top1_agreement, item.target_tokens
        case FidelityMetric.TARGET_ACCURACY:
            return item.input_target_accuracy, item.target_tokens
        case FidelityMetric.FIRST_TOKEN_AGREEMENT:
            return item.first_token_agreement, 1
        case FidelityMetric.FIRST_TOKEN_ACCURACY:
            return item.first_token_target_accuracy, 1
        case FidelityMetric.FIRST_8_AGREEMENT:
            return item.first_8_token_agreement, item.first_8_tokens
        case FidelityMetric.FIRST_8_ACCURACY:
            return item.first_8_token_target_accuracy, item.first_8_tokens


def fidelity_margin(
    records: Sequence[TeacherFidelity],
    correct: EvaluationCondition,
    control: EvaluationCondition,
    metric: FidelityMetric,
) -> FidelityMargin:
    observations: list[PairedMetricObservation] = []
    for right, wrong in pair_records(records, correct, control):
        right_value, weight = fidelity_value(right, metric)
        wrong_value, wrong_weight = fidelity_value(wrong, metric)
        assert weight == wrong_weight
        observations.append(
            PairedMetricObservation(
                right.example_id, right.dialogue_id, right_value - wrong_value, weight
            )
        )
    return FidelityMargin(
        correct_condition=correct,
        control_condition=control,
        metric=metric,
        estimate=interval(observations),
    )


def summarize_stratum(
    examples: Sequence[Example], losses: Sequence[ExampleLoss], fidelity: Sequence[TeacherFidelity]
) -> StratumDiagnostics:
    identifiers = {item.example_id for item in examples}
    losses = tuple(item for item in losses if item.example_id in identifiers)
    fidelity = tuple(item for item in fidelity if item.example_id in identifiers)
    summarized_losses: list[ConditionLoss] = []
    for condition in dict.fromkeys(item.condition for item in losses):
        selected = tuple(item for item in losses if item.condition == condition)
        cross_entropy, perplexity, tokens = summarize_losses(
            tuple(
                LossObservation(item.cross_entropy * item.target_tokens, item.target_tokens)
                for item in selected
            )
        )
        summarized_losses.append(
            ConditionLoss(
                condition=condition,
                metrics=EvaluationMetrics(
                    examples=len(selected),
                    target_tokens=tokens,
                    cross_entropy=cross_entropy,
                    perplexity=perplexity,
                ),
            )
        )
    loss_controls = {item.condition for item in losses}
    fidelity_controls = {item.condition for item in fidelity}
    return StratumDiagnostics(
        examples=len(examples),
        losses=tuple(summarized_losses),
        fidelity=tuple(
            summarize_fidelity(fidelity, condition)
            for condition in dict.fromkeys(item.condition for item in fidelity)
        ),
        loss_margins=tuple(
            loss_margin(losses, correct, control)
            for correct, control in CONTROL_PAIRS
            if control in loss_controls
        ),
        fidelity_margins=tuple(
            fidelity_margin(fidelity, correct, control, metric)
            for correct, control in CONTROL_PAIRS
            if control in fidelity_controls
            for metric in FidelityMetric
        ),
    )


def validate_records(examples: Sequence[Example], records: Sequence[TObservation]) -> None:
    by_id = {item.example_id: item for item in examples}
    if len(by_id) != len(examples):
        raise ValueError("Selected manifest repeats example IDs")
    keys = {(item.example_id, item.condition) for item in records}
    if len(keys) != len(records):
        raise ValueError("Saved metrics repeat an example/condition")
    for item in records:
        if item.example_id not in by_id or item.dialogue_id != by_id[item.example_id].dialogue_id:
            raise ValueError("Saved metrics differ from selected manifest identities")
    primary = {item.example_id for item in records if item.condition == EvaluationCondition.SPEECH}
    if primary != set(by_id):
        raise ValueError("Saved primary metrics do not cover the entire selected split")


def analyze_split(
    directory: Path, split: Split, examples: Sequence[SerializedExample]
) -> SplitDiagnostics:
    records = tuple(item.example for item in examples)
    loss_path = directory / "evaluation_losses.jsonl"
    fidelity_path = directory / "teacher_fidelity.jsonl"
    provenance_path = directory / "teacher_fidelity_provenance.json"
    provenance = FidelityProvenance.model_validate_json(provenance_path.read_bytes())
    manifest_digest = hashlib.sha256(b"".join(item.line + b"\n" for item in examples)).hexdigest()
    if manifest_digest != provenance.examples_sha256:
        raise ValueError("Fidelity provenance teacher examples SHA differs from manifest")
    losses = read_records(loss_path, ExampleLoss)
    fidelity = read_records(fidelity_path, TeacherFidelity)
    validate_records(records, losses)
    validate_records(records, fidelity)
    expected = {item.example_id for item in records[: provenance.config.conditioning_examples]}
    for condition in (
        EvaluationCondition.SHUFFLED_SPEECH,
        EvaluationCondition.ZERO_SPEECH,
        EvaluationCondition.SPEECH_NO_HISTORY,
        EvaluationCondition.SHUFFLED_SPEECH_NO_HISTORY,
    ):
        if {item.example_id for item in losses if item.condition == condition} != expected:
            raise ValueError("Saved CE control journal is incomplete or differently selected")
    for condition in (EvaluationCondition.SHUFFLED_SPEECH, EvaluationCondition.ZERO_SPEECH):
        actual = {item.example_id for item in fidelity if item.condition == condition}
        if actual != expected:
            raise ValueError(
                "Teacher fidelity control journal is incomplete or differently selected"
            )
    return SplitDiagnostics(
        split=split,
        provenance=provenance,
        inputs=tuple(artifact(path) for path in (loss_path, fidelity_path, provenance_path)),
        overall=summarize_stratum(records, losses, fidelity),
        history=tuple(
            HistoryStratum(
                history=group,
                diagnostics=summarize_stratum(selected, losses, fidelity),
            )
            for group in HistoryGroup
            if (
                selected := tuple(
                    item
                    for item in records
                    if bool(item.history) == (group == HistoryGroup.PRESENT)
                )
            )
        ),
    )


def format_estimate(estimate: MarginEstimate) -> tuple[int, str]:
    match estimate:
        case EstimatedMargin(interval=interval):
            return interval.examples, (
                f"{interval.estimate:+.4f} [{interval.lower:+.4f}, {interval.upper:+.4f}]"
            )
        case InsufficientDialogues(examples=count, dialogues=dialogues):
            return count, f"Not estimable: {dialogues} independent dialogues"


def render_report(report: TeacherConditioningReport) -> str:
    lines = [
        f"# Input sensitivity and natural history strata: {report.run.config.name}",
        "",
        "Natural history strata use the supplied manifest history, distinct from artificial "
        "history-removal controls. CE and full-prefix fidelity summaries weight target tokens. "
        "First-token accuracy is example-weighted; first-eight accuracy weights "
        "available early tokens. "
        "Teacher-forced agreement compares speech and transcript argmax under the same saved "
        "sampled teacher prefix; target accuracy need not equal one for the transcript branch.",
        "",
        "Examples are condition-specific. Control rows cover a fixed subset; full-cohort "
        "and control levels are not directly paired. Gains below use matched example IDs.",
        "",
        "Positive margins favor correct audio. Intervals are 95% dialogue-cluster percentile "
        "bootstrap intervals (2,000 draws), covering sampling but not seed or judge uncertainty. "
        "Wrong-audio states were linearly resized before projection, altering their statistics. "
        "These controls support input sensitivity, not a standalone proof "
        "of semantic understanding.",
        "",
        "| Split | Natural history | Examples | Condition | CE | Excess CE | "
        "Top1 agreement | First-token agreement | First-eight agreement |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for split in report.splits:
        strata = (("all", split.overall),) + tuple(
            (item.history.value, item.diagnostics) for item in split.history
        )
        for name, stratum in strata:
            for fidelity in stratum.fidelity:
                lines.append(
                    f"| {split.split.value} | {name} | {fidelity.examples} | "
                    f"{fidelity.condition.value} | {fidelity.input_cross_entropy:.4f} | "
                    f"{fidelity.excess_cross_entropy:.4f} | {fidelity.top1_agreement:.3f} | "
                    f"{fidelity.first_token_agreement:.3f} | "
                    f"{fidelity.first_8_token_agreement:.3f} |"
                )
    lines.extend(
        [
            "",
            "| Split | Natural history | Correct / control | Metric | "
            "Paired examples | Gain [95% CI] |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for split in report.splits:
        strata = (("all", split.overall),) + tuple(
            (item.history.value, item.diagnostics) for item in split.history
        )
        for name, stratum in strata:
            for margin in stratum.loss_margins:
                for weighting, estimate in (
                    ("example-weighted CE", margin.example_weighted),
                    ("token-weighted CE", margin.token_weighted),
                ):
                    count, value = format_estimate(estimate)
                    diagnostic = margin.diagnostic
                    lines.append(
                        f"| {split.split.value} | {name} | "
                        f"{diagnostic.correct_condition.value} / "
                        f"{diagnostic.control_condition.value} | {weighting} | "
                        f"{count} | {value} |"
                    )
            for margin in stratum.fidelity_margins:
                count, value = format_estimate(margin.estimate)
                lines.append(
                    f"| {split.split.value} | {name} | {margin.correct_condition.value} / "
                    f"{margin.control_condition.value} | {margin.metric.value} | "
                    f"{count} | {value} |"
                )
    return "\n".join(lines) + "\n"


def analyze_run(run_directory: Path, manifest: Path, output: Path) -> TeacherConditioningReport:
    run = RunResult.model_validate_json((run_directory / "result.json").read_bytes())
    entries = tuple(
        SerializedExample(Example.model_validate_json(line), line)
        for line in completed_lines(manifest)
    )
    report = TeacherConditioningReport(
        run=run,
        manifest=artifact(manifest),
        splits=tuple(
            analyze_split(
                run_directory / split.value,
                split,
                tuple(item for item in entries if item.example.split == split)[
                    : run.config.validation_examples
                    if split == Split.VALIDATION
                    else run.config.test_examples
                ],
            )
            for split in (Split.VALIDATION, Split.TEST)
        ),
    )
    if any(split.provenance.config != run.config for split in report.splits):
        raise ValueError("Fidelity provenance config differs from the completed run")
    output.mkdir(parents=True, exist_ok=True)
    (output / "conditioning_strata.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (output / "conditioning_strata.md").write_text(render_report(report), encoding="utf-8")
    return report


def analyze_suite(
    results_root: Path, manifest: Path, output: Path
) -> tuple[TeacherConditioningReport, ...]:
    state = SuiteState.model_validate_json((results_root / "suite_state.json").read_bytes())
    if state.running is not None or state.failed:
        raise ValueError("Analyze the suite after all evaluation writers exit successfully")
    results_path = results_root / "completed_results.json"
    results = TypeAdapter(tuple[RunResult, ...]).validate_json(results_path.read_bytes())
    if not results or len({item.config.name for item in results}) != len(results):
        raise ValueError("Suite analysis requires nonempty uniquely named completed results")
    reports: list[TeacherConditioningReport] = []
    for result in results:
        report = analyze_run(
            results_root / result.config.name, manifest, output / result.config.name
        )
        if report.run != result:
            raise ValueError("Completed-results index differs from its per-run result")
        reports.append(report)
    lines = [
        "# Teacher fidelity and audio conditioning by natural history",
        "",
        "CPU reaggregation from complete saved records; no new inference. Reports include "
        "natural history strata, separate history-removal controls, early-token fidelity, "
        "and paired dialogue-bootstrap intervals. Primary CE is token-weighted; paired "
        "CE reports distinguish example and token weighting.",
        "",
        f"Completed-results SHA256: `{artifact(results_path).sha256}`. "
        f"Teacher manifest SHA256: `{artifact(manifest).sha256}`.",
        "",
        "| Run | Validation / test examples | Diagnostics |",
        "|---|---:|---|",
    ]
    for report in reports:
        lines.append(
            f"| {report.run.config.name} | {report.run.validation_examples} / "
            f"{report.run.test_examples} | "
            f"[History and conditioning]({report.run.config.name}/conditioning_strata.md) |"
        )
    (output / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tuple(reports)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path)
    source.add_argument("--results-root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.results_root is not None:
        analyze_suite(arguments.results_root, arguments.manifest, arguments.output)
    else:
        analyze_run(arguments.run, arguments.manifest, arguments.output)


if __name__ == "__main__":
    main()
