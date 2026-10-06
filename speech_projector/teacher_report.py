"""Complete teacher-distillation results, fidelity curves, and fixed paired responses."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as pyplot
from pydantic import TypeAdapter

from scripts.judge_teacher_suite import JudgedGenerationSet, TeacherJudgingReport
from scripts.package_results import FileArtifact
from scripts.prepare_teacher_data import TeacherDataPreparation
from scripts.summarize_teacher_targets import TeacherTargetAudit
from speech_projector.cache import CacheStatistics
from speech_projector.evaluation import ConditioningDiagnostic
from speech_projector.models import (
    Architecture,
    ChatPromptConfig,
    EvaluationCondition,
    EvaluationMetrics,
    Example,
    ExperimentFailure,
    ExperimentStage,
    Record,
    RunResult,
    SampleGeneration,
    SamplingDecodingConfig,
    Split,
    SuiteState,
)
from speech_projector.teacher import TeacherExport
from speech_projector.teacher_evaluation import FidelitySummary

matplotlib.use("Agg")


class TeacherReportConfig(Record):
    results_root: Path
    data_root: Path
    output: Path


class TeacherReportProvenance(Record):
    configuration: TeacherReportConfig
    inputs: tuple[FileArtifact, ...]


@dataclass(frozen=True)
class RunAnalysis:
    result: RunResult
    split: Split
    fidelity: tuple[FidelitySummary, ...]
    judged: JudgedGenerationSet
    conditioning: tuple[ConditioningDiagnostic, ...]

    @property
    def speech_fidelity(self) -> FidelitySummary:
        selected = tuple(
            item for item in self.fidelity if item.condition == EvaluationCondition.SPEECH
        )
        if len(selected) != 1:
            raise ValueError("Run analysis requires exactly one speech fidelity summary")
        return selected[0]


@dataclass(frozen=True)
class TeacherReportData:
    configuration: TeacherReportConfig
    runs: tuple[RunAnalysis, ...]
    teacher_audit: TeacherTargetAudit
    preparation: TeacherDataPreparation
    cache: CacheStatistics
    quality: TeacherJudgingReport
    examples: tuple[Example, ...]
    source_examples: tuple[Example, ...]
    samples: tuple[tuple[str, Split, tuple[SampleGeneration, ...]], ...]
    baselines: tuple[tuple[str, Split, EvaluationMetrics], ...]
    failures: tuple[ExperimentFailure, ...]
    provenance: TeacherReportProvenance


def required_content(path: Path, inputs: list[FileArtifact]) -> bytes:
    if not path.is_file():
        raise ValueError(f"Required completed report input is missing: {path}")
    content = path.read_bytes()
    inputs.append(
        FileArtifact(
            path=path,
            source_path=path,
            bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
    )
    return content


def judged_set(report: TeacherJudgingReport, name: str, split: Split) -> JudgedGenerationSet:
    selected = tuple(item for item in report.sets if (item.name, item.split) == (name, split))
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one calibrated quality set for {name}/{split.value}")
    return selected[0]


def validate_program(results: Sequence[RunResult]) -> None:
    if len({item.config.name for item in results}) != len(results):
        raise ValueError("Completed research program contains duplicate run names")
    if any(item.config.projector.architecture == Architecture.CONV for item in results):
        raise ValueError("The teacher program excludes convolution experiments")
    scaling = {
        item.train_examples
        for item in results
        if item.config.stage == ExperimentStage.V1
        and item.config.projector.architecture == Architecture.MLP
        and item.config.projector.compression_factor == 5
    }
    if scaling != {1000, 3000, 10000, 20000}:
        raise ValueError("Complete V1 teacher scaling requires 1k/3k/10k/20k runs")
    compression = {
        item.config.projector.compression_factor
        for item in results
        if item.train_examples == 20000 and item.config.projector.architecture == Architecture.MLP
    }
    if compression != {2, 5, 10, 20}:
        raise ValueError("Complete V2 teacher compression requires factors 2/5/10/20")
    if not any(item.config.stage == ExperimentStage.V0 for item in results):
        raise ValueError("Completed teacher V0 is missing")
    linear = tuple(
        item for item in results if item.config.projector.architecture == Architecture.LINEAR
    )
    if len(linear) != 1 or linear[0].config.stage != ExperimentStage.V3:
        raise ValueError("Complete V3 requires exactly one linear comparison")
    for result in results:
        match result.config.prompt, result.config.decoding:
            case ChatPromptConfig(), SamplingDecodingConfig():
                pass
            case _:
                raise ValueError("The final teacher report requires native chat and sampled runs")
        expected = 32 if result.config.stage == ExperimentStage.V0 else 512
        if result.validation_examples != expected or result.validation.examples != expected:
            raise ValueError(f"Incomplete validation coverage for {result.config.name}")
        if (
            result.test is None
            or result.test_examples != expected
            or result.test.examples != expected
        ):
            raise ValueError(f"Incomplete test coverage for {result.config.name}")


def load_teacher_report(configuration: TeacherReportConfig) -> TeacherReportData:
    inputs: list[FileArtifact] = []
    root = configuration.results_root
    state = SuiteState.model_validate_json(required_content(root / "suite_state.json", inputs))
    if state.running is not None or state.failed:
        raise ValueError("Research report requires the completed, successful training suite")
    results = TypeAdapter(tuple[RunResult, ...]).validate_json(
        required_content(root / "completed_results.json", inputs)
    )
    validate_program(results)
    if set(state.completed) != {item.config.name for item in results}:
        raise ValueError("Completed suite state and result inventory disagree")
    quality = TeacherJudgingReport.model_validate_json(
        required_content(root / "response_quality/quality_report.json", inputs)
    )
    if not any(
        attempt.passed
        and attempt.calibration.config == quality.selection.selected
        and attempt.independent_challenge.config == quality.selection.selected
        for attempt in quality.selection.attempts
    ):
        raise ValueError("Quality scores require a selected judge that passed calibration")
    preparation = TeacherDataPreparation.model_validate_json(
        required_content(configuration.data_root / "preparation.json", inputs)
    )
    audit = TeacherTargetAudit.model_validate_json(
        required_content(root / "teacher_targets/target_audit.json", inputs)
    )
    if (
        audit.completed_examples != 21024
        or audit.eos_completed_examples != 21024
        or audit.issues
        or audit.incomplete_suffix_bytes
        or not audit.progress
        or audit.progress[-1].completed_examples != 21024
        or audit.source_manifest.sha256 != preparation.source_manifest_sha256
    ):
        raise ValueError("Full teacher target audit is incomplete or inconsistent")
    cache = CacheStatistics.model_validate_json(
        required_content(configuration.data_root / "cache_stats_20000.json", inputs)
    )
    journal_content = required_content(root / "teacher_targets/targets.jsonl", inputs)
    if hashlib.sha256(journal_content).hexdigest() != audit.input_journal.sha256:
        raise ValueError("Teacher journal changed after its completed target audit")
    source_content = required_content(configuration.data_root / "examples_source.jsonl", inputs)
    if hashlib.sha256(source_content).hexdigest() != preparation.source_manifest_sha256:
        raise ValueError("Teacher source manifest differs from the clean preparation")
    source_examples = tuple(
        Example.model_validate_json(line) for line in source_content.splitlines()
    )
    manifest_content = required_content(configuration.data_root / "teacher_examples.jsonl", inputs)
    examples = tuple(Example.model_validate_json(line) for line in manifest_content.splitlines())
    if len(examples) != 21024 or len({item.example_id for item in examples}) != len(examples):
        raise ValueError("Final teacher manifest requires 21024 unique examples")
    exported = TeacherExport.model_validate_json(
        required_content(configuration.data_root / "teacher_examples.provenance.json", inputs)
    )
    if (
        exported.examples != 21024
        or exported.manifest.sha256 != hashlib.sha256(manifest_content).hexdigest()
    ):
        raise ValueError("Final teacher manifest differs from its completed export provenance")
    runs: list[RunAnalysis] = []
    samples: list[tuple[str, Split, tuple[SampleGeneration, ...]]] = []
    for result in results:
        required_content(root / result.config.name / "checkpoint/projector.safetensors", inputs)
        for split in (Split.VALIDATION, Split.TEST):
            directory = root / result.config.name / split.value
            fidelity = tuple(
                FidelitySummary.model_validate_json(line)
                for line in required_content(
                    directory / "teacher_fidelity_summary.jsonl", inputs
                ).splitlines()
            )
            judged = judged_set(quality, result.config.name, split)
            conditioning = TypeAdapter(tuple[ConditioningDiagnostic, ...]).validate_json(
                required_content(directory / "evaluation_conditioning.json", inputs)
            )
            analysis = RunAnalysis(result, split, fidelity, judged, conditioning)
            expected = (
                result.validation_examples if split == Split.VALIDATION else result.test_examples
            )
            if analysis.speech_fidelity.examples != expected:
                raise ValueError(f"Incomplete fidelity for {result.config.name}/{split.value}")
            if judged.summary.requested_examples != result.config.semantic_examples:
                raise ValueError(f"Incomplete judging for {result.config.name}/{split.value}")
            runs.append(analysis)
            generation_content = required_content(
                directory / "evaluation_generations.jsonl", inputs
            )
            if hashlib.sha256(generation_content).hexdigest() != judged.inputs.generations.sha256:
                raise ValueError("Run generations changed after independent judging")
            samples.append(
                (
                    result.config.name,
                    split,
                    tuple(
                        SampleGeneration.model_validate_json(line)
                        for line in generation_content.splitlines()
                    ),
                )
            )
    baselines: list[tuple[str, Split, EvaluationMetrics]] = []
    for name in ("text", "asr"):
        for split in (Split.VALIDATION, Split.TEST):
            directory = root / "baseline" / split.value
            metrics = EvaluationMetrics.model_validate_json(
                required_content(directory / f"{name}.json", inputs)
            )
            if (
                metrics.examples != 512
                or judged_set(quality, name, split).summary.requested_examples != 512
            ):
                raise ValueError(f"Incomplete baseline coverage for {name}/{split.value}")
            baselines.append((name, split, metrics))
            generation_content = required_content(directory / f"{name}_generations.jsonl", inputs)
            judged = judged_set(quality, name, split)
            if hashlib.sha256(generation_content).hexdigest() != judged.inputs.generations.sha256:
                raise ValueError("Baseline generations changed after independent judging")
            samples.append(
                (
                    name,
                    split,
                    tuple(
                        SampleGeneration.model_validate_json(line)
                        for line in generation_content.splitlines()
                    ),
                )
            )
    failures_path = root / "failures.jsonl"
    failures = (
        tuple(
            ExperimentFailure.model_validate_json(line)
            for line in required_content(failures_path, inputs).splitlines()
        )
        if failures_path.is_file()
        else ()
    )
    return TeacherReportData(
        configuration,
        tuple(runs),
        audit,
        preparation,
        cache,
        quality,
        examples,
        source_examples,
        tuple(samples),
        tuple(baselines),
        failures,
        TeacherReportProvenance(configuration=configuration, inputs=tuple(inputs)),
    )


def validation_runs(data: TeacherReportData) -> tuple[RunAnalysis, ...]:
    return tuple(item for item in data.runs if item.split == Split.VALIDATION)


def best_run(data: TeacherReportData) -> RunAnalysis:
    eligible = tuple(item for item in validation_runs(data) if item.result.train_examples == 20000)
    return max(
        eligible,
        key=lambda item: (
            item.judged.summary.acceptable_rate_requested,
            item.speech_fidelity.first_8_token_agreement,
            -item.result.validation.cross_entropy,
        ),
    )


def run_table(rows: Sequence[RunAnalysis]) -> list[str]:
    lines = [
        "| Run | N | Parameters | Audio tokens/s | CE | Teacher top1% | First1% | First8% | "
        "Appropriate% | Capped/unknown | Hours |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in rows:
        result, fidelity, judged = item.result, item.speech_fidelity, item.judged
        metrics = result.validation if item.split == Split.VALIDATION else result.test
        assert metrics is not None
        lines.append(
            f"| {result.config.name} | {result.train_examples:,} | "
            f"{result.projector_parameters:,} | "
            f"{result.pseudo_tokens_per_second:g} | {metrics.cross_entropy:.4f} | "
            f"{100 * fidelity.top1_agreement:.1f} | {100 * fidelity.first_token_agreement:.1f} | "
            f"{100 * fidelity.first_8_token_agreement:.1f} | "
            f"{100 * judged.summary.acceptable_rate_requested:.1f} | "
            f"{judged.generation_caps.capped}/{judged.generation_caps.unknown} | "
            f"{result.runtime_seconds / 3600:.2f} |"
        )
    return lines


def stage_rows(data: TeacherReportData, stage: ExperimentStage) -> tuple[RunAnalysis, ...]:
    rows = validation_runs(data)
    match stage:
        case ExperimentStage.V1:
            return tuple(
                sorted(
                    (item for item in rows if item.result.config.stage == stage),
                    key=lambda item: item.result.train_examples,
                )
            )
        case ExperimentStage.V2:
            return tuple(
                sorted(
                    (
                        item
                        for item in rows
                        if item.result.train_examples == 20000
                        and item.result.config.projector.architecture == Architecture.MLP
                    ),
                    key=lambda item: item.result.pseudo_tokens_per_second,
                )
            )
        case ExperimentStage.V3:
            linear = next(
                item
                for item in rows
                if item.result.config.projector.architecture == Architecture.LINEAR
            )
            return tuple(
                item
                for item in rows
                if item.result.train_examples == 20000
                and item.result.config.projector.compression_factor
                == linear.result.config.projector.compression_factor
            )
        case ExperimentStage.V0:
            return tuple(item for item in rows if item.result.config.stage == stage)


def render_teacher_report(data: TeacherReportData) -> str:
    best = best_run(data)
    lengths = data.teacher_audit.teacher_response.target_tokens_with_eos
    source_lengths = data.teacher_audit.original_dataset_response.target_tokens_with_eos
    lines = [
        "# Speech projector: native Qwen teacher experiment",
        "",
        f"Best validation configuration: **{best.result.config.name}**, selected by "
        "judged appropriate-response rate, then first-eight-token fidelity and CE. "
        "Test results are held out from this selection.",
        "",
        "Whisper Small final encoder states (768 dimensions, 50 Hz, BF16) feed temporal "
        "pooling and a trainable projector into frozen Qwen3.5-2B. Only the current spoken "
        "utterance enters through speech embeddings; previous dialogue is text. Teacher "
        "targets are complete native-chat Qwen responses to the cleaned true transcript "
        "and identical history, without a system instruction or additional style prompt.",
        "",
        "Generation uses Qwen's recommended non-thinking text sampling: temperature 1, "
        "top-p 1, top-k 20, min-p 0, presence penalty 2, repetition penalty 1. "
        "[Official model guidance](https://huggingface.co/Qwen/Qwen3.5-2B#best-practices). "
        "Batch seeds are reproducible from seed, ordered example IDs and token cap. "
        "The sampled teacher response is a reference, not independently verified factual gold.",
        "",
        "## Data and teacher targets",
        "",
        f"{data.teacher_audit.completed_examples:,} completed targets: 20,000 train + "
        "512 validation + 512 test, split by dialogue. "
        f"{data.teacher_audit.completed_after_retry} needed a longer retry; all end in EOS "
        "and fit the student target budget. Teacher token lengths including EOS: "
        f"mean {lengths.mean:.1f}, median {lengths.median:.1f}, p90 {lengths.p90:.1f}, "
        f"p99 {lengths.p99:.1f}, max {lengths.maximum:.0f}; "
        f"original source responses mean {source_lengths.mean:.1f}. "
        "Original responses are audit context, never the new training target.",
        "",
        data.preparation.subset_definition,
        "",
        data.preparation.transcript_reference,
        "",
    ]
    for stage in ExperimentStage:
        lines.extend((f"## {stage.value}", "", *run_table(stage_rows(data, stage)), ""))
    lines.extend(
        (
            "## Paired heldout audio conditioning",
            "",
            "Positive control-minus-correct CE means correct audio helps. These are paired "
            "example means on the same conditioning subset, independent of full512 CE.",
            "",
            "| Run/split | Control | Paired examples | CE gap | SE | Correct audio wins% |",
            "|---|---|---:|---:|---:|---:|",
        )
    )
    for item in data.runs:
        for diagnostic in item.conditioning:
            if diagnostic.correct_condition == EvaluationCondition.SPEECH:
                lines.append(
                    f"| {item.result.config.name}/{item.split.value} | "
                    f"{diagnostic.control_condition.value} | {diagnostic.paired_examples} | "
                    f"{diagnostic.mean_control_minus_correct_ce:.4f} | "
                    f"{diagnostic.standard_error:.4f} | "
                    f"{100 * diagnostic.fraction_correct_audio_lower_loss:.1f} |"
                )
    lines.extend(
        (
            "## Baselines and heldout test",
            "",
            "| Input/split | CE | Appropriate% | Valid/requested judge verdicts |",
            "|---|---:|---:|---:|",
        )
    )
    for name, split, metrics in data.baselines:
        judged = judged_set(data.quality, name, split)
        lines.append(
            f"| {name}/{split.value} | {metrics.cross_entropy:.4f} | "
            f"{100 * judged.summary.acceptable_rate_requested:.1f} | "
            f"{judged.summary.valid_examples}/{judged.summary.requested_examples} |"
        )
    lines.extend(
        (
            "",
            *run_table(tuple(item for item in data.runs if item.split == Split.TEST)),
            "",
            "## Resource usage",
            "",
        )
    )
    progress = data.teacher_audit.progress[-1]
    training_hours = sum(item.result.runtime_seconds for item in validation_runs(data)) / 3600
    training_peak = max(item.result.peak_vram_gb for item in validation_runs(data))
    lines.append(
        f"Teacher generation: {progress.elapsed_seconds / 3600:.2f} hours, "
        f"{progress.examples_per_second:.2f} examples/s, {progress.tokens_per_second:.1f} "
        f"tokens/s, peak {progress.peak_vram_gb:.2f} GB. Training summed across runs: "
        f"{training_hours:.2f} hours; peak training VRAM: {training_peak:.2f} GB. "
        f"Cache: {data.cache.feature_bytes / 1e9:.3f} GB, {data.cache.extracted_count} new "
        f"extractions, {data.cache.extraction_seconds:.1f} s encoder GPU time. "
        f"Judge: {data.quality.seconds / 3600:.2f} hours, "
        f"peak {data.quality.peak_vram_gb:.2f} GB. These measured stage times exclude some "
        "startup and evaluation overhead; they are not exact GPU utilization or billing."
    )
    lines.extend(
        (
            "",
            "## Interpretation and limitations",
            "",
            "Teacher top1 and first-token/first-eight agreement compare speech and "
            "transcript logits under the same teacher-forced prefix. They measure "
            "conditional fidelity, not free-running success. Overall CE/top1/KL weight "
            "target tokens, so long responses dominate; early-token metrics reduce "
            "dependence on the gold response prefix. Training averages examples within "
            "accumulation groups, whereas heldout CE weights target tokens. Fixed "
            "train-set reductions establish learning; paired heldout shuffled/zero-audio "
            "controls establish conditioning.",
            "",
            "Quality uses independent candidate-only judging by "
            f"{data.quality.selection.selected.model_name}, after calibration. "
            "Appropriate% uses all requested examples as denominator; failed verdicts "
            "do not count as successes. Calibration is limited and does not replace "
            "human review or complete fact checking. Cached text/reference cosine is 1 "
            "by construction. BF16 batching can change near-tie greedy choices; transcript "
            "fidelity rescoring therefore need not be 100%. Capped and unknown-completion "
            "generations remain visible. Sampling targets need not equal teacher argmaxes; "
            "fidelity compares raw teacher/student logits, rather than the sampled token "
            "alone, under their identical forced prefixes.",
            "",
            "V0 uses 32 heldout examples and 16 primary generations, so its generalization "
            "estimates are preliminary. Compression runs hold data/epochs/LR fixed; "
            "data scaling changes update count as well as data exposure. The speech "
            "encoder and text model stay frozen; no emotional training, LoRA, streaming "
            "or speech output is included.",
            "",
            "## Failures",
            "",
        )
    )
    if data.failures:
        lines.extend(
            f"- {item.run_name}, attempt {item.attempt}: {item.exception_type}: {item.message}"
            for item in data.failures
        )
    else:
        lines.append(
            "No failed suite training attempts were recorded. Earlier preparation/generation "
            "failures and prototype artifacts are retained separately; this does not imply "
            "they never occurred."
        )
    lines.extend(
        (
            "",
            "## Recommended next experiment",
            "",
            "Inspect the fixed paired audio/response set first. Replicate the best "
            "configuration with a second seed and a heldout dialogue/domain slice, "
            "prioritizing semantic and factual errors rather than further interface complexity.",
            "",
            "See qualitative_comparisons.md for the fixed first 8 validation + first 8 test "
            "cases and report_inputs.json for exact input hashes.",
            "",
        )
    )
    return "\n".join(lines)


def selected_sample(data: TeacherReportData, name: str, example: Example) -> SampleGeneration:
    matches = tuple(
        sample
        for run, split, samples in data.samples
        if run == name and split == example.split
        for sample in samples
        if sample.example_id == example.example_id
        and sample.condition
        in (EvaluationCondition.TEXT, EvaluationCondition.ASR, EvaluationCondition.SPEECH)
    )
    if len(matches) != 1:
        raise ValueError(f"Fixed paired sample missing/duplicated:{name}/{example.example_id}")
    return matches[0]


def render_qualitative(data: TeacherReportData) -> str:
    best = best_run(data).result.config.name
    weaker = min(
        stage_rows(data, ExperimentStage.V2),
        key=lambda item: item.judged.summary.acceptable_rate_requested,
    ).result.config.name
    lines = [
        "# Fixed paired responses",
        "",
        "Exactly first 8 validation + first 8 test examples in the final teacher manifest; "
        "no cherry-picking.",
        "",
    ]
    for split in (Split.VALIDATION, Split.TEST):
        selected = tuple(item for item in data.examples if item.split == split)[:8]
        if len(selected) != 8:
            raise ValueError("Fixed paired report requires8examples per heldout split")
        for example in selected:
            lines.extend(
                (
                    f"## {split.value}/{example.example_id}",
                    "",
                    f"True user text:{example.user_text}",
                    "",
                )
            )
            original = tuple(
                item for item in data.source_examples if item.example_id == example.example_id
            )
            if len(original) != 1:
                raise ValueError("Fixed audit context requires one original source example")
            lines.extend(
                (f"Original dataset reply (audit context only): {original[0].target_text}", "")
            )
            lines.extend(f"History {turn.role.value}: {turn.text}" for turn in example.history)
            for name in dict.fromkeys(("text", "asr", best, weaker)):
                sample = selected_sample(data, name, example)
                status = (
                    sample.generation.kind.value
                    if sample.generation is not None
                    else "unknown completion"
                )
                if sample.asr_transcript is not None:
                    lines.extend(("", f"ASR recognized transcript:{sample.asr_transcript}"))
                lines.extend(("", f"**{name} ({status}):**", "", sample.generated_response))
            lines.append("")
    return "\n".join(lines)


def save_plots(data: TeacherReportData) -> None:
    for stage, name, xlabel in (
        (ExperimentStage.V1, "teacher_data_scaling.png", "Training examples"),
        (ExperimentStage.V2, "teacher_compression.png", "Speech pseudo-tokens/audio second"),
    ):
        rows = stage_rows(data, stage)
        horizontal = [
            item.result.train_examples
            if stage == ExperimentStage.V1
            else item.result.pseudo_tokens_per_second
            for item in rows
        ]
        figure, axes = pyplot.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        axes[0].plot(
            horizontal,
            [100 * item.judged.summary.acceptable_rate_requested for item in rows],
            marker="o",
        )
        axes[0].set_ylabel("Judged appropriate response (%)")
        for baseline_name, style in (("text", "--"), ("asr", ":")):
            quality = judged_set(data.quality, baseline_name, Split.VALIDATION)
            axes[0].axhline(
                100 * quality.summary.acceptable_rate_requested,
                linestyle=style,
                label=baseline_name,
            )
        axes[0].legend()
        axes[1].plot(
            horizontal,
            [100 * item.speech_fidelity.top1_agreement for item in rows],
            marker="o",
            label="All teacher-prefix tokens",
        )
        axes[1].plot(
            horizontal,
            [100 * item.speech_fidelity.first_8_token_agreement for item in rows],
            marker="s",
            label="First8tokens",
        )
        axes[1].set_ylabel("Teacher top1 agreement (%)")
        axes[1].legend()
        axes[2].plot(
            horizontal, [item.result.validation.cross_entropy for item in rows], marker="o"
        )
        axes[2].set_ylabel("Validation target-token CE")
        for axis in axes:
            axis.set_xlabel(xlabel)
            axis.set_xscale("log")
            axis.grid(alpha=0.3)
        figure.savefig(data.configuration.output / name, dpi=180)
        pyplot.close(figure)
    rows = stage_rows(data, ExperimentStage.V3)
    labels = [item.result.config.projector.architecture.value for item in rows]
    figure, axes = pyplot.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].bar(labels, [100 * item.judged.summary.acceptable_rate_requested for item in rows])
    axes[0].set_ylabel("Judged appropriate response (%)")
    axes[1].bar(labels, [100 * item.speech_fidelity.first_8_token_agreement for item in rows])
    axes[1].set_ylabel("First 8 teacher-prefix tokens agreement (%)")
    figure.savefig(data.configuration.output / "teacher_architectures.png", dpi=180)
    pyplot.close(figure)


def write_teacher_report(configuration: TeacherReportConfig) -> None:
    data = load_teacher_report(configuration)
    report = render_teacher_report(data)
    qualitative = render_qualitative(data)
    configuration.output.mkdir(parents=True, exist_ok=True)
    (configuration.output / "teacher_research_report.md").write_text(report, encoding="utf-8")
    (configuration.output / "qualitative_comparisons.md").write_text(qualitative, encoding="utf-8")
    (configuration.output / "report_inputs.json").write_text(
        data.provenance.model_dump_json(indent=2), encoding="utf-8"
    )
    save_plots(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    write_teacher_report(
        TeacherReportConfig(
            results_root=arguments.results_root,
            data_root=arguments.data_root,
            output=arguments.output,
        )
    )


if __name__ == "__main__":
    main()
