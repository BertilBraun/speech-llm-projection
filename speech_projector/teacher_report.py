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

from scripts.audit_synthesis_alignment import normalize
from scripts.judge_teacher_suite import (
    ControlQualityReport,
    JudgedGenerationSet,
    TeacherJudgingReport,
)
from scripts.package_results import FileArtifact
from scripts.prepare_teacher_data import TeacherDataPreparation
from scripts.summarize_teacher_targets import HistoryGroup, TeacherTargetAudit
from scripts.teacher_asr_quality import AsrObservation, TeacherAsrQuality, aggregate
from speech_projector.cache import CacheStatistics
from speech_projector.data import DatasetReport, Distribution
from speech_projector.evaluation import ConditioningDiagnostic
from speech_projector.models import (
    Architecture,
    AsrTranscript,
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
from speech_projector.teacher import TeacherExport, TeacherFailure
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
    greedy_pilot_failures: tuple[TeacherFailure, ...]
    greedy_pilot_results: tuple[RunResult, ...]
    dataset: DatasetReport
    asr_quality: TeacherAsrQuality
    asr_observations: tuple[AsrObservation, ...]
    control_quality: ControlQualityReport


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


def validate_dataset_evidence(
    dataset: DatasetReport,
    preparation: TeacherDataPreparation,
    examples: Sequence[Example],
) -> None:
    counts = (
        (Split.TRAIN, dataset.train_examples),
        (Split.VALIDATION, dataset.validation_examples),
        (Split.TEST, dataset.test_examples),
    )
    if {item.split for item in preparation.splits} != set(Split) or len(preparation.splits) != 3:
        raise ValueError("Dataset preparation requires exactly one record per split")
    for split, count in counts:
        selected = tuple(item for item in examples if item.split == split)
        prepared = next(item for item in preparation.splits if item.split == split)
        if (
            len(selected) != count
            or prepared.selected_examples != count
            or prepared.distinct_dialogues != len({item.dialogue_id for item in selected})
            or prepared.selected_ids != tuple(item.example_id for item in selected)
        ):
            raise ValueError("Dataset report/selection evidence differs from the source manifest")


def validate_asr_evidence(
    quality: TeacherAsrQuality,
    observations: Sequence[AsrObservation],
    examples: Sequence[Example],
    transcripts: Sequence[AsrTranscript],
) -> None:
    heldout = {item.example_id: item for item in examples if item.split != Split.TRAIN}
    recognized = {item.example_id: item.text for item in transcripts}
    if len(recognized) != len(transcripts):
        raise ValueError("ASR transcripts repeat example IDs")
    if len({item.example_id for item in observations}) != len(observations):
        raise ValueError("ASR observations repeat example IDs")
    if {item.example_id for item in observations} != set(heldout):
        raise ValueError("ASR observations do not cover the exact final heldout examples")
    for item in observations:
        example = heldout[item.example_id]
        history = HistoryGroup.PRESENT if example.history else HistoryGroup.EMPTY
        if (
            (item.dialogue_id, item.split, item.domain, item.history)
            != (example.dialogue_id, example.split, example.domain, history)
            or item.reference != normalize(example.user_text)
            or item.example_id not in recognized
            or item.recognized != normalize(recognized[item.example_id])
        ):
            raise ValueError("ASR observation differs from final cleaned input or transcript")
    if len(quality.splits) != 2 or {item.split for item in quality.splits} != {
        Split.VALIDATION,
        Split.TEST,
    }:
        raise ValueError("Final ASR quality requires one validation and one test aggregate")
    for summary in quality.splits:
        selected = tuple(item for item in observations if item.split == summary.split)
        if summary.metrics.examples != 512 or aggregate(selected) != summary.metrics:
            raise ValueError("Final ASR aggregate differs from its full512 saved observations")


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
    control_quality = ControlQualityReport.model_validate_json(
        required_content(root / "response_quality/control_quality.json", inputs)
    )
    if control_quality.selection != quality.selection:
        raise ValueError("Main and control judging use different calibrated model selections")
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
    dataset = DatasetReport.model_validate_json(
        required_content(configuration.data_root / "dataset_report.json", inputs)
    )
    validate_dataset_evidence(dataset, preparation, source_examples)
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
    asr_quality = TeacherAsrQuality.model_validate_json(
        required_content(root / "dataset/asr_quality.json", inputs)
    )
    transcripts_content = required_content(
        configuration.data_root / "asr_transcripts.jsonl", inputs
    )
    if (
        asr_quality.source_manifest.sha256 != hashlib.sha256(manifest_content).hexdigest()
        or asr_quality.asr_transcripts.sha256 != hashlib.sha256(transcripts_content).hexdigest()
    ):
        raise ValueError("Final ASR audit manifest/transcript provenance SHA differs")
    observations_content = required_content(root / "dataset/asr_observations.jsonl", inputs)
    if observations_content and not observations_content.endswith(b"\n"):
        raise ValueError("Final ASR observation journal has an incomplete suffix")
    asr_observations = tuple(
        AsrObservation.model_validate_json(line) for line in observations_content.splitlines()
    )
    transcripts = tuple(
        AsrTranscript.model_validate_json(line) for line in transcripts_content.splitlines()
    )
    validate_asr_evidence(asr_quality, asr_observations, examples, transcripts)
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
    pilot_failure_path = root / "pilot_greedy/teacher_targets/failures.jsonl"
    pilot_failures = (
        tuple(
            TeacherFailure.model_validate_json(line)
            for line in required_content(pilot_failure_path, inputs).splitlines()
        )
        if pilot_failure_path.is_file()
        else ()
    )
    pilot_result_path = root / "pilot_greedy/teacher_v0_256_mlp_10hz/result.json"
    pilot_results = (
        (RunResult.model_validate_json(required_content(pilot_result_path, inputs)),)
        if pilot_result_path.is_file()
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
        pilot_failures,
        pilot_results,
        dataset,
        asr_quality,
        asr_observations,
        control_quality,
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


def distribution_row(label: str, measured: Distribution) -> str:
    return (
        f"| {label} | {measured.mean:.3f} | {measured.median:.3f} | "
        f"{measured.p10:.3f} | {measured.p90:.3f} | {measured.p99:.3f} | "
        f"{measured.minimum:.3f}–{measured.maximum:.3f} |"
    )


def render_dataset_evidence(data: TeacherReportData) -> list[str]:
    dataset = data.dataset
    lines = [
        "DeepDialogue XTTS is a synthetic dialogue/TTS corpus, not recorded human "
        "conversation. LLM1 and the next alternating LLM2 supply user/assistant roles. "
        "Dialogue identity includes model_dir/conversation_id; role conventions, repeated "
        "templates, emotional language and roleplay limit transfer to human speech. "
        "Emotion metadata is not a conditioning/training objective here.",
        "",
        f"Source corpus: {dataset.source_rows:,} turns, {dataset.dialogues:,} dialogues, "
        f"{dataset.domains} domains, {dataset.model_pairs} model-pair directories and "
        f"{dataset.usable_pairs:,} source-eligible pairs. These are source-corpus counts, "
        "distinct from the selected dialogues below.",
        "",
        "| Selected split | Examples | Distinct selected dialogues | "
        "Naturally empty / present history |",
        "|---|---:|---:|---:|",
    ]
    for prepared in data.preparation.splits:
        selected = tuple(item for item in data.source_examples if item.split == prepared.split)
        empty = sum(not item.history for item in selected)
        lines.append(
            f"| {prepared.split.value} | {prepared.selected_examples:,} | "
            f"{prepared.distinct_dialogues:,} | {empty:,} / {len(selected) - empty:,} |"
        )
    lines.extend(
        (
            "",
            f"History uses up to {data.preparation.configuration.history_turns} previous turns "
            f"and {data.teacher_audit.provenance.config.run.max_history_tokens} text tokens. "
            "The natural history counts differ from artificial history-removal controls.",
            "",
            "| Distribution | Mean | Median | P10 | P90 | P99 | Min–max |",
            "|---|---:|---:|---:|---:|---:|---:|",
            distribution_row("Selected audio seconds", dataset.duration),
            distribution_row("Selected cleaned user words", dataset.user_words),
            distribution_row("Selected original-response words (provenance)", dataset.target_words),
            distribution_row("Source-corpus turns per dialogue", dataset.dialogue_turns),
            "",
            "Selected audio/user distributions cover all selected train/validation/test inputs. "
            "Source dialogue lengths cover the original corpus. Original response words are "
            "provenance, not sampled teacher-target lengths.",
            "",
            dataset.split_method,
            "",
            "Source-stage filter counts: "
            f"missing audio {dataset.filters.missing_audio:,}; "
            f"invalid duration {dataset.filters.invalid_duration:,}; "
            f"empty/too-short text {dataset.filters.empty_text:,}; "
            f"nonalternating roles {dataset.filters.nonalternating:,}; "
            f"cross-split duplicate pairs {dataset.filters.cross_split_duplicate_pair:,}.",
            "",
            "| Later candidate pool | Candidates | Material turn/audio-text mismatches | "
            "Lexical synthesis substitutions | Missing synthesis text | "
            "Excluded whole collision dialogues |",
            "|---|---:|---:|---:|---:|---:|",
        )
    )
    for prepared in data.preparation.splits:
        lines.append(
            f"| {prepared.split.value} | {prepared.candidate_examples:,} | "
            f"{prepared.material_alignment_exclusions:,} | "
            f"{prepared.lexical_substitution_exclusions:,} | "
            f"{prepared.missing_synthesis_text_exclusions:,} | "
            f"{len(prepared.collision_dialogue_exclusions):,} |"
        )
    lines.extend(
        (
            "",
            "Material-mismatch and substitution counters can overlap. Whole-dialogue collision "
            "counts describe candidate-pool dialogues, not that many selected examples. "
            "Source eligibility and later cleaning are different stages; their counts must "
            "not be summed as disjoint selected-example removals. Formatting-only cleanup "
            "is retained. Synthesis text remains a metadata reference, not independently "
            "verified waveform contents.",
            "",
        )
    )
    return lines


def markdown_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def render_asr_evidence(data: TeacherReportData) -> list[str]:
    lines = [
        "## Heldout transcription quality",
        "",
        data.asr_quality.normalization,
        "",
        data.asr_quality.interpretation,
        "",
        "WER is pooled edit count / reference words, not mean per-clip WER. "
        "Insertions can make an individual clip's WER exceed 100%. This measures lexical "
        "transcription against documented synthesis input, separately from response quality.",
        "",
        "| Split | Examples | Reference words | Pooled WER% | "
        "Substitutions / deletions / insertions |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in data.asr_quality.splits:
        metrics = summary.metrics
        lines.append(
            f"| {summary.split.value} | {metrics.examples} | {metrics.reference_words:,} | "
            f"{100 * metrics.word_error_rate:.3f} | {metrics.edits.substitutions} / "
            f"{metrics.edits.deletions} / {metrics.edits.insertions} |"
        )
    lines.extend(
        (
            "",
            "Worst three saved ASR disagreements per split (ties ordered by example ID):",
            "",
            "| Split / example ID | Domain | Clip WER% | S / D / I | "
            "Cleaned synthesis reference | Recognized text |",
            "|---|---|---:|---:|---|---|",
        )
    )
    for split in (Split.VALIDATION, Split.TEST):
        worst = sorted(
            (item for item in data.asr_observations if item.split == split),
            key=lambda item: (-item.word_error_rate, item.example_id),
        )[:3]
        for item in worst:
            lines.append(
                f"| {item.split.value}/{item.example_id} | {markdown_cell(item.domain)} | "
                f"{100 * item.word_error_rate:.1f} | {item.edits.substitutions} / "
                f"{item.edits.deletions} / {item.edits.insertions} | "
                f"{markdown_cell(item.reference)} | {markdown_cell(item.recognized)} |"
            )
    lines.extend(
        (
            "",
            "A large disagreement can reflect transcription error, ASR hallucination or "
            "synthesized speech that differs from metadata. This table alone cannot determine "
            "which occurred. The fixed main heldout sets remain unchanged; any outlier "
            "sensitivity must be labeled separately. No listening or independent waveform "
            "transcript certification is implied.",
            "",
        )
    )
    return lines


def render_resource_usage(data: TeacherReportData) -> list[str]:
    runs = validation_runs(data)
    progress = data.teacher_audit.progress[-1]
    training_seconds = sum(item.result.runtime_seconds for item in runs)
    evaluation_seconds = 0.0
    for item in runs:
        assert item.result.test is not None
        evaluation_seconds += (
            item.result.validation.evaluation_seconds + item.result.test.evaluation_seconds
        )
    baseline_seconds = sum(metrics.evaluation_seconds for _, _, metrics in data.baselines)
    judge_seconds = data.quality.seconds + data.control_quality.seconds
    lines = [
        f"Teacher cumulative generation: {progress.elapsed_seconds / 3600:.2f} code wall hours, "
        f"{progress.examples_per_second:.2f} examples/s and {progress.tokens_per_second:.1f} "
        f"tokens/s; peak PyTorch allocated memory: {progress.peak_vram_gb:.2f} decimal GB. "
        "This cumulative timer includes the bootstrap; it is counted once.",
        "",
        f"Training loops summed once per run: {training_seconds / 3600:.2f} wall hours, "
        "including periodic validation/persistence inside the training timer; "
        f"peak PyTorch allocated training memory: "
        f"{max(item.result.peak_vram_gb for item in runs):.2f} decimal GB. "
        f"Separate final validation/test evaluation: {evaluation_seconds / 3600:.2f} hours; "
        f"baseline evaluation: {baseline_seconds / 3600:.2f} hours. Generation, controls "
        "and semantic work are nested in evaluation timers and are not added again.",
        "",
        f"Selected cached inputs: {data.cache.feature_count:,} sequences, "
        f"{data.cache.feature_bytes / 1e9:.3f} "
        f"decimal GB; {data.cache.extracted_count:,} newly extracted and "
        f"{data.cache.feature_count - data.cache.extracted_count:,} reused. Incremental cache "
        f"pipeline: {data.cache.cache_wall_seconds:.1f} s; encoder calls "
        f"{data.cache.extraction_seconds:.1f} s and ASR {data.cache.asr_seconds:.1f} s are "
        "nested/component timings, not additional whole-dataset extraction costs. "
        f"Peak PyTorch allocated cache memory: {data.cache.peak_vram_gb:.2f} decimal GB.",
        "",
        f"Main response judging: {data.quality.seconds / 3600:.3f} hours; subsequent small "
        f"control judging: {data.control_quality.seconds / 3600:.3f} hours; additive total "
        f"{judge_seconds / 3600:.3f} hours. Per-set judgment timers overlap these enclosing "
        "timers and are not added again. Main judge peak PyTorch allocated memory: "
        f"{data.quality.peak_vram_gb:.2f} decimal GB. Calibration/challenge attempts and "
        "model startup/downloads are additional costs outside these judgment timers.",
        "",
        "Teacher-prefix fidelity/KL rescoring has no elapsed-duration field and remains "
        "unmeasured here. These phase measurements exclude additional startup, probes, "
        "debugging and some gaps; they are not complete GPU-busy hours or provider billing. "
        "PyTorch allocated decimal GB is not NVIDIA-observed physical device usage or free "
        "memory: allocator reservations, CUDA contexts and library workspaces can differ. "
        "Point observations belong to the separately scoped operational addendum.",
    ]
    if data.greedy_pilot_results:
        lines.append(
            f"Archived greedy pilot training: {len(data.greedy_pilot_results)} run(s), "
            f"{sum(item.runtime_seconds for item in data.greedy_pilot_results):.1f} seconds; "
            "excluded from the sampled program's training total and comparisons."
        )
    return lines


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
        *render_dataset_evidence(data),
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
            *render_asr_evidence(data),
            "## Resource usage",
            "",
            *render_resource_usage(data),
        )
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
            "controls assess whether the current audio changes predictions.",
            "",
            "Wrong-audio encoder states are linearly resized to the correct native state "
            "length before projection, holding the pseudo-token count fixed. This alters "
            "feature statistics; measured input sensitivity is not a standalone proof "
            "of semantic understanding.",
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
        lines.append("No failed sampled suite training attempts were recorded.")
    lines.append("")
    lines.append("Judge candidate attempts (both original calibration and independent challenge):")
    lines.extend(
        f"- {attempt.calibration.config.model_name}: "
        f"original passed={attempt.calibration.passed}; "
        f"independent passed={attempt.independent_challenge.passed}."
        for attempt in data.quality.selection.attempts
    )
    if data.greedy_pilot_failures:
        lines.extend(
            (
                "",
                f"Archived greedy teacher failures: {len(data.greedy_pilot_failures)} raw "
                "records in pilot_greedy/teacher_targets/failures.jsonl. "
                "These attempts are excluded from the completed sampled target journal.",
            )
        )
        for failure in data.greedy_pilot_failures:
            attempt_counts = ", ".join(
                f"{attempt.kind.value}: {len(attempt.token_ids)} tokens"
                for attempt in failure.attempts
            )
            lines.append(
                f"- {failure.example.example_id}: {failure.reason.value}; "
                f"attempts [{attempt_counts}]. User: "
                f"{failure.example.user_text.replace(chr(10), ' ')}"
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
            "History-specific teacher fidelity and paired audio-conditioning intervals "
            "are in [the conditioning reports](conditioning_strata/index.md). "
            "Small generated-response audio controls are in "
            "[the control-quality report](../response_quality/control_quality.md).",
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
