"""Refresh compact research tables, figures, and qualitative comparisons."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as pyplot
from pydantic import TypeAdapter

from speech_projector.cache import CacheStatistics
from speech_projector.data import DatasetReport
from speech_projector.evaluation import ConditioningDiagnostic
from speech_projector.models import (
    EvaluationMetrics,
    ExperimentFailure,
    ExperimentStage,
    RunResult,
    SampleGeneration,
    SuiteState,
)

matplotlib.use("Agg")


def load_results(root: Path) -> tuple[RunResult, ...]:
    return tuple(
        RunResult.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*/result.json"))
    )


def _number(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _run_table(results: Sequence[RunResult]) -> list[str]:
    lines = [
        "| Run | Training examples | Architecture | Parameters | Tokens/s audio | Val CE | PPL | "
        "Semantic cosine | Train examples/s | Peak GB | Hours |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.config.name} | {result.train_examples} | "
            f"{result.config.projector.architecture.value} | {result.projector_parameters:,} | "
            f"{result.pseudo_tokens_per_second:g} | {_number(result.validation.cross_entropy)} | "
            f"{_number(result.validation.perplexity, 2)} | "
            f"{_number(result.validation.semantic_similarity)} | "
            f"{result.examples_per_second:.2f} | "
            f"{result.peak_vram_gb:.2f} | {result.runtime_seconds / 3600:.3f} |"
        )
    return lines


def _stage_results(results: Sequence[RunResult], stage: ExperimentStage) -> list[RunResult]:
    selected = [result for result in results if result.config.stage == stage]
    if not selected or stage not in (ExperimentStage.V2, ExperimentStage.V3):
        return selected
    reference = selected[0]
    baseline_candidates = [
        result
        for result in results
        if result.train_examples == reference.train_examples
        and result.config.projector.architecture.value == "mlp"
        and result.config.epochs == reference.config.epochs
        and result.config.learning_rate == reference.config.learning_rate
    ]
    if stage == ExperimentStage.V3:
        baseline_candidates = [
            result
            for result in baseline_candidates
            if result.config.projector.compression_factor
            == reference.config.projector.compression_factor
        ]
    else:
        baseline_candidates = [
            result for result in baseline_candidates if result.config.stage == ExperimentStage.V1
        ]
    names = {result.config.name for result in selected}
    return selected + [result for result in baseline_candidates if result.config.name not in names]


def _save_csv(results: Sequence[RunResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "run",
                "stage",
                "train_examples",
                "architecture",
                "parameters",
                "compression_factor",
                "pseudo_tokens_per_second",
                "mean_pseudo_tokens",
                "val_ce",
                "val_perplexity",
                "semantic_similarity",
                "runtime_seconds",
                "peak_vram_gb",
                "examples_per_second",
                "target_tokens_per_second",
                "generation_tokens_per_second",
                "shuffled_ce",
                "zero_ce",
                "no_history_ce",
                "no_history_shuffled_ce",
                "git_commit",
            ]
        )
        for result in results:
            metrics = result.validation
            writer.writerow(
                [
                    result.config.name,
                    result.config.stage.value,
                    result.train_examples,
                    result.config.projector.architecture.value,
                    result.projector_parameters,
                    result.config.projector.compression_factor,
                    result.pseudo_tokens_per_second,
                    result.mean_pseudo_tokens,
                    metrics.cross_entropy,
                    metrics.perplexity,
                    metrics.semantic_similarity,
                    result.runtime_seconds,
                    result.peak_vram_gb,
                    result.examples_per_second,
                    result.target_tokens_per_second,
                    metrics.generated_tokens / metrics.generation_seconds
                    if metrics.generation_seconds
                    else None,
                    metrics.shuffled_audio_cross_entropy,
                    metrics.zero_audio_cross_entropy,
                    metrics.no_history_cross_entropy,
                    metrics.no_history_shuffled_cross_entropy,
                    result.git_commit,
                ]
            )


def _save_plots(results: Sequence[RunResult], root: Path) -> None:
    for stage, xlabel, filename in (
        (ExperimentStage.V1, "Training examples", "data_scaling.png"),
        (ExperimentStage.V2, "Speech pseudo-tokens per audio second", "compression_quality.png"),
    ):
        selected = _stage_results(results, stage)
        if not selected:
            continue
        selected.sort(
            key=lambda result: (
                result.train_examples
                if stage == ExperimentStage.V1
                else result.pseudo_tokens_per_second
            )
        )
        horizontal = [
            result.train_examples
            if stage == ExperimentStage.V1
            else result.pseudo_tokens_per_second
            for result in selected
        ]
        figure, axes = pyplot.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].plot(
            horizontal, [result.validation.cross_entropy for result in selected], marker="o"
        )
        axes[0].set_ylabel("Validation assistant-token CE (lower is better)")
        semantic = [
            result for result in selected if result.validation.semantic_similarity is not None
        ]
        axes[1].plot(
            [
                result.train_examples
                if stage == ExperimentStage.V1
                else result.pseudo_tokens_per_second
                for result in semantic
            ],
            [result.validation.semantic_similarity for result in semantic],
            marker="o",
        )
        axes[1].set_ylabel("Response/reference cosine (higher is better)")
        for axis in axes:
            axis.set_xlabel(xlabel)
            axis.set_xscale("log")
            axis.grid(alpha=0.3)
        figure.savefig(root / filename, dpi=160)
        pyplot.close(figure)


def _qualitative(root: Path, results: Sequence[RunResult]) -> None:
    samples: list[tuple[str, SampleGeneration]] = []
    for path in sorted(root.glob("**/*generations.jsonl")):
        if path.name.startswith("initial") or path.parent.name not in ("validation", "test"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                samples.append(
                    (
                        path.parent.relative_to(root).as_posix(),
                        SampleGeneration.model_validate_json(line),
                    )
                )
    if not samples:
        return
    best_name = (
        min(results, key=lambda result: result.validation.cross_entropy).config.name
        if results
        else ""
    )
    selected_ids = list(dict.fromkeys(sample.example_id for _, sample in samples))[:16]
    lines = [
        "# Fixed qualitative comparisons",
        "",
        f"Lowest validation-CE projector: {best_name or '[pending]'}.",
        "",
    ]
    for example_id in selected_ids:
        matching = [(name, sample) for name, sample in samples if sample.example_id == example_id]
        first = matching[0][1]
        history = " / ".join(f"{turn.role.value}: {turn.text}" for turn in first.history)
        lines.extend(
            [
                f"## {example_id}",
                "",
                f"History: {history or '[none]'}",
                "",
                f"User transcript (audit only): {first.user_transcript}",
                "",
                f"Gold assistant: {first.gold_response}",
                "",
            ]
        )
        for name, sample in matching:
            lines.extend(
                [f"**{name} / {sample.condition.value}**: {sample.generated_response}", ""]
            )
            if sample.asr_transcript is not None:
                lines.extend([f"ASR transcript: {sample.asr_transcript}", ""])
    (root / "qualitative_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def aggregate_report(root: Path, data_report: Path | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    results = load_results(root)
    lines = ["# Whisper Small → frozen Qwen3.5-2B research results", ""]
    lines.extend(
        [
            "The speech system uses frozen 768-dimensional Whisper final encoder states at 50 Hz, "
            "duration-trimmed BF16 caches, temporal compression, and a trainable projector into "
            "frozen Qwen input embeddings. Assistant tokens alone receive teacher-forced CE. "
            "Histories use text; the current user transcript is withheld from speech runs.",
            "",
            "BF16 Whisper features occupy about 76,800 bytes per audio second before serialization "
            "overhead. Compression rates are 50 divided by the configured factor. Final-token "
            "boundaries use rounded-up feature groups, so observed token counts differ slightly "
            "from nominal rates.",
            "",
        ]
    )
    if data_report is not None and data_report.exists():
        dataset = DatasetReport.model_validate_json(data_report.read_text(encoding="utf-8"))
        lines.extend(
            [
                "## Dataset",
                "",
                f"{dataset.source_rows:,} source turns, {dataset.dialogues:,} dialogues; "
                f"{dataset.usable_pairs:,} usable conversational pairs. "
                f"Selected train/validation/test: {dataset.train_examples:,}/"
                f"{dataset.validation_examples:,}/{dataset.test_examples:,}. "
                f"Split: {dataset.split_method}.",
                "",
                f"Audio duration median/p90/p99: {dataset.duration.median:.2f}/"
                f"{dataset.duration.p90:.2f}/{dataset.duration.p99:.2f}s. User words median/p90: "
                f"{dataset.user_words.median:.1f}/{dataset.user_words.p90:.1f}; assistant words: "
                f"{dataset.target_words.median:.1f}/{dataset.target_words.p90:.1f}.",
                "",
                "Filtering counts: "
                + dataset.filters.model_dump_json()
                + ". Exact source metadata and sampled dialogues are saved in the dataset report.",
                "",
                "DeepDialogue-xtts audio is synthesized, and dialogues originate from paired text "
                "models. These results concern this conversational synthetic-speech distribution; "
                "they cannot establish generalization to natural human audio.",
                "",
            ]
        )
    for stage, title in (
        (ExperimentStage.V0, "V0 feasibility"),
        (ExperimentStage.V1, "V1 data scaling"),
        (ExperimentStage.V2, "V2 temporal compression"),
        (ExperimentStage.V3, "V3 projector architecture"),
    ):
        selected = _stage_results(results, stage)
        lines.extend([f"## {title}", ""])
        lines.extend(
            _run_table(selected) if selected else ["No completed result is available yet."]
        )
        lines.append("")
        if stage == ExperimentStage.V0:
            for result in selected:
                lines.extend(
                    [
                        f"{result.config.name}: initial validation CE "
                        f"{result.initial_validation_loss:.4f}, final held-out CE "
                        f"{result.validation.cross_entropy:.4f}; final training CE "
                        f"{result.final_training_loss:.4f}. Fixed training-probe CE changed "
                        f"from {result.initial_training_loss:.4f} "
                        f"to {result.final_fixed_training_loss:.4f}. "
                        "Gradient checks and saved outputs "
                        "must be inspected together with loss reduction.",
                        "",
                    ]
                )
    lines.extend(
        [
            "## Audio-conditioning checks",
            "",
            "Controls use a fixed subset of up to 32 held-out examples, with replacement audio "
            "from a different dialogue. Control encoder states are linearly resampled to the "
            "correct input's state length before projection, holding pseudo-token counts fixed. "
            "Compare paired CE files and conditioning confidence "
            "estimates; the full validation CE and subset-control CE have different denominators.",
            "",
            "| Run | Shuffled CE | Zero CE | No-history CE | No-history shuffled CE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        metrics = result.validation
        lines.append(
            f"| {result.config.name} | {_number(metrics.shuffled_audio_cross_entropy)} | "
            f"{_number(metrics.zero_audio_cross_entropy)} | "
            f"{_number(metrics.no_history_cross_entropy)} | "
            f"{_number(metrics.no_history_shuffled_cross_entropy)} |"
        )
    lines.extend(
        [
            "",
            "Paired audio-control margins (positive favors correct audio):",
            "",
            "| Run/split | Correct/control | Pairs | Mean CE margin | SE | Correct audio wins |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        path = root / result.config.name / "validation" / "evaluation_conditioning.json"
        if path.exists():
            diagnostic_adapter = TypeAdapter(tuple[ConditioningDiagnostic, ...])
            for diagnostic in diagnostic_adapter.validate_json(path.read_bytes()):
                lines.append(
                    f"| {result.config.name}/validation | "
                    f"{diagnostic.correct_condition.value}/{diagnostic.control_condition.value} | "
                    f"{diagnostic.paired_examples} | "
                    f"{diagnostic.mean_control_minus_correct_ce:.4f} | "
                    f"{diagnostic.standard_error:.4f} | "
                    f"{diagnostic.fraction_correct_audio_lower_loss:.1%} |"
                )
    lines.extend(
        [
            "",
            "A positive paired control-minus-correct CE suggests useful audio conditioning. "
            "The no-history control checks whether textual history alone explains performance. "
            "Resampling changes the replacement utterance's temporal structure, so this "
            "is a diagnostic, not proof of semantic understanding.",
            "",
            "## Baselines",
            "",
            "| Baseline | Examples | CE | PPL | Semantic cosine |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for path in sorted(root.glob("baseline/**/*.json")):
        if path.stem in ("text", "asr"):
            metrics = EvaluationMetrics.model_validate_json(path.read_text(encoding="utf-8"))
            lines.append(
                f"| {path.parent.name}/{path.stem} | {metrics.examples} | "
                f"{_number(metrics.cross_entropy)} | {_number(metrics.perplexity, 2)} | "
                f"{_number(metrics.semantic_similarity)} |"
            )
    lines.extend(
        [
            "",
            "## Fixed held-out test results",
            "",
            "| Run | Test examples | CE | PPL | Semantic cosine |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        if result.test is not None:
            lines.append(
                f"| {result.config.name} | {result.test.examples} | "
                f"{_number(result.test.cross_entropy)} | {_number(result.test.perplexity, 2)} | "
                f"{_number(result.test.semantic_similarity)} |"
            )
    lines.extend(
        [
            "",
            "Semantic cosine uses the CPU sentence-transformers/all-MiniLM-L6-v2 encoder. "
            "It is an English sentence similarity metric, truncates long inputs, and does not "
            "measure factual correctness or whether the reply is the uniquely appropriate "
            "continuation. Read the fixed generated comparisons alongside CE.",
            "",
            "The transcript baseline is a perfect-input reference, not a mathematical CE upper "
            "bound. The trained projector can also learn the dataset's response style while "
            "the text and ASR baselines use the original frozen model. Lower projector CE "
            "must be interpreted together with correct-versus-shuffled audio margins and "
            "utterance-specific generated replies.",
            "",
            "## Resource usage",
            "",
            f"Summed recorded training-run wall time: "
            f"{sum(result.runtime_seconds for result in results) / 3600:.3f} hours. "
            "Recorded training runtime includes periodic validation but excludes initial "
            "validation and final evaluation. Feature extraction and baseline evaluation "
            "are additional. "
            "Feature cache timing/storage is recorded in the cache report.",
            "",
        ]
    )
    if results:
        best = min(results, key=lambda result: result.validation.cross_entropy)
        lines.extend(
            [
                "## Best measured configuration",
                "",
                f"{best.config.name}: {best.train_examples:,} training pairs; "
                f"{best.config.projector.architecture.value}, {best.projector_parameters:,} "
                f"parameters, {best.pseudo_tokens_per_second:g} nominal tokens/s; "
                f"validation CE {best.validation.cross_entropy:.4f}. "
                "This ranking uses held-out CE, with semantic and qualitative evidence "
                "to inspect in parallel.",
                "",
            ]
        )
    state_path = root / "suite_state.json"
    if state_path.exists():
        state = SuiteState.model_validate_json(state_path.read_bytes())
        lines.extend(
            [
                f"Suite elapsed wall time at last update: "
                f"{(state.updated_at - state.started_at) / 3600:.3f} hours; "
                f"currently running: {state.running or '[none]'}. "
                f"Completed runs: {len(state.completed)}. Failed runs: {len(state.failed)}.",
                "",
            ]
        )
    if data_report is not None:
        cache_records = [
            CacheStatistics.model_validate_json(path.read_bytes())
            for path in sorted(data_report.parent.glob("cache_stats_*.json"))
        ]
        if cache_records:
            largest = max(cache_records, key=lambda record: record.feature_count)
            cache_hours = (
                sum(record.extraction_seconds + record.asr_seconds for record in cache_records)
                / 3600
            )
            lines.extend(
                [
                    f"Cached encoder features: {largest.feature_count:,} clips; "
                    f"{largest.feature_bytes / 1024**3:.3f} GiB; "
                    f"{largest.total_audio_seconds / 3600:.3f} audio hours. "
                    f"Recorded feature extraction + ASR GPU wall time: {cache_hours:.3f} hours. "
                    f"Last extraction throughput: "
                    f"{largest.extraction_audio_seconds_per_second:.2f} audio seconds/second.",
                    "",
                    f"Whisper padding/masking: {largest.masking}",
                    "",
                ]
            )
    lines.extend(
        [
            "## Failures and limitations",
            "",
            "Completed runs appear only after checkpoint/evaluation persistence. Missing planned "
            "stages remain pending or failed; consult suite_state.json and launcher logs "
            "for exact reasons. Failed attempts must remain in the state/log archive. "
            "One seed and small fixed evaluation subsets make architecture/scaling "
            "comparisons exploratory.",
            "",
            "## Recommended next experiments",
            "",
            "Repeat the best settings with multiple seeds and matched optimizer-update budgets, "
            "enlarge held-out semantic/qualitative evaluation, and test natural human recordings "
            "plus stronger paired-audio controls. Review these V0–V3 results before considering "
            "emotional data or LLM adaptation.",
            "",
            "[Readable fixed qualitative comparisons](qualitative_comparison.md). Raw per-run "
            "configs, checkpoints, metrics, conditioning diagnostics, and generations remain "
            "in each run directory.",
            "",
        ]
    )
    for failure_path in (root / "validation_failures.jsonl", root / "failures.jsonl"):
        if failure_path.exists():
            lines.extend(["Recorded failures:", ""])
            for line in failure_path.read_text(encoding="utf-8").splitlines():
                if line:
                    failure = ExperimentFailure.model_validate_json(line)
                    lines.append(
                        f"- {failure.run_name}, attempt {failure.attempt}: "
                        f"{failure.exception_type}: {failure.message}"
                    )
            lines.append("")
    _save_csv(results, root / "summary.csv")
    _save_plots(results, root)
    _qualitative(root, results)
    path = root / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-report", type=Path)
    arguments = parser.parse_args()
    print(aggregate_report(arguments.root, arguments.data_report))


if __name__ == "__main__":
    main()
