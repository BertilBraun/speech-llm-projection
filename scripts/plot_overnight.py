"""Plot measured fixed-budget compression results and export their exact values."""

import argparse
import csv
from pathlib import Path

from matplotlib import pyplot

from speech_projector.models import Architecture, RunResult
from speech_projector.overnight_evaluation import SweepCandidate, SweepDecision

pyplot.switch_backend("Agg")


def draw_sweep(decision_path: Path, results_root: Path, output_directory: Path) -> None:
    decision = SweepDecision.model_validate_json(decision_path.read_bytes())
    candidates = decision.candidates
    if not candidates:
        raise ValueError("Plots require completed, measured candidates")
    results = tuple(
        RunResult.model_validate_json(
            (results_root / candidate.configuration.name / "result.json").read_bytes()
        )
        for candidate in candidates
    )
    if any(
        result.config != candidate.configuration
        for result, candidate in zip(results, candidates, strict=True)
    ):
        raise ValueError("Recorded results and selection configurations differ")
    if any(result.steps != 2000 for result in results):
        raise ValueError("This plot compares only completed 2,000-update sweep runs")
    output_directory.mkdir(parents=True, exist_ok=True)
    write_values(output_directory / "compression_values.csv", candidates, results)
    figure, axes = pyplot.subplots(1, 3, figsize=(15, 4.6))
    for architecture, label, color, marker in (
        (Architecture.MLP, "Mean pooling + MLP", "#2463a0", "o"),
        (Architecture.STACKED_MLP, "Stacking + MLP", "#bc541e", "s"),
    ):
        selected = sorted(
            (
                item
                for item in candidates
                if item.configuration.projector.architecture == architecture
            ),
            key=lambda item: item.pseudo_tokens_per_second,
        )
        if not selected:
            continue
        rates = [item.pseudo_tokens_per_second for item in selected]
        axes[0].plot(
            rates,
            [item.validation.macro_cross_entropy for item in selected],
            marker=marker,
            color=color,
            label=label,
        )
        intervals = [item.new_neu_preference.resized_matching_margin for item in selected]
        axes[1].vlines(
            rates,
            [item.lower for item in intervals],
            [item.upper for item in intervals],
            color=color,
            alpha=0.75,
        )
        axes[1].plot(
            rates,
            [item.estimate for item in intervals],
            marker=marker,
            color=color,
        )
        axes[2].plot(
            rates,
            [item.training_seconds_per_update for item in selected],
            marker=marker,
            color=color,
        )
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks([2.5, 5, 10, 25], labels=["2.5", "5", "10", "25"])
        axis.set_xlabel("Speech pseudo-tokens / second")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Validation macro CE (nats); lower is better")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("New Neu matching margin (nats); higher is better")
    axes[1].axhline(0, color="gray", linewidth=1, linestyle="--")
    axes[1].set_title("Duration-matched swaps; family bootstrap 95% CI", fontsize=9)
    axes[2].set_ylabel("Training wall seconds / optimizer update")
    figure.suptitle("Measured compression sweep — fixed 2,000 updates per configuration")
    figure.tight_layout()
    figure.savefig(output_directory / "compression_sweep.png", dpi=180)
    figure.savefig(output_directory / "compression_sweep.pdf")
    pyplot.close(figure)


def write_values(
    destination: Path, candidates: tuple[SweepCandidate, ...], results: tuple[RunResult, ...]
) -> None:
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "run",
                "architecture",
                "pseudo_tokens_per_second",
                "parameters",
                "updates",
                "macro_ce",
                "ordinary_ce",
                "old_emotional_ce",
                "new_neu_ce",
                "new_neu_raw_margin",
                "new_neu_resized_margin",
                "resized_ci_lower",
                "resized_ci_upper",
                "seconds_per_update",
                "allocated_peak_gb",
            )
        )
        for candidate, result in zip(candidates, results, strict=True):
            preference = candidate.new_neu_preference
            writer.writerow(
                (
                    candidate.configuration.name,
                    candidate.configuration.projector.architecture.value,
                    candidate.pseudo_tokens_per_second,
                    result.projector_parameters,
                    result.steps,
                    candidate.validation.macro_cross_entropy,
                    candidate.validation.old_ordinary.cross_entropy,
                    candidate.validation.old_emotional.cross_entropy,
                    candidate.validation.new_neu_emotional.cross_entropy,
                    preference.matching_margin.estimate,
                    preference.resized_matching_margin.estimate,
                    preference.resized_matching_margin.lower,
                    preference.resized_matching_margin.upper,
                    candidate.training_seconds_per_update,
                    result.peak_vram_gb,
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    draw_sweep(arguments.decision, arguments.results_root, arguments.output)


if __name__ == "__main__":
    main()
