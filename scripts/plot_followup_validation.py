"""Plot the completed matched validation decision separately from descriptive TEST comparisons."""

import argparse
import csv
from pathlib import Path

import matplotlib

from speech_projector.followup_selection import AlignmentSelectionDecision
from speech_projector.models import (
    OrdinaryResponseKLObjective,
    ResponseCrossEntropyObjective,
    TranscriptMixtureObjective,
)
from speech_projector.overnight_evaluation import SweepCandidate

matplotlib.use("Agg")
from matplotlib import pyplot as plotting  # noqa: E402


def candidate_label(candidate: SweepCandidate, control_run: str) -> str:
    if candidate.configuration.name == control_run:
        return "CE control"
    match candidate.configuration.objective:
        case TranscriptMixtureObjective():
            return "Transcript mix"
        case OrdinaryResponseKLObjective():
            return "Teacher KL"
        case ResponseCrossEntropyObjective():
            return "Response CE"


def write_validation_plot(decision: AlignmentSelectionDecision, directory: Path) -> None:
    candidates = decision.validation_candidates
    positions = tuple(range(len(candidates)))
    labels = tuple(candidate_label(row, decision.control_run) for row in candidates)
    figure, axes = plotting.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    axes[0].bar(positions, [row.validation.old_ordinary.cross_entropy for row in candidates])
    axes[1].bar(positions, [row.validation.macro_cross_entropy for row in candidates])
    for index, candidate in enumerate(candidates):
        for offset, interval, label in (
            (-0.12, candidate.new_neu_preference.matching_margin, "Raw audio"),
            (0.12, candidate.new_neu_preference.resized_matching_margin, "Resized wrong audio"),
        ):
            axes[2].errorbar(
                index + offset,
                interval.estimate,
                yerr=[[interval.estimate - interval.lower], [interval.upper - interval.estimate]],
                fmt="o",
                capsize=3,
                color="C0" if offset < 0 else "C1",
                label=label if index == 0 else None,
            )
    ordinary_counts = "/".join(
        str(count) for count in sorted({row.validation.old_ordinary.examples for row in candidates})
    )
    panel_counts = "/".join(
        str(count)
        for count in sorted(
            {
                row.validation.old_ordinary.examples
                + row.validation.old_emotional.examples
                + row.validation.new_neu_emotional.examples
                for row in candidates
            }
        )
    )
    pair_counts = "/".join(
        str(count) for count in sorted({row.new_neu_preference.pairs for row in candidates})
    )
    axes[0].set_ylabel(f"Ordinary target CE ↓ (n={ordinary_counts})")
    axes[1].set_ylabel(f"Three-cohort macro CE ↓ (panel n={panel_counts})")
    axes[2].set_ylabel(f"Neu paired matching benefit ↑ (n={pair_counts} pairs)")
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    selected = next(row for row in candidates if row.configuration.name == decision.selected_run)
    figure.suptitle(
        f"Matched {decision.policy.endpoint_updates}-update VAL branches; "
        f"selected: {candidate_label(selected, decision.control_run)}\n"
        "Frozen rule; TEST unused; CE uncertainty unmeasured; "
        "margin bars: 95% family-bootstrap CI",
        fontsize=11,
    )
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"validation_branches.{extension}", dpi=180)
    plotting.close(figure)
    with (directory / "validation_branches.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "run_name",
                "objective",
                "ordinary_ce",
                "old_emotional_ce",
                "neu_ce",
                "macro_ce",
                "raw_neu_margin",
                "raw_lower_95",
                "raw_upper_95",
                "resized_neu_margin",
                "resized_lower_95",
                "resized_upper_95",
                "robust_neu_margin",
                "strict_raw_assignment_win",
                "tie_fraction",
                "pairs",
                "families",
                "eligible",
                "macro_band",
                "selected",
            )
        )
        for candidate in candidates:
            name = candidate.configuration.name
            preference = candidate.new_neu_preference
            raw, resized = preference.matching_margin, preference.resized_matching_margin
            writer.writerow(
                (
                    name,
                    candidate.configuration.objective.kind,
                    candidate.validation.old_ordinary.cross_entropy,
                    candidate.validation.old_emotional.cross_entropy,
                    candidate.validation.new_neu_emotional.cross_entropy,
                    candidate.validation.macro_cross_entropy,
                    raw.estimate,
                    raw.lower,
                    raw.upper,
                    resized.estimate,
                    resized.lower,
                    resized.upper,
                    candidate.robust_neu_margin,
                    preference.matching_win_rate.estimate,
                    preference.tie_rate,
                    preference.pairs,
                    preference.family_clusters,
                    name in decision.eligible_runs,
                    name in decision.macro_band_runs,
                    name == decision.selected_run,
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    decision = AlignmentSelectionDecision.model_validate_json(arguments.decision.read_bytes())
    write_validation_plot(decision, arguments.output)


if __name__ == "__main__":
    main()
