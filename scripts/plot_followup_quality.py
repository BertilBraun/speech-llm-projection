"""Plot locked ordinal response comparisons without modifying ratings or rerunning inference."""

import argparse
import csv
from pathlib import Path

import matplotlib

from speech_projector.followup_quality import QualityScoredReport

matplotlib.use("Agg")
from matplotlib import pyplot as plotting  # noqa: E402


def write_quality_plot(
    report: QualityScoredReport,
    output_directory: Path,
    first_label: str,
    second_label: str,
) -> None:
    if not first_label.strip() or not second_label.strip() or first_label == second_label:
        raise ValueError("Quality plot needs two distinct nonblank method labels")
    groups = tuple((row.cohort.value, row.comparison) for row in report.cohorts)
    if len(groups) > 1:
        groups = (("All emotional", report.overall),) + groups
    if not groups:
        raise ValueError("Quality plot requires scored cohorts")
    labels = tuple(
        f"{name.replace('_', ' ')}\nn={summary.examples}; {summary.families} families"
        for name, summary in groups
    )
    figure, axes = plotting.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    positions = tuple(range(len(groups)))
    for index in range(2):
        measurements = tuple(
            summary.tone if index == 0 else summary.grounding for _, summary in groups
        )
        axes[0, index].bar(
            tuple(position - 0.18 for position in positions),
            tuple(row.first_mean for row in measurements),
            width=0.34,
            label=first_label,
        )
        axes[0, index].bar(
            tuple(position + 0.18 for position in positions),
            tuple(row.second_mean for row in measurements),
            width=0.34,
            label=second_label,
        )
        axes[0, index].set_ylim(0, 2)
        axes[0, index].set_xticks(positions, labels, fontsize=9)
        axes[0, index].set_ylabel("Mean ordinal score (0–2) ↑")
        axes[0, index].legend(fontsize=8)
        axes[0, index].grid(axis="y", alpha=0.25)
        axes[0, index].set_axisbelow(True)
    for position, (_name, summary) in enumerate(groups):
        for index, axis in enumerate((summary.tone, summary.grounding)):
            interval = axis.first_minus_second
            axes[1, index].errorbar(
                interval.estimate,
                position,
                xerr=[[interval.estimate - interval.lower], [interval.upper - interval.estimate]],
                fmt="o",
                capsize=4,
                color=f"C{position}",
            )
            axes[1, index].annotate(
                f"W/T/L {axis.wins}/{axis.ties}/{axis.losses}",
                (interval.estimate, position),
                xytext=(0, 13),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
    for index, title in enumerate(("Tone-appropriate helpfulness", "Literal grounding")):
        axes[0, index].set_title(title)
        axis = axes[1, index]
        axis.set_yticks(tuple(range(len(labels))), labels)
        axis.set_ylim(-0.5, len(labels) - 0.4)
        axis.axvline(0, color="gray", linewidth=0.8)
        axis.grid(axis="x", alpha=0.25)
        axis.set_xlabel("Paired mean difference on 0–2 rubric ↑\n95% family-bootstrap CI")
    figure.suptitle(
        f"{first_label} − {second_label}\n"
        "Blinded model-assisted judgments; intended synthetic tones, not emotion accuracy\n"
        "Means: heuristic ordinal point estimates; difference bars: paired family-bootstrap CI",
        fontsize=12,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(output_directory / f"quality_comparison.{extension}", dpi=180)
    plotting.close(figure)
    with (output_directory / "quality_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "first_source",
                "second_source",
                "cohort",
                "cases",
                "families",
                "axis",
                "first_mean",
                "second_mean",
                "difference",
                "lower_95",
                "upper_95",
                "wins",
                "ties",
                "losses",
                "half_credit_win_fraction",
            )
        )
        for name, summary in groups:
            for axis_name, axis in (("tone", summary.tone), ("grounding", summary.grounding)):
                interval = axis.first_minus_second
                writer.writerow(
                    (
                        report.provenance.configuration.first.name,
                        report.provenance.configuration.second.name,
                        name,
                        summary.examples,
                        summary.families,
                        axis_name,
                        axis.first_mean,
                        axis.second_mean,
                        interval.estimate,
                        interval.lower,
                        interval.upper,
                        axis.wins,
                        axis.ties,
                        axis.losses,
                        axis.half_credit_win_fraction.estimate,
                    )
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-label", required=True)
    parser.add_argument("--second-label", required=True)
    arguments = parser.parse_args()
    report = QualityScoredReport.model_validate_json(arguments.report.read_bytes())
    write_quality_plot(report, arguments.output, arguments.first_label, arguments.second_label)


if __name__ == "__main__":
    main()
