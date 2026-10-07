"""Render concise README figures directly from the locked experiment reports."""

import argparse
from pathlib import Path

import matplotlib

from speech_projector.followup_matched_metrics import MatchedMetricsReport
from speech_projector.followup_quality import QualityScoredReport
from speech_projector.followup_report import FollowupMeasuredReport
from speech_projector.overnight_data import Cohort

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
from matplotlib import pyplot as plotting  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


def save_figure(figure: Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        destination = directory / f"{name}.{extension}"
        figure.savefig(destination, dpi=180)
        if extension == "svg":
            destination.write_text(
                "\n".join(line.rstrip() for line in destination.read_text().splitlines()) + "\n",
                encoding="utf-8",
                newline="\n",
            )
    plotting.close(figure)


def plot_quality(analysis: Path, output: Path) -> None:
    reports = tuple(
        QualityScoredReport.model_validate_json(
            (analysis / "quality" / source / "quality_summary.json").read_bytes()
        )
        for source in ("selected_speech_vs_asr", "asr_tone_vs_asr")
    )
    summaries = tuple(
        next(row.comparison for row in report.cohorts if row.cohort == Cohort.NEU_EMOTIONAL)
        for report in reports
    )
    figure, axes = plotting.subplots(1, 2, figsize=(11, 3.8))
    for axis, title, measurements in zip(
        axes,
        ("Emotional helpfulness", "Factual grounding"),
        (
            tuple(summary.tone for summary in summaries),
            tuple(summary.grounding for summary in summaries),
        ),
        strict=True,
    ):
        for position, measurement in enumerate(measurements):
            interval = measurement.first_minus_second
            axis.errorbar(
                interval.estimate,
                1 - position,
                xerr=[[interval.estimate - interval.lower], [interval.upper - interval.estimate]],
                fmt="o",
                color=("#3268aa", "#bf6b26")[position],
                markersize=7,
                capsize=5,
            )
            axis.annotate(
                f"{interval.estimate:+.3g} [{interval.lower:.3g}, {interval.upper:.3g}]",
                (interval.estimate, 1 - position),
                xytext=(0, 13),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        axis.axvline(0, color="#666666", linewidth=1)
        axis.set_title(title)
        axis.set_ylim(-0.5, 1.65)
        axis.set_xlim((-0.04, 0.56) if title == "Emotional helpfulness" else (-0.68, 0.32))
        axis.set_yticks((1, 0), ("Speech projector", "ASR + predicted tone"))
        axis.set_xlabel("Difference from plain ASR (0–2 rubric points) → better")
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Tone gains over plain ASR; grounding benefit unproven", fontsize=13)
    figure.text(
        0.5,
        0.015,
        "Neu synthetic panel: 64 replies each; 95% family-bootstrap intervals. "
        "Separately judged comparisons, not a direct head-to-head test.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    save_figure(figure, output, "emotional_tradeoff")


def plot_alignment(analysis: Path, output: Path) -> None:
    report = FollowupMeasuredReport.model_validate_json(
        (analysis / "final_core_comparison" / "measured_comparison.json").read_bytes()
    )
    matched = MatchedMetricsReport.model_validate_json(
        (analysis / "final_neu64_matched" / "matched_metrics.json").read_bytes()
    )
    names = ("followup_mean_10hz_ce_control_6775", "true_TEXT", "plain_ASR")
    ordinary = tuple(next(row for row in report.measurements if row.name == name) for name in names)
    emotional_names = (
        "followup_mean_10hz_ce_control_6775",
        "true_text",
        "plain_asr",
        "asr_predicted_tone",
    )
    emotional = tuple(
        next(row for row in matched.measurements if row.name == name) for name in emotional_names
    )
    figure, axes = plotting.subplots(1, 3, figsize=(12, 4.1))
    values = (
        tuple(row.cohorts.old_ordinary.cross_entropy for row in ordinary),
        tuple(row.metrics.cross_entropy for row in emotional),
        tuple(row.metrics.semantic_similarity for row in emotional),
    )
    assert all(value is not None for value in values[2])
    labels = ("Speech projector", "True transcript", "Plain ASR", "ASR + tone")
    for axis, measurements, title in zip(
        axes,
        values,
        (
            "Ordinary teacher loss ↓\n96 examples",
            "Emotional teacher loss ↓\n64 matched replies",
            "Emotional reference similarity ↑\n64 matched replies",
        ),
        strict=True,
    ):
        positions = tuple(range(len(measurements)))
        axis.barh(
            positions,
            measurements,
            color=("#3268aa", "#728394", "#92a992", "#bf6b26")[: len(measurements)],
        )
        axis.set_yticks(positions, labels[: len(measurements)])
        axis.invert_yaxis()
        axis.set_title(title, fontsize=11)
        axis.set_xlim(0, 1.82 if len(measurements) == 3 else 1)
        axis.bar_label(
            axis.containers[0],
            labels=tuple(f"{value:.3g}" for value in measurements),
            padding=4,
            fontsize=9,
        )
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    figure.suptitle(
        "Teacher imitation and semantic similarity measure different things", fontsize=13
    )
    figure.text(
        0.5,
        0.015,
        "Saved one-seed point estimates; uncertainty unmeasured. "
        "Emotional teachers saw intended tone. These metrics are not answer accuracy.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.93))
    save_figure(figure, output, "teacher_alignment")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    plot_quality(arguments.analysis, arguments.output)
    plot_alignment(arguments.analysis, arguments.output)


if __name__ == "__main__":
    main()
