"""Paired dialogue-bootstrap confidence intervals for blinded quality judgments."""

import argparse
from pathlib import Path

from scripts.package_results import FileArtifact, file_digest
from speech_projector.judge import (
    BootstrapInterval,
    JudgeMetric,
    JudgeOutcome,
    JudgeSuccess,
    PairedMetricObservation,
    load_judgments,
    paired_dialogue_bootstrap,
    verdict_metric,
)
from speech_projector.models import Record


class JudgmentMetricComparison(Record):
    metric: JudgeMetric
    left_minus_right: BootstrapInterval


class JudgmentComparison(Record):
    inputs: tuple[FileArtifact, ...]
    left_requested: int
    right_requested: int
    shared_requested: int
    paired_valid: int
    shared_failed: int
    metrics: tuple[JudgmentMetricComparison, ...]


def paired_values(
    left: tuple[JudgeOutcome, ...], right: tuple[JudgeOutcome, ...], metric: JudgeMetric
) -> tuple[PairedMetricObservation, ...]:
    pairs: list[PairedMetricObservation] = []
    right_by_id = {item.request.example_id: item for item in right}
    if len(right_by_id) != len(right) or len({item.request.example_id for item in left}) != len(
        left
    ):
        raise ValueError("Judgment comparison requires unique example IDs per system")
    for item in left:
        if item.request.example_id not in right_by_id:
            continue
        reference = right_by_id[item.request.example_id]
        if (
            item.request.dialogue_id,
            item.request.history,
            item.request.transcript,
        ) != (
            reference.request.dialogue_id,
            reference.request.history,
            reference.request.transcript,
        ):
            raise ValueError("Paired quality judgments differ in transcript/history")
        match item, reference:
            case JudgeSuccess(), JudgeSuccess():
                pairs.append(
                    PairedMetricObservation(
                        example_id=item.request.example_id,
                        dialogue_id=item.request.dialogue_id,
                        difference=verdict_metric(item.verdict, metric)
                        - verdict_metric(reference.verdict, metric),
                    )
                )
            case _:
                continue
    return tuple(pairs)


def compare_journals(left_path: Path, right_path: Path) -> JudgmentComparison:
    left, right = load_judgments(left_path), load_judgments(right_path)
    shared = len(
        {item.request.example_id for item in left} & {item.request.example_id for item in right}
    )
    metrics = tuple(
        JudgmentMetricComparison(
            metric=metric,
            left_minus_right=paired_dialogue_bootstrap(paired_values(left, right, metric)),
        )
        for metric in JudgeMetric
    )
    paired = metrics[0].left_minus_right.examples
    return JudgmentComparison(
        inputs=tuple(
            FileArtifact(
                path=path, source_path=path, bytes=path.stat().st_size, sha256=file_digest(path)
            )
            for path in (left_path, right_path)
        ),
        left_requested=len(left),
        right_requested=len(right),
        shared_requested=shared,
        paired_valid=paired,
        shared_failed=shared - paired,
        metrics=metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    comparison = compare_journals(arguments.left, arguments.right)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(comparison.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(comparison.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
