"""Fit the CPU-only Neu classifier on existing frozen Whisper cached states."""

import argparse
from pathlib import Path

from speech_projector.tone_classifier import ToneClassifierConfig, train_tone_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-git-commit", required=True)
    arguments = parser.parse_args()
    config = ToneClassifierConfig.model_validate_json(arguments.config.read_bytes())
    report = train_tone_classifier(config, arguments.source_git_commit)
    print(
        f"Saved {len(report.selected_example_ids)} Neu classifier observations: "
        f"{config.output_directory}"
    )
    for metric in report.metrics:
        print(
            f"{metric.split.value}: n={metric.examples} "
            f"balanced_acc={metric.balanced_accuracy:.6f} macro_f1={metric.macro_f1:.6f} "
            f"both_correct_pairs={metric.both_deliveries_correct_fraction:.6f}"
        )


if __name__ == "__main__":
    main()
