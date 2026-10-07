"""Reaggregate actual saved responses and losses on an identical cohort panel."""

import argparse
from pathlib import Path

from speech_projector.followup_matched_metrics import MatchedMetricsConfig, compare_matched_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    compare_matched_metrics(MatchedMetricsConfig.model_validate_json(arguments.config.read_bytes()))


if __name__ == "__main__":
    main()
