"""Prepare blinded emotional comparison cards without inference or exposing source mappings."""

import argparse
from pathlib import Path

from speech_projector.followup_quality import QualityComparisonConfig, prepare_quality_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = QualityComparisonConfig.model_validate_json(arguments.config.read_bytes())
    cards = prepare_quality_comparison(configuration)
    print(f"Prepared {len(cards.cases)} blinded cases; mappings remain in a separate file")


if __name__ == "__main__":
    main()
