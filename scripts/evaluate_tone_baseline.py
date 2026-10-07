"""Run a separately labelled predicted-tone or oracle-tone ASR baseline."""

import argparse
from pathlib import Path

from pydantic import TypeAdapter

from speech_projector.followup_tone_baselines import (
    ToneBaselineConfiguration,
    evaluate_tone_baseline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = TypeAdapter(ToneBaselineConfiguration).validate_json(
        arguments.config.read_bytes()
    )
    evaluate_tone_baseline(configuration)


if __name__ == "__main__":
    main()
