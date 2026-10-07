"""Evaluate one follow-up checkpoint, then release the GPU before the next job."""

import argparse
from pathlib import Path

from speech_projector.followup_evaluation import FollowupEvaluationConfig, evaluate_followup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = FollowupEvaluationConfig.model_validate_json(arguments.config.read_bytes())
    evaluate_followup(configuration)


if __name__ == "__main__":
    main()
