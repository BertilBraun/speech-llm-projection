"""Render measured follow-up comparisons and scientific plots without model inference."""

import argparse
from pathlib import Path

from speech_projector.followup_report import FollowupReportConfig, write_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    write_comparison(FollowupReportConfig.model_validate_json(arguments.config.read_bytes()))


if __name__ == "__main__":
    main()
