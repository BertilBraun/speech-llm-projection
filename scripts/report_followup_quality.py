"""Freeze complete response ratings once, or report already locked ratings."""

import argparse
from pathlib import Path

from speech_projector.followup_quality import freeze_quality_ratings, report_quality_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    arguments = parser.parse_args()
    if arguments.freeze is not None:
        freeze_quality_ratings(
            arguments.directory / "blind_cases.json", arguments.ratings, arguments.freeze
        )
    else:
        report_quality_comparison(arguments.directory, arguments.ratings)


if __name__ == "__main__":
    main()
