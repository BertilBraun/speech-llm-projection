"""Report the committed selection, recorded results and preserved operational failures."""

import argparse
from pathlib import Path

from speech_projector.overnight_report import OvernightReportConfig, report_overnight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-run-directory", type=Path, action="append", default=[])
    parser.add_argument("--note", action="append", default=[])
    arguments = parser.parse_args()
    print(
        report_overnight(
            OvernightReportConfig(
                results_root=arguments.results_root,
                preparation_path=arguments.preparation,
                decision_path=arguments.decision,
                output_directory=arguments.output,
                final_run_directories=tuple(arguments.final_run_directory),
                operational_notes=tuple(arguments.note),
            )
        )
    )


if __name__ == "__main__":
    main()
