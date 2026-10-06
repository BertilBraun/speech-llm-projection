"""Snapshot paired emotional dataset progress without generation or journal mutation."""

import argparse
from pathlib import Path

from speech_projector.emotional_report import EmotionalReportConfig, write_emotional_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-directory", type=Path, required=True)
    parser.add_argument("--audio-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    report = write_emotional_report(
        EmotionalReportConfig(
            dataset_directory=arguments.dataset_directory,
            audio_directory=arguments.audio_directory,
            output_directory=arguments.output_directory,
        )
    )
    print(
        f"{report.available_utterances}/{report.planned_utterances} utterances; "
        f"{report.available_teacher_targets} targets; {report.available_audio_clips} clips"
    )


if __name__ == "__main__":
    main()
