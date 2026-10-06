"""Wait for supervised text generation, then run resumable audio generation."""

import argparse
from pathlib import Path

from speech_projector.emotional_audio_queue import EmotionalAudioQueueConfig, run_audio_queue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = EmotionalAudioQueueConfig.model_validate_json(arguments.config.read_bytes())
    run_audio_queue(configuration)


if __name__ == "__main__":
    main()
