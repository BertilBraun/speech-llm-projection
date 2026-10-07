"""Transcribe neutral follow-ups once, release Whisper, then compare conversation pipelines."""

import argparse
from pathlib import Path

from speech_projector.followup_conversation import (
    FollowupConversationConfig,
    run_followup_conversation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = FollowupConversationConfig.model_validate_json(arguments.config.read_bytes())
    run_followup_conversation(configuration)


if __name__ == "__main__":
    main()
