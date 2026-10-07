"""Evaluate ASR words with a hash-bound classifier cue on the initial USER message only."""

import argparse
from pathlib import Path

from speech_projector.followup_conversation_tone import (
    PredictedInitialToneConversationConfig,
    prepare_conversation_tone,
    run_predicted_initial_tone,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    configuration = PredictedInitialToneConversationConfig.model_validate_json(
        arguments.config.read_bytes()
    )
    if arguments.validate_only:
        fixtures, provenance = prepare_conversation_tone(configuration)
        print(
            f"Validated {len(fixtures.scenarios)} held-out families / "
            f"{len(provenance.initial_inputs)} classifier-bound initial utterances; "
            "no inference performed."
        )
    else:
        run_predicted_initial_tone(configuration)


if __name__ == "__main__":
    main()
