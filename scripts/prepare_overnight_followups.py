"""Write four shared neutral evaluation-only follow-ups; these are not training examples."""

import argparse
from pathlib import Path

from speech_projector.tts_pilot import PilotEmotion, TtsPilotCase, TtsPilotManifest


def followup_manifest() -> TtsPilotManifest:
    return TtsPilotManifest(
        cases=(
            TtsPilotCase(
                case_id="followup_short",
                utterance_id="followup_short",
                text="I'd like to keep this brief. Please remember the details I mentioned.",
                emotion=PilotEmotion.NEUTRAL,
                seed=42,
            ),
            TtsPilotCase(
                case_id="followup_next_step",
                utterance_id="followup_next_step",
                text="Given what I told you earlier, what would be a sensible next step?",
                emotion=PilotEmotion.NEUTRAL,
                seed=42,
            ),
            TtsPilotCase(
                case_id="followup_summary",
                utterance_id="followup_summary",
                text="Please summarize the situation I described without adding any new details.",
                emotion=PilotEmotion.NEUTRAL,
                seed=42,
            ),
            TtsPilotCase(
                case_id="followup_stop",
                utterance_id="followup_stop",
                text=(
                    "Thanks. I want to end this conversation now, "
                    "so please do not ask another question."
                ),
                emotion=PilotEmotion.NEUTRAL,
                seed=42,
            ),
        ),
        warmup_text="I'd like to keep this brief. Please remember the details I mentioned.",
        warmup_seed=41,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = followup_manifest()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if arguments.output.exists():
        if TtsPilotManifest.model_validate_json(arguments.output.read_bytes()) != manifest:
            raise ValueError("Refusing to replace a different evaluation-only follow-up manifest")
    else:
        arguments.output.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(arguments.output)


if __name__ == "__main__":
    main()
