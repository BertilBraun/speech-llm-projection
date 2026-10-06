"""Render final conversation evidence and prepare judge requests without inference."""

import argparse
from pathlib import Path

from pydantic import TypeAdapter

from speech_projector.journal import read_journal
from speech_projector.overnight_conversation import DelayedCueFixtures
from speech_projector.overnight_conversation_execution import SavedConversationReply
from speech_projector.overnight_conversation_reporting import (
    conversation_judge_requests,
    render_conversations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--replies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-rollout-next-step", action="store_true")
    arguments = parser.parse_args()
    contents = arguments.replies.read_bytes()
    if contents and not contents.endswith(b"\n"):
        raise ValueError("Conversation reporting requires a complete immutable reply journal")
    fixtures = DelayedCueFixtures.model_validate_json(arguments.fixtures.read_bytes())
    replies = read_journal(arguments.replies, TypeAdapter(SavedConversationReply))
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "conversation_evidence.md").write_text(
        render_conversations(replies, fixtures), encoding="utf-8"
    )
    requests = conversation_judge_requests(
        replies, fixtures, include_rollout_next_step=arguments.include_rollout_next_step
    )
    (arguments.output / "judge_requests.jsonl").write_text(
        "\n".join(item.model_dump_json() for item in requests) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(requests)} judge requests from {len(replies)} actual saved replies")


if __name__ == "__main__":
    main()
