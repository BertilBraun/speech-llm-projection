"""Run final held-out four-turn inference with a recorded extended history budget."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from speech_projector.launcher import git_revision, initialize_projector
from speech_projector.llm import FrozenQwen
from speech_projector.models import RunResult
from speech_projector.overnight_conversation import DelayedCueFixtures
from speech_projector.overnight_conversation_execution import evaluate_conversations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-tokens", type=int, default=2048)
    arguments = parser.parse_args()
    if arguments.history_tokens < 1:
        parser.error("--history-tokens must be positive")
    result = RunResult.model_validate_json(arguments.run.read_bytes())
    configuration = result.config.model_copy(
        update={"history_turns": 6, "max_history_tokens": arguments.history_tokens}
    )
    fixtures = DelayedCueFixtures.model_validate_json(arguments.fixtures.read_bytes())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = FrozenQwen(configuration, device)
    projector = initialize_projector(configuration, device)
    projector.load_state_dict(load_file(str(result.checkpoint_path), device=str(device)))
    replies = evaluate_conversations(
        wrapper, projector, fixtures, arguments.output, source_commit=git_revision()
    )
    print(f"Completed {len(replies)} saved conversation responses", flush=True)


if __name__ == "__main__":
    main()
