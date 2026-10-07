"""CPU-only exact-identity merge, preserving completed conversation source artifacts."""

import argparse
from pathlib import Path

from speech_projector.conversation_reuse import (
    ConversationReuseConfig,
    merge_conversation_references,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = ConversationReuseConfig.model_validate_json(arguments.config.read_bytes())
    receipt = merge_conversation_references(configuration)
    print(receipt.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
