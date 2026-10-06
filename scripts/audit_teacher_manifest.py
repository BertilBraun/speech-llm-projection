"""Verify the completed teacher-target manifest before projector training."""

import argparse
from pathlib import Path

from scripts.prepare_teacher_data import audit
from speech_projector.data import load_examples
from speech_projector.models import Split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    examples = [example for split in Split for example in load_examples(arguments.manifest, split)]
    audit(examples, arguments.output)


if __name__ == "__main__":
    main()
