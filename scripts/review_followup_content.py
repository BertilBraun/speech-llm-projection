"""Screen saved replies for literal-risk review without inference or scoring word overlap."""

import argparse
from pathlib import Path

from speech_projector.content_review import ContentReviewConfig, run_content_review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    run_content_review(ContentReviewConfig.model_validate_json(arguments.config.read_bytes()))


if __name__ == "__main__":
    main()
