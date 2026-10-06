"""Generate metadata-aware Qwen preview replies, with optional transcript-only controls."""

import argparse
from pathlib import Path

from speech_projector.preview_responses import PreviewResponseConfig, run_preview_responses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--delivery-only", action="store_true")
    arguments = parser.parse_args()
    summary = run_preview_responses(
        PreviewResponseConfig(
            plan_path=arguments.plan,
            output_directory=arguments.output,
            revision=arguments.revision,
            source_git_commit=arguments.source_git_commit,
            include_transcript_controls=not arguments.delivery_only,
        )
    )
    print(summary.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
