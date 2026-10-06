"""Generate resumable, completed frozen-Qwen targets for a clean dialogue manifest."""

import argparse
from pathlib import Path

from speech_projector.generation import INITIAL_TEACHER_TOKEN_CAP, RETRY_TEACHER_TOKEN_CAP
from speech_projector.models import RunConfig
from speech_projector.teacher import (
    BootstrapTeacherSelection,
    FullTeacherSelection,
    TeacherConfig,
    run_teacher,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--initial-max-new-tokens", type=int, default=INITIAL_TEACHER_TOKEN_CAP)
    parser.add_argument("--retry-max-new-tokens", type=int, default=RETRY_TEACHER_TOKEN_CAP)
    arguments = parser.parse_args()
    progress = run_teacher(
        TeacherConfig(
            run=RunConfig.model_validate_json(arguments.run_config.read_bytes()),
            manifest=arguments.manifest,
            output_directory=arguments.output,
            batch_size=arguments.batch_size,
            initial_max_new_tokens=arguments.initial_max_new_tokens,
            retry_max_new_tokens=arguments.retry_max_new_tokens,
        ),
        BootstrapTeacherSelection(output_manifest=arguments.output_manifest)
        if arguments.bootstrap
        else FullTeacherSelection(output_manifest=arguments.output_manifest),
    )
    print(progress.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
