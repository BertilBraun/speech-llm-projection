"""Pause a live Linux suite after its final V2 result becomes complete."""

import argparse
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from speech_projector.models import ExperimentStage, Record, RunResult


class BoundaryPause(Record):
    process_id: int
    paused_at_unix: float
    paused_at_utc: str
    completed_result_path: Path
    completed_run_name: str
    completed_validation_cross_entropy: float
    source_git_commit: str


def completed_boundary(path: Path, expected_commit: str) -> RunResult | None:
    if not path.exists():
        return None
    try:
        result = RunResult.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as error:
        if all(problem["type"] == "json_invalid" for problem in error.errors()):
            return None
        raise
    if (
        result.config.stage != ExperimentStage.V2
        or result.config.projector.compression_factor != 20
    ):
        raise ValueError("The selected result is not the final factor-20 V2 boundary")
    if result.git_commit != expected_commit:
        raise ValueError("The boundary result source commit differs from the live suite")
    if not result.checkpoint_path.is_file():
        raise ValueError("The completed V2 checkpoint is absent")
    return result


def pause_at_boundary(
    process_id: int, result_path: Path, output_path: Path, expected_commit: str
) -> None:
    command_path = Path(f"/proc/{process_id}/cmdline")
    while True:
        command = command_path.read_bytes().split(b"\0")
        if b"speech_projector.launcher" not in command or b"--suite" not in command:
            raise ValueError("The supplied PID no longer identifies the expected live suite")
        result = completed_boundary(result_path, expected_commit)
        if result is not None:
            paused_at = time.time()
            record = BoundaryPause(
                process_id=process_id,
                paused_at_unix=paused_at,
                paused_at_utc=datetime.fromtimestamp(paused_at, timezone.utc).isoformat(),
                completed_result_path=result_path,
                completed_run_name=result.config.name,
                completed_validation_cross_entropy=result.validation.cross_entropy,
                source_git_commit=result.git_commit,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pending_path = output_path.with_suffix(".pending.json")
            pending_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
            os.kill(process_id, signal.SIGSTOP)
            pending_path.replace(output_path)
            print(record.model_dump_json(indent=2), flush=True)
            return
        time.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    arguments = parser.parse_args()
    if arguments.process_id <= 0:
        raise ValueError("The suite process ID must be positive")
    pause_at_boundary(
        arguments.process_id, arguments.result, arguments.output, arguments.source_commit
    )


if __name__ == "__main__":
    main()
