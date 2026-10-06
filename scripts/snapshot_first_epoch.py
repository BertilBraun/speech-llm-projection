"""Preserve a larger V1 run's first-epoch checkpoint for matched-update analysis."""

import argparse
import hashlib
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.package_results import FileArtifact
from speech_projector.models import ExperimentStage, Record, RunConfig
from speech_projector.training import TrainingState


class SnapshotConfiguration(Record):
    run_directory: Path
    output_directory: Path
    source_git_commit: str


class SnapshotProvenance(Record):
    configuration: SnapshotConfiguration
    captured_at: datetime
    run_config: RunConfig
    training_state: TrainingState
    artifacts: tuple[FileArtifact, ...]


def same_file(before: os.stat_result, after: os.stat_result) -> bool:
    return (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )


def capture_ready_snapshot(
    config: SnapshotConfiguration, run_config: RunConfig
) -> SnapshotProvenance | None:
    checkpoint_directory = config.run_directory / "checkpoint"
    state_path = checkpoint_directory / "state.json"
    projector_path = checkpoint_directory / "projector.safetensors"
    if not state_path.is_file() or not projector_path.is_file():
        return None
    initial_state_stat = state_path.stat()
    state_content = state_path.read_bytes()
    state = TrainingState.model_validate_json(state_content)
    target_step = math.ceil(run_config.train_examples / run_config.gradient_accumulation)
    if state.step > target_step:
        raise ValueError("The first-epoch checkpoint was overwritten before it was preserved")
    if state.step != target_step or state.epoch != 1 or state.offset != 0:
        return None
    if state.examples_seen != run_config.train_examples:
        raise ValueError("The first-epoch exposure count differs from the configured subset")
    initial_projector_stat = projector_path.stat()
    if initial_projector_stat.st_mtime_ns >= initial_state_stat.st_mtime_ns:
        return None
    if not same_file(initial_state_stat, state_path.stat()):
        return None
    projector_content = projector_path.read_bytes()
    sources = (
        (projector_path, projector_content, initial_projector_stat),
        (state_path, state_content, initial_state_stat),
    )
    config.output_directory.mkdir(parents=True, exist_ok=True)
    artifacts: list[FileArtifact] = []
    for source_path, content, initial_stat in sources:
        destination = config.output_directory / source_path.name
        pending = destination.with_suffix(destination.suffix + ".pending")
        pending.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        if not same_file(initial_stat, source_path.stat()):
            return None
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != digest:
            return None
        artifacts.append(
            FileArtifact(
                path=Path(source_path.name),
                source_path=source_path,
                bytes=len(content),
                sha256=digest,
            )
        )
    if not all(same_file(initial_stat, path.stat()) for path, _, initial_stat in sources):
        return None
    provenance = SnapshotProvenance(
        configuration=config,
        captured_at=datetime.now(timezone.utc),
        run_config=run_config,
        training_state=state,
        artifacts=tuple(artifacts),
    )
    for source_path, _, _ in sources:
        destination = config.output_directory / source_path.name
        destination.with_suffix(destination.suffix + ".pending").replace(destination)
    provenance_path = config.output_directory / "provenance.json"
    pending_provenance = provenance_path.with_suffix(".pending.json")
    pending_provenance.write_text(provenance.model_dump_json(indent=2), encoding="utf-8")
    pending_provenance.replace(provenance_path)
    return provenance


def preserve_first_epoch(config: SnapshotConfiguration) -> SnapshotProvenance:
    provenance_path = config.output_directory / "provenance.json"
    if provenance_path.exists():
        raise ValueError("A completed snapshot already exists; it will not be overwritten")
    run_config = RunConfig.model_validate_json(
        (config.run_directory / "config.json").read_text(encoding="utf-8")
    )
    if run_config.stage != ExperimentStage.V1 or run_config.epochs < 2:
        raise ValueError("First-epoch comparison requires a multi-epoch V1 run")
    if run_config.microbatch_size != 1:
        raise ValueError("Matched updates require the experiment's single-example microbatch")
    while True:
        provenance = capture_ready_snapshot(config, run_config)
        if provenance is not None:
            print(provenance.model_dump_json(indent=2), flush=True)
            return provenance
        time.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    arguments = parser.parse_args()
    preserve_first_epoch(
        SnapshotConfiguration(
            run_directory=arguments.run_dir,
            output_directory=arguments.output_dir,
            source_git_commit=arguments.source_commit,
        )
    )


if __name__ == "__main__":
    main()
