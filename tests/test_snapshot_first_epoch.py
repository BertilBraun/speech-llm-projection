import hashlib
import os
from pathlib import Path

import pytest

from scripts.snapshot_first_epoch import SnapshotConfiguration, capture_ready_snapshot
from speech_projector.models import ExperimentStage, MlpProjectorConfig, RunConfig
from speech_projector.training import TrainingState


def checkpoint_fixture(directory: Path) -> tuple[SnapshotConfiguration, RunConfig]:
    checkpoint = directory / "checkpoint"
    checkpoint.mkdir()
    projector = checkpoint / "projector.safetensors"
    projector.write_bytes(b"test-local projector checkpoint")
    state = TrainingState(
        epoch=1,
        offset=0,
        step=2500,
        examples_seen=20000,
        target_tokens_seen=500000,
        elapsed_seconds=2500.0,
        initial_validation_loss=3.0,
        initial_training_loss=3.0,
        final_training_loss=2.0,
    )
    state_path = checkpoint / "state.json"
    state_path.write_text(state.model_dump_json(), encoding="utf-8")
    os.utime(projector, ns=(1_000_000_000, 1_000_000_000))
    os.utime(state_path, ns=(2_000_000_000, 2_000_000_000))
    config = RunConfig(
        name="v1_20000_mlp_10hz",
        stage=ExperimentStage.V1,
        train_examples=20000,
        epochs=2,
        learning_rate=0.001,
        projector=MlpProjectorConfig(compression_factor=5),
    )
    return (
        SnapshotConfiguration(
            run_directory=directory,
            output_directory=directory / "matched_budget",
            source_git_commit="source",
        ),
        config,
    )


def test_snapshot_preserves_exact_completed_state_and_projector(tmp_path: Path) -> None:
    configuration, run_config = checkpoint_fixture(tmp_path)
    provenance = capture_ready_snapshot(configuration, run_config)
    assert provenance is not None
    assert provenance.training_state.step == 2500
    assert provenance.training_state.epoch == 1
    for artifact in provenance.artifacts:
        copied = (configuration.output_directory / artifact.path).read_bytes()
        assert copied == artifact.source_path.read_bytes()
        assert hashlib.sha256(copied).hexdigest() == artifact.sha256
        assert len(copied) == artifact.bytes
    assert {path.name for path in configuration.output_directory.iterdir()} == {
        "projector.safetensors",
        "state.json",
        "provenance.json",
    }


def test_snapshot_rejects_projector_newer_than_completed_state(tmp_path: Path) -> None:
    configuration, run_config = checkpoint_fixture(tmp_path)
    os.utime(tmp_path / "checkpoint" / "projector.safetensors", ns=(3_000_000_000, 3_000_000_000))
    assert capture_ready_snapshot(configuration, run_config) is None
    assert not configuration.output_directory.exists()


def test_snapshot_detects_source_mutation_during_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration, run_config = checkpoint_fixture(tmp_path)
    original_write = Path.write_bytes

    def write_and_change_source(path: Path, content: bytes) -> int:
        written = original_write(path, content)
        if path.name == "projector.safetensors.pending":
            original_write(tmp_path / "checkpoint" / "projector.safetensors", b"changed")
        return written

    monkeypatch.setattr(Path, "write_bytes", write_and_change_source)
    assert capture_ready_snapshot(configuration, run_config) is None
    assert not (configuration.output_directory / "provenance.json").exists()
    assert not (configuration.output_directory / "projector.safetensors").exists()
