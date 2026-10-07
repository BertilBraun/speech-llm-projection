from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from speech_projector.models import (
    OrdinaryResponseKLObjective,
    ResponseCrossEntropyObjective,
    RunConfig,
    TranscriptMixtureObjective,
)
from speech_projector.objective_branch import ObjectiveBranchJob
from speech_projector.overnight_configuration import sweep_runs
from speech_projector.overnight_continuation import prepare_continuation, prepare_objective_branch
from speech_projector.training import (
    TrainingResourceRecord,
    TrainingState,
    ValidationCheckpointRecord,
)


def finished_parent(directory: Path) -> tuple[RunConfig, TrainingState]:
    configuration = sweep_runs(38193, 1)[0].model_copy(
        update={"name": "parent", "max_optimizer_updates": 4775}
    )
    checkpoint = directory / "checkpoint"
    checkpoint.mkdir(parents=True)
    (directory / "config.json").write_text(configuration.model_dump_json(), encoding="utf-8")
    state = TrainingState(
        epoch=1,
        offset=0,
        step=4775,
        examples_seen=38193,
        target_tokens_seen=4314521,
        elapsed_seconds=5000,
        initial_validation_loss=3,
        initial_training_loss=3.1,
        final_validation_loss=1.1,
        final_fixed_training_loss=1.05,
        final_training_loss=1.2,
    )
    (checkpoint / "state.json").write_text(state.model_dump_json(), encoding="utf-8")
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], lr=configuration.learning_rate)
    parameter.sum().backward()
    optimizer.step()
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    save_file({"weight": parameter.detach()}, checkpoint / "projector.safetensors")
    save_file({"weight": torch.zeros(2)}, directory / "best_projector.safetensors")
    (directory / "best_validation.json").write_text(
        ValidationCheckpointRecord(
            step=4000,
            cross_entropy=1,
            checkpoint_path=directory / "best_projector.safetensors",
        ).model_dump_json(),
        encoding="utf-8",
    )
    (directory / "training_resources.json").write_text(
        TrainingResourceRecord(peak_vram_gb=8).model_dump_json(), encoding="utf-8"
    )
    return configuration, state


@pytest.mark.parametrize(
    "objective",
    (ResponseCrossEntropyObjective(), TranscriptMixtureObjective(), OrdinaryResponseKLObjective()),
)
def test_branches_copy_exact_parent_moments_cursor_and_provenance(
    tmp_path: Path,
    objective: ResponseCrossEntropyObjective
    | TranscriptMixtureObjective
    | OrdinaryResponseKLObjective,
) -> None:
    source = tmp_path / "parent"
    original, state = finished_parent(source)
    configuration = original.model_copy(
        update={
            "name": "branch",
            "max_optimizer_updates": 6775,
            "learning_rate": 2e-4,
            "objective": objective,
        }
    )
    destination = tmp_path / "branch"
    provenance = prepare_objective_branch(source, destination, configuration)
    assert provenance.source_configuration == original
    assert provenance.continuation_configuration == configuration
    assert len(provenance.source_artifacts) == 5
    for filename in ("optimizer.pt", "projector.safetensors"):
        assert (source / "checkpoint" / filename).read_bytes() == (
            destination / "checkpoint" / filename
        ).read_bytes()
    continued = TrainingState.model_validate_json(
        (destination / "checkpoint" / "state.json").read_bytes()
    )
    assert continued == state.model_copy(
        update={"final_validation_loss": None, "final_fixed_training_loss": None}
    )
    assert (
        TrainingState.model_validate_json((source / "checkpoint" / "state.json").read_bytes())
        == state
    )
    assert prepare_objective_branch(source, destination, configuration) == provenance
    with pytest.raises(ValueError, match="only run name"):
        prepare_continuation(source, tmp_path / "not_a_continuation", configuration)
    best = ValidationCheckpointRecord.model_validate_json(
        (destination / "best_validation.json").read_bytes()
    )
    assert best.checkpoint_path == destination / "best_projector.safetensors"
    assert (destination / "best_projector.safetensors").read_bytes() == (
        source / "best_projector.safetensors"
    ).read_bytes()


def test_branch_rejects_architecture_changes_and_unfinished_parent(tmp_path: Path) -> None:
    source = tmp_path / "parent"
    original, state = finished_parent(source)
    configuration = original.model_copy(update={"name": "branch", "max_optimizer_updates": 6775})
    with pytest.raises(ValueError, match="only name, objective"):
        prepare_objective_branch(
            source,
            tmp_path / "changed",
            configuration.model_copy(update={"seed": original.seed + 1}),
        )
    (source / "checkpoint" / "state.json").write_text(
        state.model_copy(update={"offset": 3}).model_dump_json(), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="completed full-pass"):
        prepare_objective_branch(source, tmp_path / "unfinished", configuration)


def test_job_paths_have_portable_posix_serialization() -> None:
    job = ObjectiveBranchJob(
        source_run=Path("/workspace/speech-projector/parent"),
        data_root=Path("/workspace/speech-projector/data"),
        output_root=Path("/workspace/speech-projector/results"),
        configuration=sweep_runs(38193, 1)[0],
    )
    serialized = job.model_dump_json()
    assert "/workspace/speech-projector/parent" in serialized
    assert "\\\\workspace" not in serialized
    assert ObjectiveBranchJob.model_validate_json(serialized) == job
