"""An auxiliary objective continuation must preserve Adam moments and its exact sample cursor."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from speech_projector.models import (
    OrdinaryResponseKLObjective,
    ResponseCrossEntropyObjective,
    Split,
    TranscriptMixtureObjective,
)
from speech_projector.objective_branch import (
    ObjectiveBranchData,
    ObjectiveBranchJob,
    load_branch_data,
)
from speech_projector.objective_continuation import prepare_objective_continuation
from speech_projector.overnight_configuration import sweep_runs
from speech_projector.overnight_data import OrdinaryExampleSource
from speech_projector.training import (
    TrainingResourceRecord,
    TrainingState,
    ValidationCheckpointRecord,
)
from tests.test_overnight_evaluation import candidate, example
from tests.test_overnight_report import result


def completed_branch(
    directory: Path,
    objective: ResponseCrossEntropyObjective
    | TranscriptMixtureObjective
    | OrdinaryResponseKLObjective,
) -> tuple[ObjectiveBranchJob, ObjectiveBranchData, TrainingState]:
    source = directory / "parent"
    checkpoint = source / "checkpoint"
    checkpoint.mkdir(parents=True)
    data_root = directory / "data"
    data_root.mkdir()
    feature_path = data_root / "cached.pt"
    torch.save(torch.ones(2, 768), feature_path)
    training = [
        example(f"ordinary:{index}", f"Question {index}.").model_copy(
            update={"split": Split.TRAIN, "feature_path": feature_path}
        )
        for index in range(17)
    ]
    training_sources = tuple(
        OrdinaryExampleSource(
            example_id=row.example_id,
            source_manifest=Path("original.jsonl"),
            source_example_id=row.example_id,
        )
        for row in training
    )
    validation = (
        example("heldout", "A held-out question.").model_copy(
            update={"feature_path": feature_path}
        ),
    )
    validation_sources = (
        OrdinaryExampleSource(
            example_id="heldout",
            source_manifest=Path("original.jsonl"),
            source_example_id="heldout",
        ),
    )
    (data_root / "examples.jsonl").write_bytes(
        b"".join(row.model_dump_json().encode() + b"\n" for row in training)
    )
    (source / "subset.jsonl").write_bytes((data_root / "examples.jsonl").read_bytes())
    (data_root / "sources.jsonl").write_bytes(
        b"".join(row.model_dump_json().encode() + b"\n" for row in training_sources)
    )
    (data_root / "fixed_validation.jsonl").write_bytes(
        validation[0].model_dump_json().encode() + b"\n"
    )
    (data_root / "fixed_validation_sources.jsonl").write_bytes(
        validation_sources[0].model_dump_json().encode() + b"\n"
    )
    original = sweep_runs(17, 1)[0].model_copy(
        update={
            "name": "parent",
            "max_optimizer_updates": 4,
            "learning_rate": 2e-4,
            "objective": objective,
        }
    )
    (source / "config.json").write_text(original.model_dump_json(), encoding="utf-8")
    state = TrainingState(
        epoch=1,
        offset=8,
        step=4,
        examples_seen=25,
        target_tokens_seen=777,
        elapsed_seconds=1000,
        initial_validation_loss=2,
        initial_training_loss=2.1,
        final_validation_loss=1.1,
        final_fixed_training_loss=1.05,
        final_training_loss=1.2,
    )
    (checkpoint / "state.json").write_text(state.model_dump_json(), encoding="utf-8")
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], lr=original.learning_rate)
    for _ in range(4):
        optimizer.zero_grad()
        parameter.sum().backward()
        optimizer.step()
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    save_file({"weight": parameter.detach()}, checkpoint / "projector.safetensors")
    save_file({"weight": parameter.detach()}, source / "best_projector.safetensors")
    (source / "best_validation.json").write_text(
        ValidationCheckpointRecord(
            step=4, cross_entropy=1.1, checkpoint_path=source / "best_projector.safetensors"
        ).model_dump_json(),
        encoding="utf-8",
    )
    (source / "training_resources.json").write_text(
        TrainingResourceRecord(peak_vram_gb=6.5).model_dump_json(), encoding="utf-8"
    )
    selected = candidate("parent", 1.1, 1.1, 0.1, 10).model_copy(update={"configuration": original})
    (source / "candidate.json").write_text(selected.model_dump_json(), encoding="utf-8")
    (source / "result.json").write_text(
        result().model_copy(update={"config": original, "steps": state.step}).model_dump_json(),
        encoding="utf-8",
    )
    job = ObjectiveBranchJob(
        source_run=source,
        data_root=data_root,
        output_root=directory / "continued",
        configuration=original.model_copy(
            update={"name": "second_pass", "max_optimizer_updates": 6}
        ),
    )
    return (
        job,
        ObjectiveBranchData(training, training_sources, validation, validation_sources),
        state,
    )


@pytest.mark.parametrize(
    "objective",
    (ResponseCrossEntropyObjective(), TranscriptMixtureObjective(), OrdinaryResponseKLObjective()),
)
def test_second_pass_preserves_auxiliary_objective_optimizer_and_sample_offset(
    tmp_path: Path,
    objective: ResponseCrossEntropyObjective
    | TranscriptMixtureObjective
    | OrdinaryResponseKLObjective,
) -> None:
    job, data, parent_state = completed_branch(tmp_path, objective)
    assert load_branch_data(job) == data
    proof = prepare_objective_continuation(job, data)
    destination = job.output_root / job.configuration.name
    assert proof.continuation.source_configuration.objective == objective
    assert proof.continuation.continuation_configuration.objective == objective
    assert proof.continuation.source_configuration.learning_rate == job.configuration.learning_rate
    resumed = TrainingState.model_validate_json(
        (destination / "checkpoint" / "state.json").read_bytes()
    )
    assert resumed == parent_state.model_copy(
        update={"final_validation_loss": None, "final_fixed_training_loss": None}
    )
    for filename in ("optimizer.pt", "projector.safetensors"):
        assert (job.source_run / "checkpoint" / filename).read_bytes() == (
            destination / "checkpoint" / filename
        ).read_bytes()
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.AdamW([parameter], lr=job.configuration.learning_rate)
    optimizer.load_state_dict(
        torch.load(destination / "checkpoint" / "optimizer.pt", weights_only=True)
    )
    assert optimizer.state[parameter]["step"].item() == 4
    assert optimizer.state[parameter]["exp_avg"].abs().sum().item() > 0
    assert optimizer.state[parameter]["exp_avg_sq"].abs().sum().item() > 0
    assert optimizer.param_groups[0]["lr"] == 2e-4
    assert len(proof.data_artifacts) == 7
    assert prepare_objective_continuation(job, data) == proof
    assert (
        TrainingState.model_validate_json(
            (job.source_run / "checkpoint" / "state.json").read_bytes()
        )
        == parent_state
    )


@pytest.mark.parametrize(
    "change", ("objective", "learning_rate", "ceiling", "subset", "cursor", "unfinished")
)
def test_continuation_rejects_changed_supervision_policy_or_inconsistent_parent(
    tmp_path: Path, change: str
) -> None:
    job, data, state = completed_branch(tmp_path, TranscriptMixtureObjective())
    if change == "objective":
        job = job.model_copy(
            update={
                "configuration": job.configuration.model_copy(
                    update={"objective": ResponseCrossEntropyObjective()}
                )
            }
        )
    elif change == "learning_rate":
        job = job.model_copy(
            update={"configuration": job.configuration.model_copy(update={"learning_rate": 0.001})}
        )
    elif change == "ceiling":
        job = job.model_copy(
            update={
                "configuration": job.configuration.model_copy(update={"max_optimizer_updates": 5})
            }
        )
    elif change == "subset":
        data.training[0] = data.training[0].model_copy(update={"target_text": "Changed target."})
    else:
        state = state.model_copy(
            update={"offset": 7} if change == "cursor" else {"final_validation_loss": None}
        )
        (job.source_run / "checkpoint" / "state.json").write_text(
            state.model_dump_json(), encoding="utf-8"
        )
    with pytest.raises(
        ValueError, match="only run name|second complete|saved subset|cursor|completed"
    ):
        prepare_objective_continuation(job, data)
