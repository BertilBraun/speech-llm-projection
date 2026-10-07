"""Preserve selected sweep artifacts while continuing their exact optimizer trajectory."""

import shutil
from pathlib import Path

from scripts.inventory_results import stable_digest
from scripts.package_results import FileArtifact
from speech_projector.models import (
    OrdinaryResponseKLObjective,
    Record,
    ResponseCrossEntropyObjective,
    RunConfig,
    TranscriptMixtureObjective,
)
from speech_projector.overnight_preparation import artifact, write_immutable
from speech_projector.training import TrainingState, ValidationCheckpointRecord


class ContinuationProvenance(Record):
    source_configuration: RunConfig
    continuation_configuration: RunConfig
    source_artifacts: tuple[FileArtifact, ...]


def prepare_continuation(
    source: Path, destination: Path, config: RunConfig
) -> ContinuationProvenance:
    original = RunConfig.model_validate_json((source / "config.json").read_bytes())
    if (
        original.model_copy(
            update={"name": config.name, "max_optimizer_updates": config.max_optimizer_updates}
        )
        != config
    ):
        raise ValueError("Continuation may change only run name and total update ceiling")
    return _prepare_snapshot(source, destination, original, config, "continuation.json")


def prepare_objective_branch(
    source: Path, destination: Path, config: RunConfig
) -> ContinuationProvenance:
    original = RunConfig.model_validate_json((source / "config.json").read_bytes())
    if original != original.model_copy(update={"objective": ResponseCrossEntropyObjective()}):
        raise ValueError("Objective branches require the same response-CE parent")
    if (
        original.model_copy(
            update={
                "name": config.name,
                "max_optimizer_updates": config.max_optimizer_updates,
                "learning_rate": config.learning_rate,
                "objective": config.objective,
            }
        )
        != config
    ):
        raise ValueError(
            "Objective branch may change only name, objective, learning rate and ceiling"
        )
    match config.objective:
        case TranscriptMixtureObjective() | OrdinaryResponseKLObjective():
            if config.microbatch_size != 1:
                raise ValueError("Lexical branches require single-example microbatches")
        case ResponseCrossEntropyObjective():
            pass
    state = TrainingState.model_validate_json((source / "checkpoint" / "state.json").read_bytes())
    if (
        state.epoch < 1
        or state.offset != 0
        or state.final_validation_loss is None
        or state.final_fixed_training_loss is None
    ):
        raise ValueError("Objective branches require a completed full-pass parent checkpoint")
    return _prepare_snapshot(source, destination, original, config, "objective_branch.json")


def _prepare_snapshot(
    source: Path,
    destination: Path,
    original: RunConfig,
    config: RunConfig,
    receipt_filename: str,
) -> ContinuationProvenance:
    state_path = source / "checkpoint" / "state.json"
    state = TrainingState.model_validate_json(state_path.read_bytes())
    if config.max_optimizer_updates is None or config.max_optimizer_updates <= state.step:
        raise ValueError("Continuation needs a larger cumulative update ceiling")
    paths = (
        source / "config.json",
        state_path,
        source / "checkpoint" / "projector.safetensors",
        source / "checkpoint" / "optimizer.pt",
        source / "training_resources.json",
    )
    records = tuple(artifact(path) for path in paths)
    provenance = ContinuationProvenance(
        source_configuration=original, continuation_configuration=config, source_artifacts=records
    )
    provenance_path = destination / receipt_filename
    if provenance_path.exists():
        if ContinuationProvenance.model_validate_json(provenance_path.read_bytes()) != provenance:
            raise ValueError("Existing continuation differs from its immutable source checkpoint")
        return provenance
    if destination.exists():
        raise ValueError("Continuation destination exists without a committed provenance receipt")
    pending = destination.with_name(destination.name + ".pending")
    if pending.exists():
        raise ValueError("An interrupted continuation copy must be inspected before retrying")
    (pending / "checkpoint").mkdir(parents=True)
    for filename in ("projector.safetensors", "optimizer.pt"):
        shutil.copyfile(source / "checkpoint" / filename, pending / "checkpoint" / filename)
    shutil.copyfile(source / "training_resources.json", pending / "training_resources.json")
    continued_state = state.model_copy(
        update={"final_validation_loss": None, "final_fixed_training_loss": None}
    )
    write_immutable(
        pending / "checkpoint" / "state.json", continued_state.model_dump_json(indent=2).encode()
    )
    if (source / "best_validation.json").exists():
        best = ValidationCheckpointRecord.model_validate_json(
            (source / "best_validation.json").read_bytes()
        )
        shutil.copyfile(best.checkpoint_path, pending / "best_projector.safetensors")
        copied_best = best.model_copy(
            update={"checkpoint_path": destination / "best_projector.safetensors"}
        )
        write_immutable(
            pending / "best_validation.json", copied_best.model_dump_json(indent=2).encode()
        )
    for record in records:
        if stable_digest(record.path) != (record.bytes, record.sha256):
            raise ValueError("Source checkpoint changed during continuation snapshot")
    write_immutable(pending / receipt_filename, provenance.model_dump_json(indent=2).encode())
    pending.replace(destination)
    return provenance
