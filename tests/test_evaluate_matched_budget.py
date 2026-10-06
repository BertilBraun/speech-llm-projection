from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

from scripts import evaluate_matched_budget as evaluation_script
from scripts.evaluate_matched_budget import (
    MatchedBudgetEvaluation,
    evaluate_matched_budget,
    read_model_revisions,
    read_verified_snapshot,
    validate_budget,
)
from scripts.package_results import FileArtifact, ModelRevision, file_digest, write_record
from scripts.snapshot_first_epoch import SnapshotConfiguration, SnapshotProvenance
from speech_projector.llm import FrozenQwen
from speech_projector.models import (
    EvaluationMetrics,
    ExperimentStage,
    MlpProjectorConfig,
    RunConfig,
    RunResult,
)
from speech_projector.training import TrainingState


@dataclass(frozen=True)
class BudgetFixture:
    directory: Path
    reference_directory: Path
    snapshot: SnapshotProvenance
    reference: RunResult
    manifest: Path
    model_revisions_path: Path


@pytest.fixture
def budget_fixture(tmp_path: Path) -> BudgetFixture:
    config = RunConfig(
        name="v1_20000_mlp_10hz",
        stage=ExperimentStage.V1,
        train_examples=20000,
        epochs=2,
        learning_rate=0.001,
        projector=MlpProjectorConfig(compression_factor=5),
    )
    write_record(tmp_path / "config.json", config)
    snapshot_directory = tmp_path / "matched_budget_step2500"
    snapshot_directory.mkdir()
    state = TrainingState(
        epoch=1,
        offset=0,
        step=2500,
        examples_seen=20000,
        target_tokens_seen=500000,
        elapsed_seconds=2500,
        initial_validation_loss=3,
        initial_training_loss=3,
        final_training_loss=2,
    )
    write_record(snapshot_directory / "state.json", state)
    (snapshot_directory / "projector.safetensors").write_bytes(b"test-local checkpoint")
    artifacts = tuple(
        FileArtifact(
            path=Path(name),
            source_path=tmp_path / "checkpoint" / name,
            bytes=(snapshot_directory / name).stat().st_size,
            sha256=file_digest(snapshot_directory / name),
        )
        for name in ("state.json", "projector.safetensors")
    )
    snapshot = SnapshotProvenance(
        configuration=SnapshotConfiguration(
            run_directory=tmp_path, output_directory=snapshot_directory, source_git_commit="a" * 40
        ),
        captured_at=datetime(2026, 10, 6, tzinfo=timezone.utc),
        run_config=config,
        training_state=state,
        artifacts=artifacts,
    )
    write_record(snapshot_directory / "provenance.json", snapshot)
    metrics = EvaluationMetrics(
        examples=128,
        target_tokens=3456,
        cross_entropy=1.8,
        perplexity=6.05,
        semantic_similarity=0.4,
        generated_examples=48,
    )
    reference_config = config.model_copy(
        update={"name": "v1_10000_mlp_10hz", "train_examples": 10000}
    )
    reference = RunResult(
        config=reference_config,
        git_commit="b" * 40,
        train_examples=10000,
        validation_examples=128,
        test_examples=128,
        projector_parameters=2888192,
        pseudo_tokens_per_second=10,
        mean_pseudo_tokens=71,
        steps=2500,
        runtime_seconds=2700,
        peak_vram_gb=4.2,
        examples_per_second=7,
        target_tokens_per_second=200,
        initial_validation_loss=3,
        initial_training_loss=3,
        final_fixed_training_loss=1.5,
        final_training_loss=1.7,
        validation=metrics,
        test=metrics,
        checkpoint_path=tmp_path / "reference" / "checkpoint" / "projector.safetensors",
    )
    reference_directory = tmp_path / "reference"
    write_record(reference_directory / "result.json", reference)
    manifest = tmp_path / "examples.jsonl"
    manifest.write_bytes(b"test-local manifest identity")
    revisions = tuple(
        ModelRevision(model_name=name, snapshot_revisions=("c" * 40,), main_revision="c" * 40)
        for name in (config.model_name, config.speech_model_name)
    )
    revisions_path = tmp_path / "model_revisions.jsonl"
    revisions_path.write_text(
        "\n".join(item.model_dump_json() for item in revisions), encoding="utf-8"
    )
    return BudgetFixture(
        tmp_path, reference_directory, snapshot, reference, manifest, revisions_path
    )


def test_snapshot_integrity_and_equal_budget_are_accepted(budget_fixture: BudgetFixture) -> None:
    snapshot = read_verified_snapshot(budget_fixture.directory, budget_fixture.reference)
    assert snapshot == budget_fixture.snapshot
    assert snapshot.training_state.step == 2500
    assert snapshot.training_state.examples_seen == 20000


@pytest.mark.parametrize("filename", ["projector.safetensors", "state.json"])
def test_snapshot_rejects_changed_saved_content(
    budget_fixture: BudgetFixture, filename: str
) -> None:
    (budget_fixture.directory / "matched_budget_step2500" / filename).write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity failed"):
        read_verified_snapshot(budget_fixture.directory, budget_fixture.reference)


@pytest.mark.parametrize("step, exposures", [(2499, 20000), (2500, 19992), (5000, 40000)])
def test_budget_rejects_unmatched_checkpoint(
    budget_fixture: BudgetFixture, step: int, exposures: int
) -> None:
    snapshot = budget_fixture.snapshot.model_copy(
        update={
            "training_state": budget_fixture.snapshot.training_state.model_copy(
                update={"step": step, "examples_seen": exposures}
            )
        }
    )
    with pytest.raises(ValueError, match="2500 / 20000"):
        validate_budget(snapshot, budget_fixture.reference, snapshot.run_config)


def test_budget_rejects_hyperparameter_change(budget_fixture: BudgetFixture) -> None:
    changed = budget_fixture.reference.model_copy(
        update={
            "config": budget_fixture.reference.config.model_copy(update={"learning_rate": 0.002})
        }
    )
    with pytest.raises(ValueError, match="share all configuration"):
        validate_budget(budget_fixture.snapshot, changed, budget_fixture.snapshot.run_config)


def test_model_inventory_requires_both_frozen_models(budget_fixture: BudgetFixture) -> None:
    budget_fixture.model_revisions_path.write_text(
        ModelRevision(
            model_name=budget_fixture.snapshot.run_config.model_name,
            snapshot_revisions=("c" * 40,),
            main_revision="c" * 40,
        ).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Qwen and Whisper"):
        read_model_revisions(
            budget_fixture.model_revisions_path, budget_fixture.snapshot.run_config
        )


def test_resume_reuses_verified_summary_without_loading_models(
    budget_fixture: BudgetFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = FileArtifact(
        path=Path(budget_fixture.manifest.name),
        source_path=budget_fixture.manifest,
        bytes=budget_fixture.manifest.stat().st_size,
        sha256=file_digest(budget_fixture.manifest),
    )
    metrics = budget_fixture.reference.validation
    summary = MatchedBudgetEvaluation(
        snapshot=budget_fixture.snapshot,
        reference=budget_fixture.reference,
        evaluation_git_commit="d" * 40,
        manifest=manifest,
        model_revisions=read_model_revisions(
            budget_fixture.model_revisions_path, budget_fixture.snapshot.run_config
        ),
        frozen_llm_weights_sha256="e" * 64,
        validation=metrics,
        test=metrics,
        evaluation_seconds=100,
        peak_vram_gb=4.2,
    )
    write_record(budget_fixture.directory / "matched_budget" / "summary.json", summary)

    def reject_initialization(config: RunConfig, device: torch.device) -> FrozenQwen:
        raise AssertionError("Verified completed evaluation must not reload the LLM")

    monkeypatch.setattr(evaluation_script, "FrozenQwen", reject_initialization)
    actual = evaluate_matched_budget(
        budget_fixture.directory,
        budget_fixture.reference_directory,
        budget_fixture.manifest,
        budget_fixture.model_revisions_path,
        torch.device("cpu"),
    )
    assert actual == summary
    budget_fixture.manifest.write_bytes(b"changed manifest")
    with pytest.raises(ValueError, match="different input provenance"):
        evaluate_matched_budget(
            budget_fixture.directory,
            budget_fixture.reference_directory,
            budget_fixture.manifest,
            budget_fixture.model_revisions_path,
            torch.device("cpu"),
        )
