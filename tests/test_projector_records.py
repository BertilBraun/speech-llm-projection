from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from scripts.migrate_linear_records import (
    LinearProjectorRecordWithUnusedDimension,
    LinearRunConfigWithUnusedDimension,
    LinearRunResultWithUnusedDimension,
    migrate_linear_run,
)
from speech_projector.models import (
    Architecture,
    ConvProjectorConfig,
    EvaluationMetrics,
    ExperimentStage,
    LinearProjectorConfig,
    MlpProjectorConfig,
    ProjectorConfig,
    RunConfig,
    RunResult,
)


@pytest.mark.parametrize("architecture", (Architecture.MLP, Architecture.CONV))
def test_existing_nonlinear_json_retains_hidden_dimension(architecture: Architecture) -> None:
    serialized = (
        '{"architecture":"'
        + architecture.value
        + '","compression_factor":5,"encoder_dimension":768,"embedding_dimension":2048,'
        '"hidden_dimension":1024,"native_rate":50.0}'
    )
    config = TypeAdapter(ProjectorConfig).validate_json(serialized)
    match config:
        case MlpProjectorConfig() | ConvProjectorConfig():
            assert config.hidden_dimension == 1024
        case LinearProjectorConfig():
            pytest.fail("An existing nonlinear record was parsed as linear")
    assert config.architecture == architecture


def test_linear_hidden_dimension_is_rejected_at_boundary() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(ProjectorConfig).validate_json(
            '{"architecture":"linear","compression_factor":5,"hidden_dimension":1024}'
        )


def test_linear_migration_preserves_original_bytes_and_other_values(tmp_path: Path) -> None:
    config = LinearRunConfigWithUnusedDimension(
        name="linear",
        stage=ExperimentStage.V3,
        train_examples=1000,
        epochs=2,
        learning_rate=0.001,
        projector=LinearProjectorRecordWithUnusedDimension(
            compression_factor=5, hidden_dimension=1024
        ),
    )
    metrics = EvaluationMetrics(
        examples=128, target_tokens=1000, cross_entropy=1.23, perplexity=3.42
    )
    result = LinearRunResultWithUnusedDimension(
        config=config,
        git_commit="source",
        train_examples=1000,
        validation_examples=128,
        test_examples=128,
        projector_parameters=1576448,
        pseudo_tokens_per_second=10,
        mean_pseudo_tokens=70,
        steps=250,
        runtime_seconds=300,
        peak_vram_gb=4,
        examples_per_second=6.7,
        target_tokens_per_second=200,
        initial_validation_loss=3,
        initial_training_loss=3,
        final_fixed_training_loss=1.3,
        final_training_loss=1.4,
        validation=metrics,
        test=metrics,
        checkpoint_path=tmp_path / "projector.safetensors",
    )
    config_bytes = config.model_dump_json(indent=2).encode() + b"\n"
    result_bytes = result.model_dump_json(indent=2).encode() + b"\n"
    (tmp_path / "config.json").write_bytes(config_bytes)
    (tmp_path / "result.json").write_bytes(result_bytes)
    with pytest.raises(ValueError, match="writers"):
        migrate_linear_run(tmp_path, writers_stopped=False)
    provenance = migrate_linear_run(tmp_path, writers_stopped=True)
    assert (tmp_path / "original_records" / "config.json").read_bytes() == config_bytes
    assert (tmp_path / "original_records" / "result.json").read_bytes() == result_bytes
    migrated_config = RunConfig.model_validate_json((tmp_path / "config.json").read_bytes())
    migrated_result = RunResult.model_validate_json((tmp_path / "result.json").read_bytes())
    assert migrated_config.projector == LinearProjectorConfig(compression_factor=5)
    assert migrated_result.config == migrated_config
    assert migrated_result.validation == result.validation
    assert migrated_result.test == result.test
    assert migrated_result.checkpoint_path == result.checkpoint_path
    assert migrated_result.steps == result.steps
    assert migrated_result.git_commit == result.git_commit
    assert len(provenance.original_records) == len(provenance.migrated_records) == 2
    with pytest.raises(ValidationError):
        migrate_linear_run(tmp_path, writers_stopped=True)
