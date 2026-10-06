"""Controlled experiment matrix, updated using measured runtime."""

from speech_projector.models import (
    ConvProjectorConfig,
    ExperimentStage,
    LinearProjectorConfig,
    MlpProjectorConfig,
    ProjectorConfig,
    RunConfig,
)


def make_run(
    name: str,
    stage: ExperimentStage,
    train_examples: int,
    epochs: int,
    projector: ProjectorConfig,
) -> RunConfig:
    return RunConfig(
        name=name,
        stage=stage,
        train_examples=train_examples,
        epochs=epochs,
        learning_rate=0.001,
        projector=projector,
    )


def feasibility_run() -> RunConfig:
    return make_run(
        "v0_256_mlp_10hz", ExperimentStage.V0, 256, 5, MlpProjectorConfig(compression_factor=5)
    )


def scaling_runs() -> list[RunConfig]:
    return [
        make_run(
            f"v1_{count}_mlp_10hz",
            ExperimentStage.V1,
            count,
            2,
            MlpProjectorConfig(compression_factor=5),
        )
        for count in (1000, 3000, 10000)
    ]


def compression_runs(train_examples: int) -> list[RunConfig]:
    return [
        make_run(
            f"v2_{train_examples}_mlp_{factor}x",
            ExperimentStage.V2,
            train_examples,
            2,
            MlpProjectorConfig(compression_factor=factor),
        )
        for factor in (2, 10, 20)
    ]


def architecture_runs(train_examples: int, factor: int) -> list[RunConfig]:
    return [
        make_run(
            f"v3_{train_examples}_{projector.architecture.value}_{factor}x",
            ExperimentStage.V3,
            train_examples,
            2,
            projector,
        )
        for projector in (
            LinearProjectorConfig(compression_factor=factor),
            ConvProjectorConfig(compression_factor=factor),
        )
    ]
