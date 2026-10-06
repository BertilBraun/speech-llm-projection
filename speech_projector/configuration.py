"""Controlled experiment matrix, updated using measured runtime."""

from speech_projector.models import Architecture, ExperimentStage, ProjectorConfig, RunConfig


def make_run(
    name: str,
    stage: ExperimentStage,
    train_examples: int,
    epochs: int,
    compression_factor: int,
    architecture: Architecture,
) -> RunConfig:
    return RunConfig(
        name=name,
        stage=stage,
        train_examples=train_examples,
        epochs=epochs,
        learning_rate=0.001,
        projector=ProjectorConfig(
            architecture=architecture,
            compression_factor=compression_factor,
        ),
    )


def feasibility_run() -> RunConfig:
    return make_run("v0_256_mlp_10hz", ExperimentStage.V0, 256, 5, 5, Architecture.MLP)


def scaling_runs() -> list[RunConfig]:
    return [
        make_run(f"v1_{count}_mlp_10hz", ExperimentStage.V1, count, 2, 5, Architecture.MLP)
        for count in (1000, 3000, 10000)
    ]


def compression_runs(train_examples: int) -> list[RunConfig]:
    return [
        make_run(
            f"v2_{train_examples}_mlp_{factor}x",
            ExperimentStage.V2,
            train_examples,
            2,
            factor,
            Architecture.MLP,
        )
        for factor in (2, 10, 20)
    ]


def architecture_runs(train_examples: int, factor: int) -> list[RunConfig]:
    return [
        make_run(
            f"v3_{train_examples}_{architecture.value}_{factor}x",
            ExperimentStage.V3,
            train_examples,
            2,
            factor,
            architecture,
        )
        for architecture in (Architecture.LINEAR, Architecture.CONV)
    ]
