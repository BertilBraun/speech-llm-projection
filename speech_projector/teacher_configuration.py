"""Controlled configurations for transcript-teacher distillation."""

from speech_projector.generation import RETRY_TEACHER_TOKEN_CAP
from speech_projector.models import (
    ChatPromptConfig,
    ExperimentStage,
    LinearProjectorConfig,
    MlpProjectorConfig,
    ProjectorConfig,
    RunConfig,
)

TEACHER_PROMPT = ChatPromptConfig()


def teacher_run(
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
        validation_examples=512,
        test_examples=512,
        training_validation_examples=128,
        epochs=epochs,
        learning_rate=0.001,
        gradient_accumulation=8,
        evaluation_interval=250,
        checkpoint_interval=100,
        max_target_tokens=RETRY_TEACHER_TOKEN_CAP + 1,
        max_new_tokens=RETRY_TEACHER_TOKEN_CAP,
        qualitative_examples=16,
        semantic_examples=512,
        conditioning_examples=128,
        prompt=TEACHER_PROMPT,
        projector=projector,
    )


def teacher_feasibility_run() -> RunConfig:
    return teacher_run(
        "teacher_v0_256_mlp_10hz",
        ExperimentStage.V0,
        256,
        5,
        MlpProjectorConfig(compression_factor=5),
    ).model_copy(
        update={
            "validation_examples": 32,
            "test_examples": 32,
            "training_validation_examples": 32,
            "semantic_examples": 16,
            "conditioning_examples": 32,
            "evaluation_interval": 80,
        }
    )


def teacher_compression_runs() -> tuple[RunConfig, ...]:
    return tuple(
        teacher_run(
            f"teacher_20000_mlp_{rate}hz",
            ExperimentStage.V1 if factor == 5 else ExperimentStage.V2,
            20000,
            2,
            MlpProjectorConfig(compression_factor=factor),
        )
        for factor, rate in ((5, "10"), (2, "25"), (10, "5"), (20, "2.5"))
    )


def teacher_linear_run(compression_factor: int) -> RunConfig:
    return teacher_run(
        f"teacher_20000_linear_{compression_factor}x",
        ExperimentStage.V3,
        20000,
        2,
        LinearProjectorConfig(compression_factor=compression_factor),
    )


def teacher_scaling_runs() -> tuple[RunConfig, ...]:
    return tuple(
        teacher_run(
            f"teacher_{count}_mlp_10hz",
            ExperimentStage.V1,
            count,
            2,
            MlpProjectorConfig(compression_factor=5),
        )
        for count in (1000, 3000, 10000)
    )
