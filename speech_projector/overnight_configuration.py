"""Controlled compression and stacking configurations for the mixed corpus."""

from speech_projector.models import (
    ChatPromptConfig,
    ExperimentStage,
    GreedyDecodingConfig,
    MlpProjectorConfig,
    RunConfig,
    StackedMlpProjectorConfig,
)


def sweep_runs(training_examples: int, microbatch_size: int) -> tuple[RunConfig, ...]:
    if microbatch_size not in (1, 2, 4):
        raise ValueError("Controlled microbatch size must be1,2or4")
    settings = (
        ("overnight_mean_10hz", MlpProjectorConfig(compression_factor=5)),
        ("overnight_mean_5hz", MlpProjectorConfig(compression_factor=10)),
        ("overnight_stack_5hz", StackedMlpProjectorConfig(compression_factor=10)),
        ("overnight_stack_10hz", StackedMlpProjectorConfig(compression_factor=5)),
        ("overnight_mean_2p5hz", MlpProjectorConfig(compression_factor=20)),
        ("overnight_mean_25hz", MlpProjectorConfig(compression_factor=2)),
    )
    return tuple(
        RunConfig(
            name=name,
            stage=ExperimentStage.V2,
            train_examples=training_examples,
            validation_examples=128,
            test_examples=128,
            epochs=3,
            max_optimizer_updates=2000,
            learning_rate=0.001,
            microbatch_size=microbatch_size,
            gradient_accumulation=8 // microbatch_size,
            max_target_tokens=4097,
            max_new_tokens=256,
            evaluation_interval=250,
            checkpoint_interval=100,
            qualitative_examples=24,
            semantic_examples=24,
            generation_batch_size=8,
            conditioning_examples=24,
            prompt=ChatPromptConfig(),
            decoding=GreedyDecodingConfig(),
            projector=projector,
        )
        for name, projector in settings
    )
