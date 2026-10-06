"""Canonical experiment models and serialization boundaries."""

from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ExperimentStage(str, Enum):
    V0 = "V0"
    V1 = "V1"
    V2 = "V2"
    V3 = "V3"


class EvaluationCondition(str, Enum):
    TEXT = "text"
    ASR = "asr"
    SPEECH = "speech"
    SHUFFLED_SPEECH = "shuffled_speech"
    ZERO_SPEECH = "zero_speech"
    SPEECH_NO_HISTORY = "speech_no_history"
    SHUFFLED_SPEECH_NO_HISTORY = "shuffled_speech_no_history"


class GenerationKind(str, Enum):
    COMPLETED = "completed"
    TOKEN_LIMIT = "token_limit"


class GenerationDetails(Record):
    kind: GenerationKind
    token_ids: tuple[int, ...]


class Turn(Record):
    role: Role
    text: str


class Example(Record):
    example_id: str
    dialogue_id: str
    split: Split
    history: tuple[Turn, ...]
    user_text: str
    target_text: str
    audio_path: Path
    duration: float = Field(gt=0, le=30)
    domain: str
    emotion: str
    feature_path: Path


class Architecture(str, Enum):
    LINEAR = "linear"
    MLP = "mlp"
    CONV = "conv"


class ProjectorDimensions(Record):
    compression_factor: int = Field(ge=1)
    encoder_dimension: int = 768
    embedding_dimension: int = 2048
    native_rate: float = 50.0


class LinearProjectorConfig(ProjectorDimensions):
    architecture: Literal[Architecture.LINEAR] = Architecture.LINEAR


class MlpProjectorConfig(ProjectorDimensions):
    architecture: Literal[Architecture.MLP] = Architecture.MLP
    hidden_dimension: int = 1024


class ConvProjectorConfig(ProjectorDimensions):
    architecture: Literal[Architecture.CONV] = Architecture.CONV
    hidden_dimension: int = 1024


ProjectorConfig: TypeAlias = Annotated[
    LinearProjectorConfig | MlpProjectorConfig | ConvProjectorConfig,
    Field(discriminator="architecture"),
]


class SystemPromptConfig(Record):
    kind: Literal["system"] = "system"
    system_text: str = Field(
        default=(
            "You are a helpful conversational assistant. Reply naturally to the user's utterance."
        ),
        min_length=1,
    )


class ChatPromptConfig(Record):
    kind: Literal["chat"] = "chat"


PromptConfig: TypeAlias = Annotated[
    SystemPromptConfig | ChatPromptConfig, Field(discriminator="kind")
]


class GreedyDecodingConfig(Record):
    kind: Literal["greedy"] = "greedy"


class SamplingDecodingConfig(Record):
    kind: Literal["sampling"] = "sampling"
    temperature: float = Field(default=1.0, gt=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=20, ge=0)
    min_p: float = Field(default=0.0, ge=0, le=1)
    presence_penalty: float = Field(default=2.0, ge=0, le=2)
    repetition_penalty: float = Field(default=1.0, gt=0)


DecodingConfig: TypeAlias = Annotated[
    GreedyDecodingConfig | SamplingDecodingConfig, Field(discriminator="kind")
]


class RunConfig(Record):
    name: str
    stage: ExperimentStage
    seed: int = 42
    train_examples: int = Field(gt=0)
    validation_examples: int = 128
    test_examples: int = 128
    training_validation_examples: int | None = Field(default=None, gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    microbatch_size: int = 1
    gradient_accumulation: int = 8
    history_turns: int = 2
    max_history_tokens: int = 256
    max_target_tokens: int = 128
    max_new_tokens: int = 96
    sequence_length_multiple: int = Field(default=64, ge=1)
    gradient_checkpointing: bool = True
    evaluation_interval: int = 100
    checkpoint_interval: int = 100
    qualitative_examples: int = 16
    semantic_examples: int = 48
    generation_batch_size: int = Field(default=1, ge=1)
    conditioning_examples: int = Field(default=32, ge=1)
    model_name: str = "Qwen/Qwen3.5-2B"
    speech_model_name: str = "openai/whisper-small"
    prompt: PromptConfig = SystemPromptConfig()
    decoding: DecodingConfig = GreedyDecodingConfig()
    projector: ProjectorConfig


class EvaluationMetrics(Record):
    examples: int
    target_tokens: int
    cross_entropy: float
    perplexity: float
    semantic_similarity: float | None = None
    generated_examples: int = 0
    generation_seconds: float = 0.0
    generated_tokens: int = 0
    completed_generations: int | None = None
    token_limited_generations: int | None = None
    shuffled_audio_cross_entropy: float | None = None
    zero_audio_cross_entropy: float | None = None
    no_history_cross_entropy: float | None = None
    no_history_shuffled_cross_entropy: float | None = None
    evaluation_seconds: float = 0.0


class AsrTranscript(Record):
    example_id: str
    text: str


class SampleGeneration(Record):
    example_id: str
    dialogue_id: str
    condition: EvaluationCondition
    history: tuple[Turn, ...]
    user_transcript: str
    asr_transcript: str | None = None
    gold_response: str
    generated_response: str
    generation: GenerationDetails | None = None
    duration: float
    pseudo_tokens: int | None = None
    semantic_similarity: float | None = None


class TrainLog(Record):
    step: int
    epoch: int
    training_loss: float
    validation_loss: float | None = None
    elapsed_seconds: float
    examples_seen: int
    target_tokens_seen: int


class RunResult(Record):
    config: RunConfig
    git_commit: str
    train_examples: int
    validation_examples: int
    test_examples: int
    projector_parameters: int
    pseudo_tokens_per_second: float
    mean_pseudo_tokens: float
    optimizer: str = "AdamW"
    steps: int
    runtime_seconds: float
    peak_vram_gb: float
    examples_per_second: float
    target_tokens_per_second: float
    initial_validation_loss: float
    initial_training_loss: float
    final_fixed_training_loss: float
    final_training_loss: float
    validation: EvaluationMetrics
    test: EvaluationMetrics | None = None
    checkpoint_path: Path


class GradientCheck(Record):
    loss: float
    projector_gradient_norm: float
    projector_changed: bool
    frozen_parameters: int
    llm_has_gradients: bool
    llm_weights_unchanged: bool
    peak_vram_gb: float
    step_seconds: float


class ExperimentFailure(Record):
    run_name: str
    attempt: int
    exception_type: str
    message: str
    traceback: str
    timestamp: float


class SuiteState(Record):
    completed: tuple[str, ...]
    running: str | None
    failed: tuple[str, ...]
    started_at: float
    updated_at: float


class ExperimentDecision(Record):
    selected_run: str
    candidate_runs: tuple[str, ...]
    rationale: str
