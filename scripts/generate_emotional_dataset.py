"""Generate resumable paired emotional utterances and frozen-Qwen teacher targets."""

import argparse
from enum import Enum
from pathlib import Path

import torch

from speech_projector.emotional_generation import (
    CachedGeneration,
    EmotionalGenerationConfig,
    FrozenTextGenerator,
    GenerationBackend,
    generate_drafts,
    generate_teacher_targets,
    summarize_generation,
)


class Stage(str, Enum):
    DRAFTS = "drafts"
    TARGETS = "targets"
    BOTH = "both"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", type=Stage, choices=tuple(stage.value for stage in Stage), default=Stage.BOTH
    )
    arguments = parser.parse_args()
    config = EmotionalGenerationConfig.model_validate_json(arguments.config.read_bytes())
    if config.backend != GenerationBackend.HUGGING_FACE:
        raise ValueError("This entry point requires the Hugging Face backend")
    generator = FrozenTextGenerator(config, torch.device("cuda"))
    cached = CachedGeneration(config, generator.generate)
    if arguments.stage in (Stage.DRAFTS, Stage.BOTH):
        generate_drafts(config, cached)
    if arguments.stage in (Stage.TARGETS, Stage.BOTH):
        generate_teacher_targets(config, cached)
    print(summarize_generation(config).model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
