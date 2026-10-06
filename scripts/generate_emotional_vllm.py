"""Run the emotional generation pipeline in the separately installed vLLM runtime."""

import argparse
from pathlib import Path

from scripts.generate_emotional_dataset import Stage
from speech_projector.emotional_generation import (
    CachedGeneration,
    EmotionalGenerationConfig,
    GenerationBackend,
    VllmRuntimeConfig,
    generate_drafts,
    generate_teacher_targets,
    summarize_generation,
)
from speech_projector.emotional_vllm import VllmTextGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", type=Stage, choices=tuple(stage.value for stage in Stage), default=Stage.BOTH
    )
    arguments = parser.parse_args()
    config = EmotionalGenerationConfig.model_validate_json(arguments.config.read_bytes())
    if config.backend != GenerationBackend.VLLM:
        raise ValueError("This entry point requires the vLLM backend")
    generator = VllmTextGenerator(config, VllmRuntimeConfig())
    cached = CachedGeneration(config, generator.generate)
    if arguments.stage in (Stage.DRAFTS, Stage.BOTH):
        generate_drafts(config, cached)
    if arguments.stage in (Stage.TARGETS, Stage.BOTH):
        generate_teacher_targets(config, cached)
    print(summarize_generation(config).model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
