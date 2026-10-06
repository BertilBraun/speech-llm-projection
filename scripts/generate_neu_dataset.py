"""Generate fresh Neu utterances or paired concise teacher replies with frozen vLLM."""

import argparse
from enum import Enum
from pathlib import Path

from scripts.inventory_results import write_record
from speech_projector.emotional_generation import (
    CachedGeneration,
    GenerationBackend,
    VllmRuntimeConfig,
)
from speech_projector.emotional_vllm import VllmTextGenerator
from speech_projector.neu_dataset import NeuDatasetProvenance, validate_neu_provenance
from speech_projector.neu_generation import (
    NeuGenerationConfig,
    generate_neu_drafts,
    generate_neu_targets,
    render_neu_pilot,
    summarize_neu_generation,
)


class Stage(str, Enum):
    DRAFTS = "drafts"
    TARGETS = "targets"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", type=Stage, choices=tuple(Stage), required=True)
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    config = NeuGenerationConfig.model_validate_json(arguments.config.read_bytes())
    if config.generation.backend != GenerationBackend.VLLM:
        raise ValueError("Neu dataset entry point requires frozen vLLM generation")
    directory = config.generation.output_directory / config.generation.trace_subdirectory
    directory.mkdir(parents=True, exist_ok=True)
    provenance_path = directory / "neu_generation_config.json"
    if provenance_path.exists():
        if NeuGenerationConfig.model_validate_json(provenance_path.read_bytes()) != config:
            raise ValueError("Neu generation source/configuration identity changed on resume")
    else:
        write_record(provenance_path, config)
    validate_neu_provenance(
        config.generation.output_directory,
        NeuDatasetProvenance(
            configuration=config.generation.dataset,
            previous_utterances=config.previous_utterances,
            previous_sha256=config.previous_sha256,
        ),
    )
    generator = VllmTextGenerator(config.generation, VllmRuntimeConfig())
    cached = CachedGeneration(config.generation, generator.generate)
    match arguments.stage:
        case Stage.DRAFTS:
            generate_neu_drafts(config, cached, arguments.limit)
        case Stage.TARGETS:
            generate_neu_targets(config, cached, arguments.limit)
    render_neu_pilot(config.generation.output_directory)
    print(summarize_neu_generation(config).model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
