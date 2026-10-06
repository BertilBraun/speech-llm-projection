"""Run a bounded CPU-only intended-delivery diagnostic after frozen feature caching."""

import argparse
from pathlib import Path

from pydantic import TypeAdapter

from speech_projector.acoustic_tone_probe import ToneProbeConfig, run_tone_probe
from speech_projector.models import Example
from speech_projector.overnight_data import NeuEmotionalExampleSource, SourceSidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    config = ToneProbeConfig.model_validate_json(arguments.config.read_bytes())
    examples = tuple(
        Example.model_validate_json(line) for line in config.manifest.read_bytes().splitlines()
    )
    adapter = TypeAdapter(SourceSidecar)
    sidecars = tuple(
        adapter.validate_json(line) for line in config.sidecar.read_bytes().splitlines()
    )
    sources: list[NeuEmotionalExampleSource] = []
    for source in sidecars:
        match source:
            case NeuEmotionalExampleSource():
                sources.append(source)
            case _:
                continue
    print(run_tone_probe(config, examples, sources).model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
