"""Capture deterministic initialization fingerprints without loading a GPU model."""

import argparse
import hashlib
from pathlib import Path

import torch
from pydantic import TypeAdapter

from speech_projector.configuration import architecture_runs, scaling_runs
from speech_projector.models import Architecture, Record
from speech_projector.projectors import Projector
from speech_projector.training import weights_digest


class ProjectorFingerprint(Record):
    architecture: Architecture
    parameter_count: int
    state_keys: tuple[str, ...]
    weights_sha256: str
    output_sha256: str


def fingerprints() -> tuple[ProjectorFingerprint, ...]:
    observations: list[ProjectorFingerprint] = []
    for config in (scaling_runs()[0], *architecture_runs(1000, 5)):
        torch.manual_seed(42)
        projector = Projector(config.projector)
        features = torch.randn(23, config.projector.encoder_dimension)
        with torch.no_grad():
            output = projector(features)
        observations.append(
            ProjectorFingerprint(
                architecture=config.projector.architecture,
                parameter_count=projector.parameter_count,
                state_keys=tuple(projector.state_dict()),
                weights_sha256=weights_digest(projector),
                output_sha256=hashlib.sha256(output.numpy().tobytes()).hexdigest(),
            )
        )
    return tuple(observations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.write_bytes(
        TypeAdapter(tuple[ProjectorFingerprint, ...]).dump_json(fingerprints(), indent=2)
    )


if __name__ == "__main__":
    main()
