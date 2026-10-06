"""Run the real frozen-Qwen single-versus-batched backward gate on cached inputs."""

import argparse
from pathlib import Path

import torch

from scripts.inventory_results import write_record
from speech_projector.launcher import initialize_projector
from speech_projector.llm import FrozenQwen
from speech_projector.models import Example, RunConfig, Split
from speech_projector.overnight_preparation import load_records
from speech_projector.training import (
    BatchedParityPolicy,
    batched_gradient_sanity,
    validate_batched_parity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = RunConfig.model_validate_json(arguments.config.read_bytes())
    wrapper = FrozenQwen(configuration, torch.device("cuda"))
    training = [
        row
        for row in load_records(arguments.manifest, Example)
        if row.split == Split.TRAIN and row.feature_path.is_file()
    ]
    longest = max(training, key=wrapper.target_token_count)
    shortest = min(training, key=wrapper.target_token_count)
    check = batched_gradient_sanity(
        wrapper, initialize_projector(configuration, wrapper.device), [longest, shortest]
    )
    write_record(arguments.output, check)
    print(check.model_dump_json(indent=2), flush=True)
    validate_batched_parity(check, BatchedParityPolicy())


if __name__ == "__main__":
    main()
