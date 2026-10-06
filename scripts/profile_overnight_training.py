"""Measure frozen-model backward memory and update throughput before a sweep."""

import argparse
import math
import time
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

import torch
from pydantic import Field

from speech_projector.inputs import SpeechInput
from speech_projector.llm import FrozenQwen
from speech_projector.models import Example, Record, RunConfig, Split
from speech_projector.overnight_preparation import load_records
from speech_projector.projectors import Projector
from speech_projector.training import load_features, weights_digest


class ProfilePopulation(str, Enum):
    LONGEST = "longest"
    REPRESENTATIVE = "representative"


class BackwardProfile(Record):
    kind: Literal["completed"] = "completed"
    population: ProfilePopulation
    microbatch_size: int
    effective_batch_size: int
    examples: tuple[str, ...]
    target_tokens: tuple[int, ...]
    speech_pseudo_tokens: tuple[int, ...]
    measured_updates: int
    update_seconds: float
    peak_allocated_gb: float
    final_loss: float
    gradient_norm: float
    projector_changed: bool
    llm_has_gradients: bool
    llm_weights_unchanged: bool


class ProfileOutOfMemory(Record):
    kind: Literal["out_of_memory"] = "out_of_memory"
    population: ProfilePopulation
    microbatch_size: int
    examples: tuple[str, ...]
    peak_allocated_gb: float
    message: str


ProfileResult: TypeAlias = Annotated[
    BackwardProfile | ProfileOutOfMemory, Field(discriminator="kind")
]


class TrainingProfile(Record):
    configuration: RunConfig
    frozen_weights_sha256: str
    measurements: tuple[ProfileResult, ...]


def select_profile_examples(
    wrapper: FrozenQwen, examples: list[Example], population: ProfilePopulation
) -> tuple[Example, ...]:
    available = [example for example in examples if example.feature_path.is_file()]
    if len(available) < 8:
        raise ValueError("Training profiling requires at least eight cached examples")
    match population:
        case ProfilePopulation.LONGEST:
            longest_targets = sorted(available, key=wrapper.target_token_count, reverse=True)[:4]
            selected_ids = {example.example_id for example in longest_targets}
            longest_audio = sorted(
                (example for example in available if example.example_id not in selected_ids),
                key=lambda example: example.duration,
                reverse=True,
            )[:4]
            return tuple(longest_targets + longest_audio)
        case ProfilePopulation.REPRESENTATIVE:
            generator = torch.Generator().manual_seed(wrapper.config.seed)
            indices = torch.randperm(len(available), generator=generator).tolist()[:8]
            return tuple(available[index] for index in indices)


def profile_backward(
    wrapper: FrozenQwen,
    examples: tuple[Example, ...],
    population: ProfilePopulation,
    microbatch_size: int,
    measured_updates: int,
    frozen_digest: str,
) -> ProfileResult:
    if 8 % microbatch_size:
        raise ValueError("Profile microbatch must divide the effective batch of eight")
    torch.manual_seed(wrapper.config.seed)
    projector = Projector(wrapper.config.projector).to(wrapper.device)
    initial_projector_digest = weights_digest(projector)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=wrapper.config.learning_rate)
    wrapper.model.train(wrapper.config.gradient_checkpointing)
    feature_states = tuple(load_features(example, wrapper.device) for example in examples)
    torch.cuda.reset_peak_memory_stats(wrapper.device)
    update_seconds: list[float] = []
    final_loss = 0.0
    gradient_norm = 0.0
    try:
        for update in range(measured_updates + 1):
            torch.cuda.synchronize(wrapper.device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            final_loss = 0.0
            for offset in range(0, 8, microbatch_size):
                selected = examples[offset : offset + microbatch_size]
                inputs = tuple(
                    SpeechInput(projector(features))
                    for features in feature_states[offset : offset + microbatch_size]
                )
                loss = (
                    wrapper.loss(selected[0], inputs[0])
                    if microbatch_size == 1
                    else wrapper.loss_batch(selected, inputs)
                )
                (loss * len(selected) / 8).backward()
                final_loss += loss.item() * len(selected) / 8
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0))
            optimizer.step()
            torch.cuda.synchronize(wrapper.device)
            if update:
                update_seconds.append(time.perf_counter() - started)
        return BackwardProfile(
            population=population,
            microbatch_size=microbatch_size,
            effective_batch_size=8,
            examples=tuple(example.example_id for example in examples),
            target_tokens=tuple(wrapper.target_token_count(example) for example in examples),
            speech_pseudo_tokens=tuple(
                math.ceil(states.shape[0] / wrapper.config.projector.compression_factor)
                for states in feature_states
            ),
            measured_updates=measured_updates,
            update_seconds=sum(update_seconds) / len(update_seconds),
            peak_allocated_gb=torch.cuda.max_memory_allocated(wrapper.device) / 1e9,
            final_loss=final_loss,
            gradient_norm=gradient_norm,
            projector_changed=weights_digest(projector) != initial_projector_digest,
            llm_has_gradients=any(
                parameter.grad is not None for parameter in wrapper.model.parameters()
            ),
            llm_weights_unchanged=weights_digest(wrapper.model) == frozen_digest,
        )
    except torch.OutOfMemoryError as error:
        optimizer.zero_grad(set_to_none=True)
        return ProfileOutOfMemory(
            population=population,
            microbatch_size=microbatch_size,
            examples=tuple(example.example_id for example in examples),
            peak_allocated_gb=torch.cuda.max_memory_allocated(wrapper.device) / 1e9,
            message=str(error),
        )
    finally:
        del optimizer, projector, feature_states
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatches", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--updates", type=int, default=3)
    arguments = parser.parse_args()
    if arguments.updates < 1 or any(size not in (1, 2, 4) for size in arguments.microbatches):
        raise ValueError("Use positive profile updates and microbatches1/2/4")
    config = RunConfig.model_validate_json(arguments.config.read_bytes())
    wrapper = FrozenQwen(config, torch.device("cuda"))
    examples = [
        row for row in load_records(arguments.manifest, Example) if row.split == Split.TRAIN
    ]
    digest = weights_digest(wrapper.model)
    results: list[ProfileResult] = []
    for population in ProfilePopulation:
        selected = select_profile_examples(wrapper, examples, population)
        for size in arguments.microbatches:
            measured = profile_backward(
                wrapper, selected, population, size, arguments.updates, digest
            )
            results.append(measured)
            print(measured.model_dump_json(), flush=True)
    report = TrainingProfile(
        configuration=config, frozen_weights_sha256=digest, measurements=tuple(results)
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
