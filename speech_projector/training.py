"""Single-GPU, resumable projector-only optimization."""

import hashlib
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from speech_projector.llm import FrozenQwen
from speech_projector.models import Example, GradientCheck, Record, RunConfig, TrainLog
from speech_projector.projectors import Projector


class TrainingState(Record):
    epoch: int
    offset: int
    step: int
    examples_seen: int
    target_tokens_seen: int
    elapsed_seconds: float
    initial_validation_loss: float
    initial_training_loss: float
    final_fixed_training_loss: float | None = None
    final_training_loss: float


@dataclass(frozen=True)
class TrainingOutcome:
    steps: int
    runtime_seconds: float
    examples_seen: int
    target_tokens_seen: int
    peak_vram_gb: float
    initial_validation_loss: float
    initial_training_loss: float
    final_fixed_training_loss: float
    final_training_loss: float
    checkpoint_path: Path


def load_features(example: Example, device: torch.device) -> Tensor:
    features: Tensor = torch.load(example.feature_path, map_location=device, weights_only=True)
    return features


def weights_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def validation_loss(wrapper: FrozenQwen, projector: Projector, examples: list[Example]) -> float:
    wrapper.model.eval()
    projector.eval()
    total_loss = 0.0
    total_tokens = 0
    for example in examples:
        speech = projector(load_features(example, wrapper.device))
        count = wrapper.target_token_count(example)
        total_loss += wrapper.loss(example, speech_embeddings=speech).item() * count
        total_tokens += count
    return total_loss / total_tokens


def gradient_sanity(wrapper: FrozenQwen, projector: Projector, example: Example) -> GradientCheck:
    wrapper.model.train(wrapper.config.gradient_checkpointing)
    projector.train()
    assert not any(parameter.requires_grad for parameter in wrapper.model.parameters())
    before_llm = weights_digest(wrapper.model)
    before_projector = weights_digest(projector)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=wrapper.config.learning_rate)
    if wrapper.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(wrapper.device)
        torch.cuda.synchronize(wrapper.device)
    start = time.monotonic()
    loss = wrapper.loss(
        example, speech_embeddings=projector(load_features(example, wrapper.device))
    )
    loss.backward()
    gradient_norm = math.sqrt(
        sum(
            parameter.grad.float().square().sum().item()
            for parameter in projector.parameters()
            if parameter.grad is not None
        )
    )
    optimizer.step()
    if wrapper.device.type == "cuda":
        torch.cuda.synchronize(wrapper.device)
    result = GradientCheck(
        loss=loss.item(),
        projector_gradient_norm=gradient_norm,
        projector_changed=weights_digest(projector) != before_projector,
        frozen_parameters=sum(parameter.numel() for parameter in wrapper.model.parameters()),
        llm_has_gradients=any(
            parameter.grad is not None for parameter in wrapper.model.parameters()
        ),
        llm_weights_unchanged=weights_digest(wrapper.model) == before_llm,
        peak_vram_gb=(
            torch.cuda.max_memory_allocated(wrapper.device) / 1e9
            if wrapper.device.type == "cuda"
            else 0.0
        ),
        step_seconds=time.monotonic() - start,
    )
    assert result.projector_gradient_norm > 0
    assert result.projector_changed
    assert not result.llm_has_gradients
    assert result.llm_weights_unchanged
    projector.zero_grad(set_to_none=True)
    return result


def _save_checkpoint(
    directory: Path,
    projector: Projector,
    optimizer: torch.optim.AdamW,
    state: TrainingState,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "projector.pending.safetensors"
    save_file(projector.state_dict(), str(temporary))
    temporary.replace(directory / "projector.safetensors")
    torch.save(optimizer.state_dict(), directory / "optimizer.pending.pt")
    (directory / "optimizer.pending.pt").replace(directory / "optimizer.pt")
    (directory / "state.pending.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")
    (directory / "state.pending.json").replace(directory / "state.json")
    return directory / "projector.safetensors"


def train_run(
    config: RunConfig,
    examples: list[Example],
    validation: list[Example],
    output_dir: Path,
    wrapper: FrozenQwen,
    projector: Projector,
) -> TrainingOutcome:
    if config.microbatch_size != 1:
        raise ValueError("This single-example pipeline requires microbatch_size=1")
    if len(examples) != config.train_examples:
        raise ValueError("Training subset length differs from run configuration")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    if config_path.exists():
        previous = RunConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("Cannot resume a run with a different configuration")
    else:
        config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    (output_dir / "subset.jsonl").write_text(
        "\n".join(example.model_dump_json() for example in examples) + "\n", encoding="utf-8"
    )
    projector.to(wrapper.device)
    optimizer = torch.optim.AdamW(
        projector.parameters(), lr=config.learning_rate, weight_decay=0.01
    )
    checkpoint_dir = output_dir / "checkpoint"
    fixed_training_examples = examples[:128]
    if (checkpoint_dir / "state.json").exists():
        state = TrainingState.model_validate_json(
            (checkpoint_dir / "state.json").read_text(encoding="utf-8")
        )
        projector.load_state_dict(load_file(str(checkpoint_dir / "projector.safetensors")))
        optimizer.load_state_dict(
            torch.load(
                checkpoint_dir / "optimizer.pt", weights_only=True, map_location=wrapper.device
            )
        )
    else:
        initial = validation_loss(wrapper, projector, validation)
        initial_training = validation_loss(wrapper, projector, fixed_training_examples)
        state = TrainingState(
            epoch=0,
            offset=0,
            step=0,
            examples_seen=0,
            target_tokens_seen=0,
            elapsed_seconds=0.0,
            initial_validation_loss=initial,
            initial_training_loss=initial_training,
            final_training_loss=0.0,
        )
        _save_checkpoint(checkpoint_dir, projector, optimizer, state)
    if wrapper.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(wrapper.device)
    start = time.monotonic()
    prior_elapsed = state.elapsed_seconds
    recent_losses: list[float] = []
    for epoch in range(state.epoch, config.epochs):
        order = list(range(len(examples)))
        random.Random(config.seed + epoch).shuffle(order)
        offset = state.offset if epoch == state.epoch else 0
        for group_start in range(offset, len(order), config.gradient_accumulation):
            selected = order[group_start : group_start + config.gradient_accumulation]
            wrapper.model.train(config.gradient_checkpointing)
            projector.train()
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            group_tokens = 0
            for index in selected:
                example = examples[index]
                loss = wrapper.loss(
                    example, speech_embeddings=projector(load_features(example, wrapper.device))
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at example {example.example_id}")
                (loss / len(selected)).backward()
                group_loss += loss.item()
                group_tokens += wrapper.target_token_count(example)
            torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            optimizer.step()
            step = state.step + 1
            final_loss = group_loss / len(selected)
            recent_losses.append(final_loss)
            state = TrainingState(
                epoch=epoch,
                offset=group_start + len(selected),
                step=step,
                examples_seen=state.examples_seen + len(selected),
                target_tokens_seen=state.target_tokens_seen + group_tokens,
                elapsed_seconds=prior_elapsed + time.monotonic() - start,
                initial_validation_loss=state.initial_validation_loss,
                initial_training_loss=state.initial_training_loss,
                final_training_loss=sum(recent_losses[-20:]) / len(recent_losses[-20:]),
            )
            measured_validation = None
            if step % config.evaluation_interval == 0:
                measured_validation = validation_loss(wrapper, projector, validation)
            log = TrainLog(
                step=step,
                epoch=epoch,
                training_loss=final_loss,
                validation_loss=measured_validation,
                elapsed_seconds=state.elapsed_seconds,
                examples_seen=state.examples_seen,
                target_tokens_seen=state.target_tokens_seen,
            )
            with (output_dir / "train.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(log.model_dump_json() + "\n")
            print(log.model_dump_json(), flush=True)
            if step % config.checkpoint_interval == 0:
                _save_checkpoint(checkpoint_dir, projector, optimizer, state)
        state = state.model_copy(update={"epoch": epoch + 1, "offset": 0})
        _save_checkpoint(checkpoint_dir, projector, optimizer, state)
    if state.final_fixed_training_loss is None:
        state = state.model_copy(
            update={
                "final_fixed_training_loss": validation_loss(
                    wrapper, projector, fixed_training_examples
                )
            }
        )
    state = state.model_copy(update={"elapsed_seconds": prior_elapsed + time.monotonic() - start})
    path = _save_checkpoint(checkpoint_dir, projector, optimizer, state)
    assert state.final_fixed_training_loss is not None
    return TrainingOutcome(
        steps=state.step,
        runtime_seconds=state.elapsed_seconds,
        examples_seen=state.examples_seen,
        target_tokens_seen=state.target_tokens_seen,
        peak_vram_gb=(
            torch.cuda.max_memory_allocated(wrapper.device) / 1e9
            if wrapper.device.type == "cuda"
            else 0.0
        ),
        initial_validation_loss=state.initial_validation_loss,
        initial_training_loss=state.initial_training_loss,
        final_fixed_training_loss=state.final_fixed_training_loss,
        final_training_loss=state.final_training_loss,
        checkpoint_path=path,
    )
