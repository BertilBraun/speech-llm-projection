"""Validate the typed utterance API against immutable completed-run outputs."""

import argparse
import hashlib
import math
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from pydantic import Field
from safetensors.torch import load_file

from speech_projector.data import load_examples
from speech_projector.evaluation import EvaluationCondition, ExampleLoss
from speech_projector.inputs import SpeechInput, TranscriptInput, UtteranceInput
from speech_projector.llm import FrozenQwen
from speech_projector.models import (
    Example,
    GradientCheck,
    Record,
    RunResult,
    SampleGeneration,
    Split,
)
from speech_projector.projectors import Projector
from speech_projector.training import gradient_sanity, load_features


class ParityConfiguration(Record):
    run_directory: Path
    manifest: Path
    baseline_directory: Path
    output_path: Path
    candidate_commit: str
    expected_reference_commit: str
    examples: int = Field(default=8, gt=0)
    generations: int = Field(default=2, gt=0)
    loss_tolerance: float = Field(default=1e-5, gt=0)


class LossParity(Record):
    example_id: str
    condition: EvaluationCondition
    reference_cross_entropy: float
    candidate_cross_entropy: float
    absolute_error: float
    reference_target_tokens: int
    candidate_target_tokens: int


class GenerationParity(Record):
    example_id: str
    condition: EvaluationCondition
    reference_response: str
    candidate_response: str
    exact_match: bool


class ParityReport(Record):
    config: ParityConfiguration
    run_name: str
    reference_commit: str
    checkpoint_path: Path
    checkpoint_sha256: str
    losses: tuple[LossParity, ...]
    generations: tuple[GenerationParity, ...]
    gradient: GradientCheck
    elapsed_seconds: float


def find_loss(
    records: Sequence[ExampleLoss], example: Example, condition: EvaluationCondition
) -> ExampleLoss:
    for record in records:
        if record.example_id == example.example_id and record.condition == condition:
            return record
    raise ValueError(f"Missing reference {condition.value} loss for {example.example_id}")


def find_generation(
    records: Sequence[SampleGeneration], example: Example, condition: EvaluationCondition
) -> SampleGeneration:
    for record in records:
        if record.example_id == example.example_id and record.condition == condition.value:
            if record.history != example.history or record.user_transcript != example.user_text:
                raise ValueError(f"Reference prompt differs for {example.example_id}")
            return record
    raise ValueError(f"Missing reference {condition.value} generation for {example.example_id}")


def assert_parity(report: ParityReport) -> None:
    if any(
        not math.isfinite(check.candidate_cross_entropy)
        or not math.isfinite(check.reference_cross_entropy)
        or check.absolute_error > report.config.loss_tolerance
        or check.reference_target_tokens != check.candidate_target_tokens
        for check in report.losses
    ):
        raise ValueError("Candidate teacher-forced loss or target masking differs from reference")
    if not all(check.exact_match for check in report.generations):
        raise ValueError("Candidate greedy generation differs from reference")
    assert report.gradient.projector_gradient_norm > 0
    assert report.gradient.projector_changed
    assert report.gradient.llm_weights_unchanged
    assert not report.gradient.llm_has_gradients


def run_parity(config: ParityConfiguration) -> ParityReport:
    started = time.perf_counter()
    result = RunResult.model_validate_json(
        (config.run_directory / "result.json").read_text(encoding="utf-8")
    )
    if result.git_commit != config.expected_reference_commit:
        raise ValueError("The reference run was produced by an unexpected source commit")
    if config.generations > config.examples:
        raise ValueError("Generation count must not exceed the checked example count")
    examples = load_examples(config.manifest, Split.VALIDATION, config.examples)
    checkpoint_digest = hashlib.sha256(result.checkpoint_path.read_bytes()).hexdigest()
    wrapper = FrozenQwen(result.config, torch.device("cuda"))
    projector = Projector(result.config.projector).to(wrapper.device)
    projector.load_state_dict(load_file(str(result.checkpoint_path), device=str(wrapper.device)))
    projector.eval()
    wrapper.model.eval()
    losses: list[LossParity] = []
    generations: list[GenerationParity] = []
    references = (
        (EvaluationCondition.SPEECH, config.run_directory / "validation", "evaluation"),
        (EvaluationCondition.TEXT, config.baseline_directory, "text"),
    )
    with torch.no_grad():
        for condition, directory, name in references:
            reference_losses = tuple(
                ExampleLoss.model_validate_json(line)
                for line in (directory / f"{name}_losses.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            )
            reference_generations = tuple(
                SampleGeneration.model_validate_json(line)
                for line in (directory / f"{name}_generations.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            )
            for index, example in enumerate(examples):
                utterance: UtteranceInput
                match condition:
                    case EvaluationCondition.SPEECH:
                        utterance = SpeechInput(projector(load_features(example, wrapper.device)))
                    case EvaluationCondition.TEXT:
                        utterance = TranscriptInput(example.user_text)
                reference = find_loss(reference_losses, example, condition)
                candidate_loss = float(wrapper.loss(example, utterance))
                losses.append(
                    LossParity(
                        example_id=example.example_id,
                        condition=condition,
                        reference_cross_entropy=reference.cross_entropy,
                        candidate_cross_entropy=candidate_loss,
                        absolute_error=abs(candidate_loss - reference.cross_entropy),
                        reference_target_tokens=reference.target_tokens,
                        candidate_target_tokens=wrapper.target_token_count(example),
                    )
                )
                if index < config.generations:
                    reference_generation = find_generation(
                        reference_generations, example, condition
                    )
                    candidate_generation = wrapper.generate(example, utterance)
                    generations.append(
                        GenerationParity(
                            example_id=example.example_id,
                            condition=condition,
                            reference_response=reference_generation.generated_response,
                            candidate_response=candidate_generation,
                            exact_match=candidate_generation
                            == reference_generation.generated_response,
                        )
                    )
    gradient = gradient_sanity(wrapper, projector, examples[0])
    assert hashlib.sha256(result.checkpoint_path.read_bytes()).hexdigest() == checkpoint_digest
    report = ParityReport(
        config=config,
        run_name=result.config.name,
        reference_commit=result.git_commit,
        checkpoint_path=result.checkpoint_path,
        checkpoint_sha256=checkpoint_digest,
        losses=tuple(losses),
        generations=tuple(generations),
        gradient=gradient,
        elapsed_seconds=time.perf_counter() - started,
    )
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    assert_parity(report)
    print(report.model_dump_json(indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--reference-commit", required=True)
    arguments = parser.parse_args()
    run_parity(
        ParityConfiguration(
            run_directory=arguments.run_dir,
            manifest=arguments.manifest,
            baseline_directory=arguments.baseline_dir,
            output_path=arguments.output,
            candidate_commit=arguments.candidate_commit,
            expected_reference_commit=arguments.reference_commit,
        )
    )


if __name__ == "__main__":
    main()
