"""Measure audio conditioning on a completed run's exact training subset."""

import argparse
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from speech_projector.data import load_examples
from speech_projector.evaluation import (
    EvaluationCondition,
    SemanticEvaluator,
    evaluate,
    save_evaluation,
)
from speech_projector.llm import FrozenQwen
from speech_projector.models import EvaluationMetrics, Example, Record, RunResult, Split
from speech_projector.projectors import Projector


class ProbeSummary(Record):
    run_name: str
    source_git_commit: str
    training_examples: int
    generated_primary_examples: int
    initial_fixed_training_cross_entropy: float
    final_fixed_training_cross_entropy: float
    training: EvaluationMetrics
    untrained_validation: EvaluationMetrics
    trained_validation: EvaluationMetrics
    total_seconds: float


def training_examples(run_directory: Path, manifest: Path) -> list[Example]:
    subset = [
        Example.model_validate_json(line)
        for line in (run_directory / "subset.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ][:32]
    if not subset:
        raise ValueError("The saved training subset is empty")
    manifest_examples = load_examples(manifest, Split.TRAIN)
    allowed = {example.example_id for example in manifest_examples}
    if any(example.split != Split.TRAIN or example.example_id not in allowed for example in subset):
        raise ValueError("Saved training examples do not belong to the supplied training manifest")
    if len({example.dialogue_id for example in subset}) < 2:
        raise ValueError("The training probe needs two dialogues for shuffled-audio controls")
    return subset


def run_probe(run_directory: Path, manifest: Path) -> None:
    started = time.perf_counter()
    result = RunResult.model_validate_json(
        (run_directory / "result.json").read_text(encoding="utf-8")
    )
    examples = training_examples(run_directory, manifest)
    config = result.config.model_copy(update={"qualitative_examples": 8, "semantic_examples": 8})
    wrapper = FrozenQwen(config, torch.device("cuda"))
    projector = Projector(config.projector).to(wrapper.device)
    projector.load_state_dict(load_file(str(result.checkpoint_path), device=str(wrapper.device)))
    outcome = evaluate(
        wrapper,
        projector,
        examples,
        config,
        EvaluationCondition.SPEECH,
        semantic_evaluator=SemanticEvaluator(),
        diagnostics=True,
    )
    output_directory = run_directory / "overfit_probe"
    save_evaluation(outcome, output_directory, "speech")
    (output_directory / "examples.jsonl").write_text(
        "\n".join(example.model_dump_json() for example in examples) + "\n", encoding="utf-8"
    )
    validation = load_examples(manifest, Split.VALIDATION, 32)
    control_config = result.config.model_copy(
        update={"qualitative_examples": 0, "semantic_examples": 0}
    )
    wrapper.config = control_config
    torch.manual_seed(control_config.seed)
    untrained_projector = Projector(control_config.projector).to(wrapper.device)
    untrained_validation = evaluate(
        wrapper,
        untrained_projector,
        validation,
        control_config,
        EvaluationCondition.SPEECH,
        diagnostics=True,
    )
    save_evaluation(untrained_validation, run_directory / "untrained_validation_probe", "speech")
    trained_validation = evaluate(
        wrapper,
        projector,
        validation,
        control_config,
        EvaluationCondition.SPEECH,
        diagnostics=True,
    )
    save_evaluation(trained_validation, run_directory / "trained_validation_probe", "speech")
    summary = ProbeSummary(
        run_name=config.name,
        source_git_commit=result.git_commit,
        training_examples=len(examples),
        generated_primary_examples=outcome.metrics.generated_examples,
        initial_fixed_training_cross_entropy=result.initial_training_loss,
        final_fixed_training_cross_entropy=result.final_fixed_training_loss,
        training=outcome.metrics,
        untrained_validation=untrained_validation.metrics,
        trained_validation=trained_validation.metrics,
        total_seconds=time.perf_counter() - started,
    )
    (output_directory / "summary.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )
    print(summary.model_dump_json(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args()
    run_probe(arguments.run_dir, arguments.manifest)


if __name__ == "__main__":
    main()
