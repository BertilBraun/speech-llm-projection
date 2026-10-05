"""Run and resume the controlled single-GPU experiment suite."""

import argparse
import subprocess
import time
import traceback
from pathlib import Path

import torch

from speech_projector.cache import CacheConfig, extract_features
from speech_projector.configuration import (
    architecture_runs,
    compression_runs,
    feasibility_run,
    scaling_runs,
)
from speech_projector.data import download_audio, load_examples
from speech_projector.evaluation import (
    EvaluationCondition,
    SemanticEvaluator,
    evaluate,
    load_asr_transcripts,
    save_evaluation,
)
from speech_projector.llm import FrozenQwen
from speech_projector.models import (
    Example,
    ExperimentFailure,
    RunConfig,
    RunResult,
    Split,
    SuiteState,
)
from speech_projector.projectors import Projector
from speech_projector.report import aggregate_report
from speech_projector.training import gradient_sanity, train_run


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def initialize_projector(config: RunConfig, device: torch.device) -> Projector:
    torch.manual_seed(config.seed)
    return Projector(config.projector).to(device)


def smoke(
    config: RunConfig, wrapper: FrozenQwen, examples: list[Example], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    projector = initialize_projector(config, wrapper.device)
    result = gradient_sanity(wrapper, projector, examples[0])
    (output_dir / "gradient_check.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
    with torch.no_grad():
        features = torch.load(
            examples[0].feature_path, weights_only=True, map_location=wrapper.device
        )
        speech = projector(features)
        generated = wrapper.generate(examples[0], speech_embeddings=speech)
        text_generated = wrapper.generate(examples[0], transcript=examples[0].user_text)
    lines = [
        f"User: {examples[0].user_text}",
        f"Gold: {examples[0].target_text}",
        f"Untrained speech: {generated}",
        f"Text: {text_generated}",
    ]
    (output_dir / "generations.txt").write_text("\n\n".join(lines), encoding="utf-8")
    print(result.model_dump_json(), flush=True)
    print("\n".join(lines), flush=True)


def run_experiment(
    config: RunConfig,
    wrapper: FrozenQwen,
    examples: list[Example],
    validation: list[Example],
    test: list[Example],
    output_root: Path,
    semantic_evaluator: SemanticEvaluator,
) -> RunResult:
    directory = output_root / config.name
    result_path = directory / "result.json"
    if result_path.exists():
        return RunResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    wrapper.config = config
    projector = initialize_projector(config, wrapper.device)
    outcome = train_run(
        config, examples[: config.train_examples], validation, directory, wrapper, projector
    )
    evaluated = evaluate(
        wrapper,
        projector,
        validation,
        config,
        EvaluationCondition.SPEECH,
        semantic_evaluator=semantic_evaluator,
    )
    save_evaluation(evaluated, directory / "validation")
    tested = evaluate(
        wrapper,
        projector,
        test,
        config,
        EvaluationCondition.SPEECH,
        semantic_evaluator=semantic_evaluator,
    )
    save_evaluation(tested, directory / "test")
    mean_tokens = sum(
        (
            int(torch.load(example.feature_path, weights_only=True).shape[0])
            + config.projector.compression_factor
            - 1
        )
        // config.projector.compression_factor
        for example in validation
    ) / len(validation)
    result = RunResult(
        config=config,
        git_commit=git_revision(),
        train_examples=config.train_examples,
        validation_examples=len(validation),
        test_examples=len(test),
        projector_parameters=projector.parameter_count,
        pseudo_tokens_per_second=config.projector.native_rate / config.projector.compression_factor,
        mean_pseudo_tokens=mean_tokens,
        steps=outcome.steps,
        runtime_seconds=outcome.runtime_seconds,
        peak_vram_gb=outcome.peak_vram_gb,
        examples_per_second=outcome.examples_seen / outcome.runtime_seconds,
        target_tokens_per_second=outcome.target_tokens_seen / outcome.runtime_seconds,
        initial_validation_loss=outcome.initial_validation_loss,
        initial_training_loss=outcome.initial_training_loss,
        final_fixed_training_loss=outcome.final_fixed_training_loss,
        final_training_loss=outcome.final_training_loss,
        validation=evaluated.metrics,
        test=tested.metrics,
        checkpoint_path=outcome.checkpoint_path,
    )
    result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def ensure_features(manifest: Path, count: int) -> None:
    root = manifest.parent
    selected = load_examples(manifest, Split.TRAIN, count)
    for example in selected:
        if not example.audio_path.exists():
            downloaded = download_audio(example, root)
            if not downloaded.success:
                raise ValueError(f"Audio download failed: {downloaded.error}")
    if any(not example.feature_path.exists() for example in selected):
        extract_features(CacheConfig(root=root, train_examples=count, transcribe_heldout=False))
        torch.cuda.empty_cache()


def run_baselines(
    wrapper: FrozenQwen,
    validation: list[Example],
    test: list[Example],
    manifest: Path,
    output_root: Path,
    semantic_evaluator: SemanticEvaluator,
) -> None:
    transcripts = load_asr_transcripts(manifest.parent / "asr_transcripts.jsonl")
    for split, examples in (("validation", validation), ("test", test)):
        directory = output_root / "baseline" / split
        for condition in (EvaluationCondition.TEXT, EvaluationCondition.ASR):
            if (directory / f"{condition.value}.json").exists():
                continue
            outcome = evaluate(
                wrapper,
                None,
                examples,
                wrapper.config,
                condition,
                asr_transcripts=transcripts,
                semantic_evaluator=semantic_evaluator,
            )
            save_evaluation(outcome, directory, name=condition.value)


def run_suite(
    wrapper: FrozenQwen,
    training: list[Example],
    validation: list[Example],
    test: list[Example],
    manifest: Path,
    output_root: Path,
    semantic_evaluator: SemanticEvaluator,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    completed: list[str] = []
    failed: list[str] = []

    def save_state(running: str | None) -> None:
        state = SuiteState(
            completed=tuple(completed),
            running=running,
            failed=tuple(failed),
            started_at=started,
            updated_at=time.time(),
        )
        pending = output_root / "suite_state.pending.json"
        pending.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        pending.replace(output_root / "suite_state.json")

    def execute(config: RunConfig) -> RunResult:
        save_state(config.name)
        ensure_features(manifest, config.train_examples)
        for attempt in range(1, 3):
            try:
                result = run_experiment(
                    config,
                    wrapper,
                    training,
                    validation,
                    test,
                    output_root,
                    semantic_evaluator,
                )
                completed.append(config.name)
                aggregate_report(output_root, manifest.parent / "dataset_report.json")
                save_state(None)
                return result
            except Exception as exception:
                failure = ExperimentFailure(
                    run_name=config.name,
                    attempt=attempt,
                    exception_type=type(exception).__name__,
                    message=str(exception),
                    traceback=traceback.format_exc(),
                    timestamp=time.time(),
                )
                with (output_root / "failures.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(failure.model_dump_json() + "\n")
                print(failure.model_dump_json(), flush=True)
                torch.cuda.empty_cache()
                if attempt == 2:
                    failed.append(config.name)
                    save_state(None)
                    raise
        raise AssertionError("Experiment retry loop must return or raise")

    smoke_path = output_root / "smoke" / "gradient_check.json"
    if not smoke_path.exists():
        smoke(feasibility_run(), wrapper, training, output_root / "smoke")
    feasibility = execute(feasibility_run())
    if feasibility.final_fixed_training_loss >= feasibility.initial_training_loss:
        raise ValueError("V0 did not reduce training loss; inspect before scaling")
    run_baselines(wrapper, validation, test, manifest, output_root, semantic_evaluator)
    scaling = [execute(config) for config in scaling_runs()]
    selected_count = 3000
    matched = [result for result in scaling if result.train_examples == selected_count]
    compression = matched + [execute(config) for config in compression_runs(selected_count)]
    best = min(compression, key=lambda result: result.validation.cross_entropy)
    for config in architecture_runs(selected_count, best.config.projector.compression_factor):
        execute(config)
    aggregate_report(output_root, manifest.parent / "dataset_report.json")
    save_state(None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    arguments = parser.parse_args()
    config = (
        RunConfig.model_validate_json(arguments.config.read_text(encoding="utf-8"))
        if arguments.config
        else feasibility_run()
    )
    training = load_examples(arguments.manifest, Split.TRAIN)
    validation = load_examples(arguments.manifest, Split.VALIDATION, config.validation_examples)
    test = load_examples(arguments.manifest, Split.TEST, config.test_examples)
    wrapper = FrozenQwen(config, torch.device("cuda"))
    if arguments.smoke:
        smoke(config, wrapper, training, arguments.output / "smoke")
    else:
        semantic_evaluator = SemanticEvaluator()
        if arguments.suite:
            run_suite(
                wrapper,
                training,
                validation,
                test,
                arguments.manifest,
                arguments.output,
                semantic_evaluator,
            )
        elif arguments.baselines:
            run_baselines(
                wrapper, validation, test, arguments.manifest, arguments.output, semantic_evaluator
            )
        else:
            result = run_experiment(
                config, wrapper, training, validation, test, arguments.output, semantic_evaluator
            )
            print(result.model_dump_json(), flush=True)


if __name__ == "__main__":
    main()
