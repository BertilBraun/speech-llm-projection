"""Resume the teacher-target research program on one GPU."""

import argparse
import hashlib
import subprocess
import time
import traceback
from pathlib import Path

import torch
from pydantic import TypeAdapter
from safetensors.torch import load_file

from speech_projector.data import load_examples
from speech_projector.evaluation import SemanticEvaluator
from speech_projector.launcher import run_experiment, smoke
from speech_projector.llm import FrozenQwen
from speech_projector.models import (
    Example,
    ExperimentFailure,
    Record,
    RunConfig,
    RunResult,
    Split,
    SuiteState,
)
from speech_projector.projectors import Projector
from speech_projector.report import aggregate_report
from speech_projector.teacher_baselines import TeacherBaselineConfig, run_teacher_baselines
from speech_projector.teacher_configuration import (
    teacher_compression_runs,
    teacher_feasibility_run,
    teacher_linear_run,
    teacher_scaling_runs,
)
from speech_projector.teacher_evaluation import (
    evaluate_teacher_fidelity,
    save_teacher_fidelity,
)


class LaunchProvenance(Record):
    git_revision: str
    manifest_path: Path
    manifest_sha256: str
    source_archive_path: Path
    source_archive_sha256: str
    started_at: float


def snapshot_launch(manifest: Path, output: Path) -> LaunchProvenance:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "diff", "HEAD", "--name-only"], text=True).strip():
        raise ValueError("Commit tracked source changes before launching the teacher suite")
    output.mkdir(parents=True, exist_ok=True)
    source_archive = output / f"source_{revision}.zip"
    if not source_archive.exists():
        subprocess.run(["git", "archive", revision, "-o", str(source_archive)], check=True)
    record = LaunchProvenance(
        git_revision=revision,
        manifest_path=manifest,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        source_archive_path=source_archive,
        source_archive_sha256=hashlib.sha256(source_archive.read_bytes()).hexdigest(),
        started_at=time.time(),
    )
    path = output / f"launch_{revision}.json"
    if path.exists():
        previous = LaunchProvenance.model_validate_json(path.read_bytes())
        if previous.model_copy(update={"started_at": record.started_at}) != record:
            raise ValueError("Teacher suite resume provenance differs from its original launch")
        return previous
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return record


def evaluate_fidelity(
    wrapper: FrozenQwen,
    result: RunResult,
    validation: list[Example],
    test: list[Example],
    output: Path,
) -> None:
    wrapper.config = result.config
    projector = Projector(result.config.projector).to(wrapper.device)
    projector.load_state_dict(load_file(str(result.checkpoint_path)))
    for split, examples in ((Split.VALIDATION, validation), (Split.TEST, test)):
        directory = output / result.config.name / split.value
        records = evaluate_teacher_fidelity(wrapper, projector, examples, directory)
        save_teacher_fidelity(records, directory)


def run_teacher_suite(manifest: Path, output: Path, include_scaling: bool) -> None:
    provenance = snapshot_launch(manifest, output / "reproducibility")
    configurations = teacher_compression_runs()
    training = load_examples(manifest, Split.TRAIN, configurations[0].train_examples)
    validation = load_examples(manifest, Split.VALIDATION, configurations[0].validation_examples)
    test = load_examples(manifest, Split.TEST, configurations[0].test_examples)
    if (len(training), len(validation), len(test)) != (20000, 512, 512):
        raise ValueError(
            "The teacher suite requires exactly 20k training and 512/512 heldout pairs"
        )
    for example in training + validation + test:
        if not example.feature_path.is_file():
            raise ValueError(f"Missing cached Whisper feature: {example.feature_path}")
    wrapper = FrozenQwen(configurations[0], torch.device("cuda"))
    semantic_evaluator = SemanticEvaluator()
    completed: list[str] = []
    failed: list[str] = []
    all_results: list[RunResult] = []

    def save_state(running: str | None) -> None:
        state = SuiteState(
            completed=tuple(completed),
            running=running,
            failed=tuple(failed),
            started_at=provenance.started_at,
            updated_at=time.time(),
        )
        pending = output / "suite_state.pending.json"
        pending.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        pending.replace(output / "suite_state.json")

    def execute(config: RunConfig) -> RunResult:
        save_state(config.name)
        selected_validation = validation[: config.validation_examples]
        selected_test = test[: config.test_examples]
        for attempt in range(1, 3):
            try:
                result = run_experiment(
                    config,
                    wrapper,
                    training,
                    selected_validation,
                    selected_test,
                    output,
                    semantic_evaluator,
                )
                evaluate_fidelity(wrapper, result, selected_validation, selected_test, output)
                completed.append(config.name)
                all_results.append(result)
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
                with (output / "failures.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(failure.model_dump_json() + "\n")
                print(failure.model_dump_json(), flush=True)
                torch.cuda.empty_cache()
                if attempt == 2:
                    failed.append(config.name)
                    save_state(None)
                    raise
        raise AssertionError("The run must return or raise")

    feasibility = teacher_feasibility_run()
    if not (output / "smoke" / "gradient_check.json").is_file():
        wrapper.config = feasibility
        smoke(feasibility, wrapper, training, output / "smoke")
    feasible = execute(feasibility)
    if feasible.final_fixed_training_loss >= feasible.initial_training_loss:
        raise ValueError("Teacher-target feasibility failed to reduce the fixed training loss")
    results = [execute(config) for config in configurations]
    wrapper.config = configurations[0]
    run_teacher_baselines(
        wrapper,
        validation,
        test,
        TeacherBaselineConfig(
            manifest=manifest,
            teacher_directory=output / "teacher_targets",
            output_root=output,
        ),
        semantic_evaluator,
    )
    best = min(results, key=lambda result: result.validation.cross_entropy)
    execute(teacher_linear_run(best.config.projector.compression_factor))
    if include_scaling:
        for config in teacher_scaling_runs():
            execute(config)
    save_state(None)
    results_path = output / "completed_results.json"
    results_path.write_bytes(
        TypeAdapter(tuple[RunResult, ...]).dump_json(tuple(all_results), indent=2)
    )
    dataset_report = manifest.parent / "dataset_report.json"
    if dataset_report.is_file():
        aggregate_report(output, dataset_report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-scaling", action="store_true")
    arguments = parser.parse_args()
    run_teacher_suite(arguments.manifest, arguments.output, arguments.include_scaling)


if __name__ == "__main__":
    main()
