"""Supervised sequential execution of the teacher-target rerun."""

import argparse
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pydantic import TypeAdapter

from speech_projector.evaluation import ConditioningDiagnostic
from speech_projector.generation import INITIAL_TEACHER_TOKEN_CAP, RETRY_TEACHER_TOKEN_CAP
from speech_projector.models import EvaluationCondition, GradientCheck, RunResult, SuiteState


class Phase(str, Enum):
    BOOTSTRAP = "teacher_bootstrap"
    SMOKE = "gradient_smoke"
    FEASIBILITY = "teacher_feasibility"
    CONDITIONING = "feasibility_audio_conditioning"
    CACHE = "missing_features_and_asr"
    TEACHER = "full_teacher_targets"
    AUDIT = "teacher_split_audit"
    SUITE = "teacher_research_suite"


@dataclass(frozen=True)
class Job:
    phase: Phase
    arguments: tuple[str, ...]
    outputs: tuple[Path, ...]


def jobs(root: Path, teacher_batch_size: int, include_scaling: bool) -> tuple[Job, ...]:
    data = root / "data_teacher"
    results = root / "results_teacher"
    bootstrap = data / "teacher_bootstrap.jsonl"
    full = data / "teacher_examples.jsonl"
    feasibility = results / "teacher_v0_256_mlp_10hz"
    generation = (
        "-m",
        "scripts.generate_teacher_targets",
        "--manifest",
        str(data / "examples_source.jsonl"),
        "--output",
        str(results / "teacher_targets"),
        "--run-config",
        str(root / "configs/teacher_10hz.json"),
        "--batch-size",
        str(teacher_batch_size),
        "--initial-max-new-tokens",
        str(INITIAL_TEACHER_TOKEN_CAP),
        "--retry-max-new-tokens",
        str(RETRY_TEACHER_TOKEN_CAP),
    )
    launch_feasibility = (
        "-m",
        "speech_projector.launcher",
        "--manifest",
        str(bootstrap),
        "--output",
        str(results),
        "--config",
        str(root / "configs/teacher_v0.json"),
    )
    suite = (
        "-m",
        "speech_projector.teacher_launcher",
        "--manifest",
        str(full),
        "--output",
        str(results),
    ) + (("--include-scaling",) if include_scaling else ())
    return (
        Job(
            Phase.BOOTSTRAP,
            generation + ("--bootstrap", "--output-manifest", str(bootstrap)),
            (bootstrap,),
        ),
        Job(
            Phase.SMOKE, launch_feasibility + ("--smoke",), (results / "smoke/gradient_check.json",)
        ),
        Job(Phase.FEASIBILITY, launch_feasibility, (feasibility / "result.json",)),
        Job(
            Phase.CONDITIONING,
            (
                "-m",
                "scripts.evaluate_training_probe",
                "--run-dir",
                str(feasibility),
                "--manifest",
                str(bootstrap),
            ),
            (feasibility / "overfit_probe/speech_conditioning.json",),
        ),
        Job(
            Phase.CACHE,
            (
                "-m",
                "speech_projector.cache",
                "--root",
                str(data),
                "--train-examples",
                "20000",
                "--batch-size",
                "8",
            ),
            (data / "cache_stats_20000.json",),
        ),
        Job(Phase.TEACHER, generation + ("--output-manifest", str(full)), (full,)),
        Job(
            Phase.AUDIT,
            (
                "-m",
                "scripts.audit_teacher_manifest",
                "--manifest",
                str(full),
                "--output",
                str(results / "dataset"),
            ),
            (results / "dataset/dataset_leakage.json",),
        ),
        Job(Phase.SUITE, suite, (results / "completed_results.json",)),
    )


def validate_feasibility(root: Path) -> None:
    results = root / "results_teacher"
    gradient = GradientCheck.model_validate_json(
        (results / "smoke/gradient_check.json").read_bytes()
    )
    if not (
        math.isfinite(gradient.loss)
        and math.isfinite(gradient.projector_gradient_norm)
        and gradient.projector_changed
        and gradient.projector_gradient_norm > 0
        and gradient.llm_weights_unchanged
        and not gradient.llm_has_gradients
    ):
        raise ValueError("Teacher-target gradient/frozen-weight verification failed")
    directory = results / "teacher_v0_256_mlp_10hz"
    result = RunResult.model_validate_json((directory / "result.json").read_bytes())
    if (
        not math.isfinite(result.final_fixed_training_loss)
        or not math.isfinite(result.initial_training_loss)
        or result.final_fixed_training_loss >= result.initial_training_loss
    ):
        raise ValueError("Teacher-target training did not reduce the fixed training loss")
    controls = TypeAdapter(tuple[ConditioningDiagnostic, ...]).validate_json(
        (directory / "overfit_probe/speech_conditioning.json").read_bytes()
    )
    shuffled_controls = tuple(
        item
        for item in controls
        if item.correct_condition == EvaluationCondition.SPEECH
        and item.control_condition == EvaluationCondition.SHUFFLED_SPEECH
    )
    if len(shuffled_controls) != 1:
        raise ValueError("Feasibility requires exactly one paired shuffled-audio diagnostic")
    shuffled = shuffled_controls[0]
    if (
        shuffled.paired_examples <= 0
        or not math.isfinite(shuffled.mean_control_minus_correct_ce)
        or shuffled.mean_control_minus_correct_ce <= 0
    ):
        raise ValueError(
            "Teacher-target feasibility does not show positive training audio conditioning"
        )


def run_pipeline(root: Path, teacher_batch_size: int, include_scaling: bool) -> None:
    output = root / "results_teacher"
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "pipeline_state.json"
    previous = (
        SuiteState.model_validate_json(state_path.read_bytes()) if state_path.exists() else None
    )
    completed = list(previous.completed) if previous else []
    started = previous.started_at if previous else time.time()

    def save_state(running: str | None, failed: tuple[str, ...] = ()) -> None:
        state = SuiteState(
            completed=tuple(completed),
            running=running,
            failed=failed,
            started_at=started,
            updated_at=time.time(),
        )
        pending = output / "pipeline_state.pending.json"
        pending.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        pending.replace(state_path)

    for job in jobs(root, teacher_batch_size, include_scaling):
        if job.phase.value in completed:
            if not all(path.is_file() and path.stat().st_size > 0 for path in job.outputs):
                raise ValueError(f"Completed phase lost its artifacts: {job.phase.value}")
            continue
        save_state(job.phase.value)
        print(f"Starting phase {job.phase.value}", flush=True)
        try:
            if job.phase == Phase.CACHE:
                validate_feasibility(root)
            subprocess.run([sys.executable, "-B", *job.arguments], cwd=root, check=True)
            if not all(path.is_file() and path.stat().st_size > 0 for path in job.outputs):
                raise ValueError(f"Phase did not produce its expected artifacts: {job.phase.value}")
        except Exception:
            save_state(None, (job.phase.value,))
            raise
        completed.append(job.phase.value)
        save_state(None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--teacher-batch-size", type=int, required=True)
    parser.add_argument("--skip-scaling", action="store_true")
    arguments = parser.parse_args()
    if arguments.teacher_batch_size < 1:
        parser.error("--teacher-batch-size must be positive")
    run_pipeline(arguments.root, arguments.teacher_batch_size, not arguments.skip_scaling)


if __name__ == "__main__":
    main()
