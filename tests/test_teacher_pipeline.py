import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import TypeAdapter

from scripts import run_teacher_pipeline as pipeline
from speech_projector.evaluation import ConditioningDiagnostic
from speech_projector.models import (
    EvaluationCondition,
    EvaluationMetrics,
    GradientCheck,
    RunResult,
    SuiteState,
)
from speech_projector.teacher_configuration import teacher_feasibility_run
from speech_projector.teacher_launcher import snapshot_launch


@dataclass(frozen=True)
class FeasibilityArtifacts:
    root: Path
    gradient: GradientCheck
    result: RunResult
    diagnostic: ConditioningDiagnostic

    @property
    def gradient_path(self) -> Path:
        return self.root / "results_teacher/smoke/gradient_check.json"

    @property
    def result_path(self) -> Path:
        return self.root / "results_teacher/teacher_v0_256_mlp_10hz/result.json"

    @property
    def controls_path(self) -> Path:
        return self.result_path.parent / "overfit_probe/speech_conditioning.json"


def write_controls(path: Path, controls: Sequence[ConditioningDiagnostic]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(TypeAdapter(tuple[ConditioningDiagnostic, ...]).dump_json(tuple(controls)))


@pytest.fixture
def feasibility_artifacts(tmp_path: Path) -> FeasibilityArtifacts:
    config = teacher_feasibility_run()
    metrics = EvaluationMetrics(examples=32, target_tokens=100, cross_entropy=2, perplexity=7.389)
    artifacts = FeasibilityArtifacts(
        root=tmp_path,
        gradient=GradientCheck(
            loss=3,
            projector_gradient_norm=1,
            projector_changed=True,
            frozen_parameters=2000000000,
            llm_has_gradients=False,
            llm_weights_unchanged=True,
            peak_vram_gb=4,
            step_seconds=1,
        ),
        result=RunResult(
            config=config,
            git_commit="a" * 40,
            train_examples=256,
            validation_examples=32,
            test_examples=32,
            projector_parameters=2000000,
            pseudo_tokens_per_second=10,
            mean_pseudo_tokens=70,
            steps=160,
            runtime_seconds=200,
            peak_vram_gb=4,
            examples_per_second=6.4,
            target_tokens_per_second=192,
            initial_validation_loss=3,
            initial_training_loss=3,
            final_fixed_training_loss=1,
            final_training_loss=1.1,
            validation=metrics,
            test=metrics,
            checkpoint_path=tmp_path / "projector.safetensors",
        ),
        diagnostic=ConditioningDiagnostic(
            correct_condition=EvaluationCondition.SPEECH,
            control_condition=EvaluationCondition.SHUFFLED_SPEECH,
            paired_examples=32,
            mean_control_minus_correct_ce=0.5,
            standard_error=0.05,
            fraction_correct_audio_lower_loss=0.9,
        ),
    )
    artifacts.gradient_path.parent.mkdir(parents=True)
    artifacts.gradient_path.write_text(artifacts.gradient.model_dump_json(), encoding="utf-8")
    artifacts.result_path.parent.mkdir(parents=True)
    artifacts.result_path.write_text(artifacts.result.model_dump_json(), encoding="utf-8")
    write_controls(artifacts.controls_path, (artifacts.diagnostic,))
    return artifacts


def test_feasibility_requires_learning_and_positive_paired_audio_conditioning(
    feasibility_artifacts: FeasibilityArtifacts,
) -> None:
    pipeline.validate_feasibility(feasibility_artifacts.root)


class GradientFailure(str, Enum):
    ZERO_GRADIENT = "zero_gradient"
    UNCHANGED_PROJECTOR = "unchanged_projector"
    LLM_GRADIENT = "llm_gradient"
    LLM_WEIGHT_CHANGE = "llm_weight_change"


@pytest.mark.parametrize("failure", tuple(GradientFailure))
def test_each_gradient_and_frozen_weight_gate_blocks_scaling(
    feasibility_artifacts: FeasibilityArtifacts, failure: GradientFailure
) -> None:
    original = feasibility_artifacts.gradient
    match failure:
        case GradientFailure.ZERO_GRADIENT:
            gradient = original.model_copy(update={"projector_gradient_norm": 0})
        case GradientFailure.UNCHANGED_PROJECTOR:
            gradient = original.model_copy(update={"projector_changed": False})
        case GradientFailure.LLM_GRADIENT:
            gradient = original.model_copy(update={"llm_has_gradients": True})
        case GradientFailure.LLM_WEIGHT_CHANGE:
            gradient = original.model_copy(update={"llm_weights_unchanged": False})
    feasibility_artifacts.gradient_path.write_text(gradient.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="gradient|frozen"):
        pipeline.validate_feasibility(feasibility_artifacts.root)


@pytest.mark.parametrize("final_fixed_loss", [3.0, 3.1])
def test_loss_gate_uses_fixed_subset_not_last_minibatch(
    feasibility_artifacts: FeasibilityArtifacts, final_fixed_loss: float
) -> None:
    result = feasibility_artifacts.result.model_copy(
        update={"final_fixed_training_loss": final_fixed_loss, "final_training_loss": 0.01}
    )
    feasibility_artifacts.result_path.write_text(result.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="loss"):
        pipeline.validate_feasibility(feasibility_artifacts.root)


@pytest.mark.parametrize("margin", [0.0, -0.01])
def test_loss_decrease_without_audio_conditioning_blocks_scaling(
    feasibility_artifacts: FeasibilityArtifacts, margin: float
) -> None:
    diagnostic = feasibility_artifacts.diagnostic.model_copy(
        update={"mean_control_minus_correct_ce": margin}
    )
    write_controls(feasibility_artifacts.controls_path, (diagnostic,))
    with pytest.raises(ValueError, match="conditioning"):
        pipeline.validate_feasibility(feasibility_artifacts.root)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "empty_pairs"])
def test_audio_gate_requires_one_nonempty_matching_diagnostic(
    feasibility_artifacts: FeasibilityArtifacts, failure: str
) -> None:
    diagnostic = feasibility_artifacts.diagnostic
    match failure:
        case "missing":
            controls = (
                diagnostic.model_copy(
                    update={"control_condition": EvaluationCondition.ZERO_SPEECH}
                ),
            )
        case "duplicate":
            controls = (diagnostic, diagnostic)
        case "empty_pairs":
            controls = (diagnostic.model_copy(update={"paired_examples": 0}),)
    write_controls(feasibility_artifacts.controls_path, controls)
    with pytest.raises(ValueError):
        pipeline.validate_feasibility(feasibility_artifacts.root)


@pytest.mark.parametrize(
    "field", ["initial_training_loss", "final_fixed_training_loss", "audio_margin"]
)
def test_nonfinite_json_numbers_cannot_satisfy_feasibility(
    feasibility_artifacts: FeasibilityArtifacts, field: str
) -> None:
    if field == "audio_margin":
        path = feasibility_artifacts.controls_path
        original = '"mean_control_minus_correct_ce":0.5'
        corrupted = '"mean_control_minus_correct_ce":NaN'
    else:
        path = feasibility_artifacts.result_path
        original = f'"{field}":{3.0 if field == "initial_training_loss" else 1.0}'
        corrupted = f'"{field}":NaN'
    content = path.read_text(encoding="utf-8")
    assert original in content
    path.write_text(content.replace(original, corrupted), encoding="utf-8")
    with pytest.raises(ValueError):
        pipeline.validate_feasibility(feasibility_artifacts.root)


def write_state(root: Path, completed: tuple[str, ...]) -> SuiteState:
    state = SuiteState(completed=completed, running=None, failed=(), started_at=100, updated_at=101)
    path = root / "results_teacher/pipeline_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(), encoding="utf-8")
    return state


def test_completed_phase_resume_skips_process_and_preserves_artifacts(
    feasibility_artifacts: FeasibilityArtifacts,
) -> None:
    root = feasibility_artifacts.root
    scheduled = (
        pipeline.Job(
            pipeline.Phase.SMOKE, ("must-not-run",), (feasibility_artifacts.gradient_path,)
        ),
        pipeline.Job(
            pipeline.Phase.FEASIBILITY, ("must-not-run",), (feasibility_artifacts.result_path,)
        ),
    )
    write_state(root, tuple(job.phase.value for job in scheduled))
    before = tuple(path.read_bytes() for job in scheduled for path in job.outputs)
    with (
        patch.object(pipeline, "jobs", return_value=scheduled),
        patch.object(pipeline.subprocess, "run", side_effect=AssertionError("Unexpected rerun")),
    ):
        pipeline.run_pipeline(root, 8, False)
    assert tuple(path.read_bytes() for job in scheduled for path in job.outputs) == before


@pytest.mark.parametrize("failure", ["missing", "empty"])
def test_completed_phase_missing_output_is_not_silently_rerun(
    feasibility_artifacts: FeasibilityArtifacts, failure: str
) -> None:
    path = feasibility_artifacts.gradient_path
    if failure == "missing":
        path.unlink()
    else:
        path.write_bytes(b"")
    scheduled = (pipeline.Job(pipeline.Phase.SMOKE, ("must-not-run",), (path,)),)
    write_state(feasibility_artifacts.root, (pipeline.Phase.SMOKE.value,))
    with (
        patch.object(pipeline, "jobs", return_value=scheduled),
        patch.object(pipeline.subprocess, "run", side_effect=AssertionError("Unexpected rerun")),
        pytest.raises(ValueError, match="lost its artifacts"),
    ):
        pipeline.run_pipeline(feasibility_artifacts.root, 8, False)


@pytest.mark.parametrize("phase", (pipeline.Phase.CACHE, pipeline.Phase.TEACHER))
def test_failed_gate_blocks_gpu_process_and_records_failed_phase(
    feasibility_artifacts: FeasibilityArtifacts,
    phase: pipeline.Phase,
) -> None:
    root = feasibility_artifacts.root
    invalid = feasibility_artifacts.gradient.model_copy(update={"llm_weights_unchanged": False})
    feasibility_artifacts.gradient_path.write_text(invalid.model_dump_json(), encoding="utf-8")
    job = pipeline.Job(phase, ("must-not-run",), (root / "cache.json",))
    with (
        patch.object(pipeline, "jobs", return_value=(job,)),
        patch.object(
            pipeline.subprocess, "run", side_effect=AssertionError("Unexpected GPU process")
        ),
        pytest.raises(ValueError),
    ):
        pipeline.run_pipeline(root, 8, False)
    state = SuiteState.model_validate_json(
        (root / "results_teacher/pipeline_state.json").read_bytes()
    )
    assert state.failed == (phase.value,)
    assert state.completed == ()
    assert state.running is None


def test_successful_resumed_phase_records_completion_and_preserves_start(tmp_path: Path) -> None:
    output = tmp_path / "produced.txt"
    job = pipeline.Job(pipeline.Phase.BOOTSTRAP, ("test-process",), (output,))
    previous = write_state(tmp_path, ())
    calls: list[tuple[str, ...]] = []

    def execute(
        arguments: Sequence[str], *, cwd: Path, check: bool
    ) -> subprocess.CompletedProcess[bytes]:
        assert cwd == tmp_path and check
        calls.append(tuple(arguments))
        output.write_bytes(b"test-local generated artifact")
        return subprocess.CompletedProcess(list(arguments), 0)

    with (
        patch.object(pipeline, "jobs", return_value=(job,)),
        patch.object(pipeline.subprocess, "run", side_effect=execute),
    ):
        pipeline.run_pipeline(tmp_path, 8, False)
    state = SuiteState.model_validate_json(
        (tmp_path / "results_teacher/pipeline_state.json").read_bytes()
    )
    assert state.completed == (pipeline.Phase.BOOTSTRAP.value,)
    assert state.started_at == previous.started_at
    assert state.running is None and state.failed == ()
    assert len(calls) == 1


def test_zero_exit_without_expected_output_is_failed_phase(tmp_path: Path) -> None:
    job = pipeline.Job(pipeline.Phase.BOOTSTRAP, ("test-process",), (tmp_path / "absent.json",))
    with (
        patch.object(pipeline, "jobs", return_value=(job,)),
        patch.object(pipeline.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)),
        pytest.raises(ValueError, match="expected artifacts"),
    ):
        pipeline.run_pipeline(tmp_path, 8, False)
    state = SuiteState.model_validate_json(
        (tmp_path / "results_teacher/pipeline_state.json").read_bytes()
    )
    assert state.failed == (pipeline.Phase.BOOTSTRAP.value,)
    assert state.completed == ()


@dataclass(frozen=True)
class SnapshotRepository:
    root: Path
    manifest: Path
    output: Path


@pytest.fixture
def snapshot_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SnapshotRepository:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "source.py").write_text("value = 1\n", encoding="utf-8")
    (repository / ".gitignore").write_text("data/\nresults/\n", encoding="utf-8")
    commands = (
        ("init",),
        ("add", "source.py", ".gitignore"),
        (
            "-c",
            "user.name=Research Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "Committed test source",
        ),
    )
    for arguments in commands:
        subprocess.run(["git", *arguments], cwd=repository, check=True, capture_output=True)
    manifest = repository / "data/teacher.jsonl"
    manifest.parent.mkdir()
    manifest.write_bytes(b"test-local immutable dataset identity")
    monkeypatch.chdir(repository)
    return SnapshotRepository(repository, manifest, repository / "results/reproducibility")


def test_launch_resume_reuses_original_provenance_and_start_time(
    snapshot_repository: SnapshotRepository,
) -> None:
    first = snapshot_launch(snapshot_repository.manifest, snapshot_repository.output)
    launch_path = snapshot_repository.output / f"launch_{first.git_revision}.json"
    before = launch_path.read_bytes()
    resumed = snapshot_launch(snapshot_repository.manifest, snapshot_repository.output)
    assert resumed == first
    assert resumed.started_at == first.started_at
    assert launch_path.read_bytes() == before
    assert first.source_archive_path.is_file()


@pytest.mark.parametrize("changed_artifact", ["manifest", "archive"])
def test_resume_rejects_changed_dataset_or_source_archive(
    snapshot_repository: SnapshotRepository, changed_artifact: str
) -> None:
    first = snapshot_launch(snapshot_repository.manifest, snapshot_repository.output)
    path = (
        snapshot_repository.manifest
        if changed_artifact == "manifest"
        else first.source_archive_path
    )
    with path.open("ab") as stream:
        stream.write(b"changed evidence")
    with pytest.raises(ValueError, match="provenance"):
        snapshot_launch(snapshot_repository.manifest, snapshot_repository.output)


def test_launch_rejects_uncommitted_tracked_training_source(
    snapshot_repository: SnapshotRepository,
) -> None:
    (snapshot_repository.root / "source.py").write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Commit tracked source"):
        snapshot_launch(snapshot_repository.manifest, snapshot_repository.output)
