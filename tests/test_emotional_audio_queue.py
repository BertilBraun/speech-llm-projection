import subprocess
from pathlib import Path

import pytest

from scripts.inventory_results import write_record
from speech_projector.emotional_audio_queue import (
    EmotionalAudioQueueConfig,
    SupervisorState,
    build_audio_config,
    launch_audio,
    parse_supervisor_status,
    run_audio_queue,
    validated_utterances,
)
from speech_projector.emotional_dataset import (
    EmotionalDatasetConfig,
    EmotionalUtterance,
    assemble_utterance,
    build_draft_requests,
)
from speech_projector.emotional_generation import (
    EmotionalGenerationConfig,
    EmotionalTeacherRequest,
    EmotionalTeacherTarget,
)
from speech_projector.generation import CompletedGeneration
from speech_projector.journal import append_record


def queue_config(directory: Path) -> EmotionalAudioQueueConfig:
    return EmotionalAudioQueueConfig(
        text_generation_config=directory / "text.json",
        supervisor_job="emotion-dataset-text",
        audio_python=Path("/workspace/emotion-preview/venv/bin/python"),
        audio_output=directory / "audio",
        source_commit="a" * 40,
    )


def text_config(directory: Path, count: int = 20) -> EmotionalGenerationConfig:
    return EmotionalGenerationConfig(
        output_directory=directory / "text",
        revision="b" * 40,
        source_git_commit="c" * 40,
        dataset=EmotionalDatasetConfig(utterance_count=count),
    )


def save_text(config: EmotionalGenerationConfig) -> tuple[EmotionalUtterance, ...]:
    rows = tuple(
        assemble_utterance(
            assignment, f"Please help me review the details for {assignment.base_id} today."
        )
        for request in build_draft_requests(config.dataset)
        for assignment in request.assignments
    )
    for row in rows:
        append_record(config.output_directory / "utterances.jsonl", row)
        for annotation in row.deliveries:
            append_record(
                config.output_directory / "teacher_targets.jsonl",
                EmotionalTeacherTarget(
                    request=EmotionalTeacherRequest(utterance=row, annotation=annotation),
                    response=CompletedGeneration(text="Of course, let us check.", token_ids=(1,)),
                    capped_attempts=(),
                ),
            )
    return rows


@pytest.mark.parametrize("returncode", (0, 1, 3))
def test_supervisor_exited_status_is_not_a_command_failure(returncode: int) -> None:
    assert (
        parse_supervisor_status(
            "emotion-dataset-text", "emotion-dataset-text EXITED Jun 12 03:10 PM", returncode
        )
        == SupervisorState.EXITED
    )


def test_full_pair_coverage_preserves_literal_audio_and_expressive_metadata(tmp_path: Path) -> None:
    generation = text_config(tmp_path)
    rows = save_text(generation)
    assert validated_utterances(generation) == rows
    audio = build_audio_config(queue_config(tmp_path), rows)
    assert len(audio.plan.cases) == 40
    assert audio.batch_size == 16
    assert audio.plan.max_new_tokens == 512
    assert audio.retry_max_new_tokens == 1024
    for index, case in enumerate(audio.plan.cases):
        row = rows[index // 2]
        annotation = row.deliveries[index % 2]
        assert case.text == row.text
        assert case.instruct == annotation.instruct
        assert case.case_id == f"{row.base_id}_{annotation.delivery.value}"
        assert case.seed == 42 + index


def test_equal_target_count_does_not_hide_duplicate_or_missing_delivery(tmp_path: Path) -> None:
    generation = text_config(tmp_path)
    save_text(generation)
    path = generation.output_directory / "teacher_targets.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join([lines[0], lines[0], *lines[2:]]))
    with pytest.raises(ValueError, match="duplicated"):
        validated_utterances(generation)


def test_exited_incomplete_text_fails_instead_of_launching_audio(tmp_path: Path) -> None:
    configuration = queue_config(tmp_path)
    generation = text_config(tmp_path, 5000)
    write_record(configuration.text_generation_config, generation)
    generation.output_directory.mkdir()
    (generation.output_directory / "utterances.jsonl").write_bytes(b"")
    (generation.output_directory / "teacher_targets.jsonl").write_bytes(b"")
    states = iter((SupervisorState.RUNNING, SupervisorState.EXITED))
    sleeps: list[float] = []
    launched: list[Path] = []

    def status(job: str) -> SupervisorState:
        assert job == configuration.supervisor_job
        return next(states)

    def launch(config: EmotionalAudioQueueConfig, path: Path) -> None:
        launched.append(path)

    with pytest.raises(ValueError, match="0/5000.*0/10000"):
        run_audio_queue(configuration, status=status, wait=sleeps.append, launch=launch)
    assert sleeps == [10]
    assert launched == []


@pytest.mark.parametrize("state", (SupervisorState.FATAL, SupervisorState.FAILED))
def test_failed_supervisor_job_does_not_poll_forever(
    tmp_path: Path, state: SupervisorState
) -> None:
    configuration = queue_config(tmp_path)
    write_record(configuration.text_generation_config, text_config(tmp_path, 5000))

    def status(job: str) -> SupervisorState:
        return state

    with pytest.raises(ValueError, match="did not complete successfully"):
        run_audio_queue(configuration, status=status)


def test_incomplete_journal_suffix_is_not_silently_repaired(tmp_path: Path) -> None:
    generation = text_config(tmp_path)
    save_text(generation)
    path = generation.output_directory / "teacher_targets.jsonl"
    content = path.read_bytes() + b'{"request":'
    path.write_bytes(content)
    with pytest.raises(ValueError, match="incomplete journal suffix"):
        validated_utterances(generation)
    assert path.read_bytes() == content


def test_audio_launch_uses_isolated_python_foreground_and_propagates_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = queue_config(tmp_path)
    path = tmp_path / "config.json"

    def failed_run(
        arguments: list[str], *, cwd: Path, check: bool
    ) -> subprocess.CompletedProcess[str]:
        assert arguments == [
            str(configuration.audio_python),
            "-m",
            "scripts.generate_emotional_audio",
            "--config",
            str(path),
        ]
        assert cwd == Path(__file__).resolve().parents[1]
        assert check
        raise subprocess.CalledProcessError(1, arguments)

    monkeypatch.setattr(subprocess, "run", failed_run)
    with pytest.raises(subprocess.CalledProcessError):
        launch_audio(configuration, path)
