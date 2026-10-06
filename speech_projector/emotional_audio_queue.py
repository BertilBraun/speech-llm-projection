"""CPU controller that releases a completed text job into the audio environment."""

import subprocess
import time
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TypeVar

from pydantic import Field, model_validator

from scripts.inventory_results import write_record
from speech_projector.emotion_preview import PreviewCase, PreviewPlan
from speech_projector.emotional_audio import EmotionalAudioConfig, completed_audio
from speech_projector.emotional_dataset import (
    EmotionalUtterance,
    assemble_utterance,
    build_draft_requests,
)
from speech_projector.emotional_generation import EmotionalGenerationConfig, EmotionalTeacherTarget
from speech_projector.emotional_report import (
    EmotionalDatasetReport,
    EmotionalReportConfig,
    pair_records,
    write_emotional_report,
)
from speech_projector.models import Record

RecordType = TypeVar("RecordType", bound=Record)


class SupervisorState(str, Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    BACKOFF = "BACKOFF"
    STOPPING = "STOPPING"
    EXITED = "EXITED"
    FATAL = "FATAL"
    UNKNOWN = "UNKNOWN"
    FAILED = "FAILED"


class EmotionalAudioQueueConfig(Record):
    text_generation_config: Path
    supervisor_job: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_:-]+$")
    audio_python: Path
    audio_output: Path
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    batch_size: int = Field(default=16, ge=1)
    max_new_tokens: int = Field(default=512, ge=4)
    retry_max_new_tokens: int = Field(default=1024, ge=4)
    poll_seconds: float = Field(default=10, gt=0)

    @model_validator(mode="after")
    def validate_retry(self) -> "EmotionalAudioQueueConfig":
        if self.retry_max_new_tokens <= self.max_new_tokens:
            raise ValueError("Audio retry cap must exceed the initial cap")
        return self


def parse_supervisor_status(job: str, output: str, returncode: int) -> SupervisorState:
    fields = output.strip().split()
    if len(fields) < 2 or fields[0] != job:
        raise ValueError(f"Unexpected Supervisor status for {job}: {output.strip()}")
    state = SupervisorState(fields[1])
    if returncode not in (0, 1, 3):
        raise ValueError(f"Supervisor status failed with exit code {returncode}")
    return state


def supervisor_status(job: str) -> SupervisorState:
    result = subprocess.run(
        ["supervisorctl", "status", job], capture_output=True, text=True, check=False
    )
    return parse_supervisor_status(job, result.stdout, result.returncode)


def read_complete_records(path: Path, record_type: type[RecordType]) -> tuple[RecordType, ...]:
    if not path.exists():
        raise ValueError(f"Text job exited without required journal: {path}")
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        raise ValueError(f"Text job exited with an incomplete journal suffix: {path}")
    return tuple(record_type.model_validate_json(line) for line in content.splitlines())


def validated_utterances(config: EmotionalGenerationConfig) -> tuple[EmotionalUtterance, ...]:
    utterances = read_complete_records(
        config.output_directory / "utterances.jsonl", EmotionalUtterance
    )
    targets = read_complete_records(
        config.output_directory / "teacher_targets.jsonl", EmotionalTeacherTarget
    )
    expected = tuple(
        assignment
        for request in build_draft_requests(config.dataset)
        for assignment in request.assignments
    )
    if len(utterances) != len(expected) or len(targets) != 2 * len(expected):
        raise ValueError(
            f"Text job exited with incomplete coverage: {len(utterances)}/{len(expected)} "
            f"utterances, {len(targets)}/{2 * len(expected)} teacher targets"
        )
    for assignment, utterance in zip(expected, utterances, strict=True):
        if assemble_utterance(assignment, utterance.text) != utterance:
            raise ValueError(
                f"Utterance differs from its canonical assignment: {utterance.base_id}"
            )
    pairs = pair_records(utterances, targets, ())
    if any(len(pair.teacher_targets) != 2 for pair in pairs):
        raise ValueError("Text job lacks complete canonical teacher delivery pairs")
    return utterances


def build_audio_config(
    configuration: EmotionalAudioQueueConfig, utterances: tuple[EmotionalUtterance, ...]
) -> EmotionalAudioConfig:
    cases = tuple(
        PreviewCase(
            case_id=f"{utterance.base_id}_{annotation.delivery.value}",
            text=utterance.text,
            delivery=annotation.delivery,
            instruct=annotation.instruct,
            seed=42 + index,
        )
        for index, (utterance, annotation) in enumerate(
            (utterance, annotation)
            for utterance in utterances
            for annotation in utterance.deliveries
        )
    )
    return EmotionalAudioConfig(
        output=configuration.audio_output,
        source_commit=configuration.source_commit,
        batch_size=configuration.batch_size,
        retry_max_new_tokens=configuration.retry_max_new_tokens,
        plan=PreviewPlan(cases=cases, seed=42, max_new_tokens=configuration.max_new_tokens),
    )


def launch_audio(configuration: EmotionalAudioQueueConfig, audio_config_path: Path) -> None:
    subprocess.run(
        [
            str(configuration.audio_python),
            "-m",
            "scripts.generate_emotional_audio",
            "--config",
            str(audio_config_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )


def run_audio_queue(
    configuration: EmotionalAudioQueueConfig,
    *,
    status: Callable[[str], SupervisorState] = supervisor_status,
    wait: Callable[[float], None] = time.sleep,
    launch: Callable[[EmotionalAudioQueueConfig, Path], None] = launch_audio,
) -> EmotionalDatasetReport:
    generation = EmotionalGenerationConfig.model_validate_json(
        configuration.text_generation_config.read_bytes()
    )
    if generation.dataset.utterance_count != 5000:
        raise ValueError("This queue requires the authorized 5000-utterance dataset")
    configuration.audio_output.mkdir(parents=True, exist_ok=True)
    queue_path = configuration.audio_output / "queue_config.json"
    if queue_path.exists():
        if EmotionalAudioQueueConfig.model_validate_json(queue_path.read_bytes()) != configuration:
            raise ValueError("Audio queue resume configuration differs")
    else:
        write_record(queue_path, configuration)
    while True:
        state = status(configuration.supervisor_job)
        print(f"Text Supervisor job {configuration.supervisor_job}: {state.value}", flush=True)
        match state:
            case SupervisorState.EXITED:
                break
            case SupervisorState.STARTING | SupervisorState.RUNNING | SupervisorState.STOPPING:
                wait(configuration.poll_seconds)
            case _:
                raise ValueError(f"Text job did not complete successfully: {state.value}")
    utterances = validated_utterances(generation)
    audio = build_audio_config(configuration, utterances)
    audio_path = configuration.audio_output / "config.json"
    if audio_path.exists():
        if EmotionalAudioConfig.model_validate_json(audio_path.read_bytes()) != audio:
            raise ValueError("Audio resume configuration or plan differs")
    else:
        write_record(audio_path, audio)
        write_record(configuration.audio_output / "plan.json", audio.plan)
    print(f"Launching audio: {len(audio.plan.cases)} canonical delivery clips", flush=True)
    launch(configuration, audio_path)
    if len(completed_audio(audio)) != len(audio.plan.cases):
        raise ValueError("Audio subprocess exited without complete verified clip coverage")
    report = write_emotional_report(
        EmotionalReportConfig(
            dataset_directory=generation.output_directory,
            audio_directory=configuration.audio_output,
            output_directory=configuration.audio_output / "report",
        )
    )
    print(f"Audio queue complete: {report.available_audio_clips} verified clips", flush=True)
    return report
