"""Exact teacher prompting, paired target identity and durable backend reuse."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from speech_projector.emotional_dataset import EmotionalDatasetConfig
from speech_projector.emotional_generation import (
    CachedGeneration,
    EmotionalGenerationConfig,
    TextGenerationBatch,
    TextGenerationOutcome,
    TextGenerationRequest,
)
from speech_projector.generation import CompletedGeneration
from speech_projector.journal import append_record
from speech_projector.neu_dataset import (
    NeuUtterance,
    build_neu_requests,
    neu_cases,
    write_neu_cases,
)
from speech_projector.neu_generation import (
    NeuGenerationConfig,
    generate_neu_targets,
    neu_teacher_request,
    render_neu_pilot,
    summarize_neu_generation,
)


def configuration(directory: Path) -> NeuGenerationConfig:
    return NeuGenerationConfig(
        generation=EmotionalGenerationConfig(
            output_directory=directory,
            revision="a" * 40,
            source_git_commit="b" * 40,
            dataset=EmotionalDatasetConfig(utterance_count=2),
            inference_batch_size=2,
        ),
        previous_utterances=directory.parent / "old.jsonl",
        previous_sha256="c" * 64,
    )


def source_rows(directory: Path) -> tuple[NeuUtterance, ...]:
    request = build_neu_requests(EmotionalDatasetConfig(utterance_count=2))[0]
    rows = tuple(
        NeuUtterance(
            assignment=assignment,
            text=f"Please help me check package {index} before the appointment tomorrow.",
        )
        for index, assignment in enumerate(request.assignments)
    )
    for row in rows:
        append_record(directory / "utterances.jsonl", row)
    write_neu_cases(directory, rows, 42)
    return rows


def test_teacher_metadata_remains_system_and_literal_user_is_shared(tmp_path: Path) -> None:
    config = configuration(tmp_path / "new")
    cases = neu_cases(source_rows(config.generation.output_directory), 42)
    first, second = (neu_teacher_request(case, config) for case in cases[:2])
    assert first.messages[1] == second.messages[1]
    assert first.messages[1].content == cases[0].text
    assert "happy tone" in first.messages[0].content
    assert "sad tone" in second.messages[0].content
    assert "not an instruction to imitate" in first.messages[0].content
    assert first.messages[0].content.startswith(config.generation.teacher_system)


def test_completed_targets_resume_from_exact_prompts_and_source_pin(tmp_path: Path) -> None:
    config = configuration(tmp_path / "new")
    source_rows(config.generation.output_directory)
    calls: list[tuple[str, ...]] = []

    def generate(requests: Sequence[TextGenerationRequest], cap: int) -> TextGenerationBatch:
        calls.append(tuple(request.request_id for request in requests))
        return TextGenerationBatch(
            token_cap=cap,
            seed=config.generation.seed,
            runtime_seconds=1,
            outcomes=tuple(
                TextGenerationOutcome(
                    request=request,
                    prompt_text=request.messages[0].content,
                    prompt_token_ids=(1, 2),
                    generation=CompletedGeneration(
                        text="I can help you check that.", token_ids=(3, 4)
                    ),
                )
                for request in requests
            ),
        )

    cached = CachedGeneration(config.generation, generate)
    pilot = generate_neu_targets(config, cached, limit=2)
    assert len(pilot) == 2 and len(calls) == 1
    targets = generate_neu_targets(config, cached)
    assert len(targets) == 4 and len(calls) == 2
    assert generate_neu_targets(config, cached) == targets and len(calls) == 2
    summary = summarize_neu_generation(config)
    assert summary.identical_target_pairs == 2 and summary.generation_seconds == 2
    assert "Target pending" not in render_neu_pilot(config.generation.output_directory).read_text()
    source_path = config.generation.output_directory / "utterances.jsonl"
    source_path.write_text(source_path.read_text().replace("package 0", "parcel 0"))
    with pytest.raises(ValueError, match="source manifest changed"):
        generate_neu_targets(config, cached)


def test_generation_resume_rejects_changed_provider_prompt(tmp_path: Path) -> None:
    config = configuration(tmp_path / "new")

    def generate(requests: Sequence[TextGenerationRequest], cap: int) -> TextGenerationBatch:
        raise AssertionError("No inference required for provenance validation")

    CachedGeneration(config.generation, generate)
    changed = EmotionalGenerationConfig(
        output_directory=config.generation.output_directory,
        revision="a" * 40,
        source_git_commit="b" * 40,
        dataset=config.generation.dataset,
        teacher_system="Changed teacher policy",
    )
    with pytest.raises(ValueError, match="provenance differs"):
        CachedGeneration(changed, generate)
