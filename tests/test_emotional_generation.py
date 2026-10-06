from collections.abc import Sequence
from pathlib import Path

import pytest

from speech_projector.emotional_dataset import (
    EmotionalDatasetConfig,
    EmotionalUtterance,
    GeneratedDraftBatch,
    GeneratedDraftText,
    build_draft_requests,
)
from speech_projector.emotional_generation import (
    TEACHER_SYSTEM,
    CachedGeneration,
    DraftSyntaxFailure,
    EmotionalGenerationConfig,
    EmotionalTeacherRequest,
    GenerationBackend,
    TextGenerationBatch,
    TextGenerationOutcome,
    TextGenerationRequest,
    VllmRuntimeConfig,
    completed_outcomes,
    generate_drafts,
    generate_teacher_targets,
    generation_from_finish_reason,
    request_decoding,
    request_seed,
    summarize_generation,
    teacher_generation_request,
)
from speech_projector.generation import CompletedGeneration, TokenLimitedGeneration
from speech_projector.journal import read_journal
from speech_projector.models import GreedyDecodingConfig, SamplingDecodingConfig
from speech_projector.preview_responses import ChatMessage


def configuration(directory: Path) -> EmotionalGenerationConfig:
    return EmotionalGenerationConfig(
        output_directory=directory,
        revision="a" * 40,
        source_git_commit="b" * 40,
        dataset=EmotionalDatasetConfig(utterance_count=20, batch_size=5),
        inference_batch_size=2,
    )


def test_draft_sampling_and_repair_seeds_leave_teacher_greedy(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    messages = (ChatMessage(role="user", content="Generate text."),)
    draft = TextGenerationRequest(request_id="draft:batch_0", messages=messages)
    repair = TextGenerationRequest(request_id="draft:batch_0_repair_1", messages=messages)
    teacher = TextGenerationRequest(request_id="teacher:base:happy", messages=messages)
    assert request_decoding(draft, config) == SamplingDecodingConfig(
        temperature=0.8, top_p=0.95, top_k=50, presence_penalty=0, repetition_penalty=1
    )
    assert request_decoding(teacher, config) == GreedyDecodingConfig()
    assert request_seed(draft, config) == request_seed(draft, config)
    assert request_seed(draft, config) != request_seed(repair, config)


class FixtureBackend:
    def __init__(self, config: EmotionalGenerationConfig) -> None:
        self.config = config
        self.calls: list[tuple[str, ...]] = []
        self.persisted_counts: list[int] = []
        self.drafts = {
            f"draft:{request.batch_id}": GeneratedDraftBatch(
                utterances=tuple(
                    GeneratedDraftText(
                        base_id=assignment.base_id,
                        text=f"Please help me check the detail for {assignment.base_id} today.",
                    )
                    for assignment in request.assignments
                )
            )
            for request in build_draft_requests(config.dataset)
        }

    def generate(self, requests: Sequence[TextGenerationRequest], cap: int) -> TextGenerationBatch:
        self.calls.append(tuple(item.request_id for item in requests))
        self.persisted_counts.append(
            len(read_journal(self.config.output_directory / "utterances.jsonl", EmotionalUtterance))
        )
        return TextGenerationBatch(
            token_cap=cap,
            seed=self.config.seed,
            runtime_seconds=1,
            outcomes=tuple(
                TextGenerationOutcome(
                    request=request,
                    prompt_text="\n".join(item.content for item in request.messages),
                    prompt_token_ids=(1, 2),
                    generation=CompletedGeneration(
                        text=self.drafts[request.request_id].model_dump_json()
                        if request.request_id.startswith("draft:")
                        else "Of course, what would help?",
                        token_ids=(3, 4),
                    ),
                )
                for request in requests
            ),
        )


def test_batched_drafts_persist_before_next_batch_and_teacher_pairs_resume(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    backend = FixtureBackend(config)
    cached = CachedGeneration(config, backend.generate)
    rows = generate_drafts(config, cached)
    assert len(rows) == 20
    assert tuple(len(batch) for batch in backend.calls) == (2, 2)
    assert backend.persisted_counts == [0, 10]
    targets = generate_teacher_targets(config, cached)
    assert len(targets) == 40
    for row in rows:
        assert (
            tuple(item.request.annotation for item in targets if item.request.utterance == row)
            == row.deliveries
        )
    before = len(backend.calls)
    assert generate_drafts(config, cached) == rows
    assert generate_teacher_targets(config, cached) == targets
    assert len(backend.calls) == before
    summary = summarize_generation(config)
    assert summary.completed_targets == 40
    assert summary.utterances == 20
    assert summary.generation_batches == before
    assert summary.generation_seconds == before
    assert config.decoding == GreedyDecodingConfig()


def test_teacher_uses_exact_approved_prompt_literal_text_and_no_synthesis_instructions(
    tmp_path: Path,
) -> None:
    config = configuration(tmp_path)
    rows = generate_drafts(config, CachedGeneration(config, FixtureBackend(config).generate))
    for row in rows:
        for annotation in row.deliveries:
            request = teacher_generation_request(
                EmotionalTeacherRequest(utterance=row, annotation=annotation), config
            )
            assert tuple(message.role for message in request.messages) == ("system", "user")
            assert request.messages[1].content == row.text
            assert request.messages[0].content == (
                "Reply naturally in one or two short sentences. "
                "The supplied tone describes the user's emotional delivery. "
                "Respond appropriately to that delivery and what they said.\n\n"
                f"The USER delivered this utterance with a {annotation.delivery.value} tone.\n"
                "This is metadata about the user, not an instruction to imitate their tone."
            )
            assert annotation.instruct not in request.messages[0].content
    assert config.teacher_system == TEACHER_SYSTEM


def test_cache_rejects_changed_provenance_and_changed_exact_prompt(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    backend = FixtureBackend(config)
    cached = CachedGeneration(config, backend.generate)
    request = TextGenerationRequest(
        request_id="teacher:test", messages=(ChatMessage(role="user", content="Hello"),)
    )
    cached.generate((request,), 256)
    changed = request.model_copy(
        update={"messages": (ChatMessage(role="user", content="Changed"),)}
    )
    with pytest.raises(ValueError, match="Cached prompt differs"):
        cached.generate((changed,), 256)
    with pytest.raises(ValueError, match="provenance differs"):
        CachedGeneration(config.model_copy(update={"revision": "c" * 40}), backend.generate)


@pytest.mark.parametrize("retry_completes", (True, False))
def test_capped_generations_preserve_both_attempts_and_never_become_false_eos(
    tmp_path: Path, retry_completes: bool
) -> None:
    config = configuration(tmp_path)
    request = TextGenerationRequest(
        request_id="teacher:cap", messages=(ChatMessage(role="user", content="Hello"),)
    )

    def generate(requests: Sequence[TextGenerationRequest], cap: int) -> TextGenerationBatch:
        return TextGenerationBatch(
            token_cap=cap,
            seed=42,
            runtime_seconds=1,
            outcomes=tuple(
                TextGenerationOutcome(
                    request=item,
                    prompt_text="Hello",
                    prompt_token_ids=(1,),
                    generation=CompletedGeneration(text="Complete", token_ids=(2, 3))
                    if cap == 512 and retry_completes
                    else TokenLimitedGeneration(partial_text="Partial", token_ids=(2,)),
                )
                for item in requests
            ),
        )

    cached = CachedGeneration(config, generate)
    if retry_completes:
        outputs = completed_outcomes((request,), cached, 256, 512)
        assert outputs[0][0].text == "Complete"
        assert len(outputs[0][1]) == 1
    else:
        with pytest.raises(ValueError, match="remains token-limited"):
            completed_outcomes((request,), cached, 256, 512)
    assert (
        len(read_journal(config.output_directory / "generation_batches.jsonl", TextGenerationBatch))
        == 2
    )


def test_backend_wrong_order_is_rejected_before_journal_append(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    backend = FixtureBackend(config)

    def reversed_outputs(
        requests: Sequence[TextGenerationRequest], cap: int
    ) -> TextGenerationBatch:
        batch = backend.generate(requests, cap)
        return batch.model_copy(update={"outcomes": tuple(reversed(batch.outcomes))})

    requests = tuple(
        TextGenerationRequest(
            request_id=f"teacher:{index}", messages=(ChatMessage(role="user", content="Hello"),)
        )
        for index in range(2)
    )
    with pytest.raises(ValueError, match="outputs do not match"):
        CachedGeneration(config, reversed_outputs).generate(requests, 256)
    assert not (tmp_path / "generation_batches.jsonl").exists()


def test_invalid_draft_json_gets_a_distinct_corrective_prompt_and_saved_failure(
    tmp_path: Path,
) -> None:
    config = configuration(tmp_path).model_copy(
        update={"dataset": EmotionalDatasetConfig(utterance_count=5, batch_size=5)}
    )
    fixture = FixtureBackend(config)
    original_id = next(iter(fixture.drafts))
    calls: list[TextGenerationRequest] = []

    def generate(requests: Sequence[TextGenerationRequest], cap: int) -> TextGenerationBatch:
        calls.extend(requests)
        return TextGenerationBatch(
            token_cap=cap,
            seed=42,
            runtime_seconds=1,
            outcomes=tuple(
                TextGenerationOutcome(
                    request=request,
                    prompt_text=request.messages[-1].content,
                    prompt_token_ids=(1,),
                    generation=CompletedGeneration(
                        text="not JSON"
                        if request.request_id == original_id
                        else fixture.drafts[original_id].model_dump_json(),
                        token_ids=(2, 3),
                    ),
                )
                for request in requests
            ),
        )

    rows = generate_drafts(config, CachedGeneration(config, generate))
    assert len(rows) == 5
    assert len(calls) == 2
    assert calls[0].request_id != calls[1].request_id
    assert "Previous schema error" in calls[1].messages[-1].content
    failures = read_journal(tmp_path / "draft_syntax_failures.jsonl", DraftSyntaxFailure)
    assert len(failures) == 1
    assert failures[0].response.text == "not JSON"


@pytest.mark.parametrize("finish_reason", ("stop", "length"))
def test_provider_stop_status_does_not_require_a_returned_eos_token(finish_reason: str) -> None:
    response = generation_from_finish_reason("  Reply  ", (10, 11), finish_reason)
    assert response.token_ids == (10, 11)
    assert isinstance(response, CompletedGeneration) == (finish_reason == "stop")


@pytest.mark.parametrize("finish_reason", (None, "abort", "tool_calls"))
def test_unrecognized_provider_stop_never_becomes_completed(finish_reason: str | None) -> None:
    with pytest.raises(ValueError, match="Unexpected provider termination"):
        generation_from_finish_reason("Reply", (10, 11), finish_reason)


def test_provider_trace_directories_preserve_independent_provenance_for_shared_dataset(
    tmp_path: Path,
) -> None:
    first = configuration(tmp_path)
    backend = FixtureBackend(first)
    requests = (
        TextGenerationRequest(
            request_id="teacher:test", messages=(ChatMessage(role="user", content="Hello"),)
        ),
    )
    CachedGeneration(first, backend.generate).generate(requests, 256)
    original = (tmp_path / "generation_provenance.json").read_bytes()
    second = first.model_copy(
        update={
            "backend": GenerationBackend.VLLM,
            "trace_subdirectory": Path("vllm"),
            "source_git_commit": "c" * 40,
        }
    )
    CachedGeneration(second, backend.generate).generate(requests, 256)
    assert (tmp_path / "generation_provenance.json").read_bytes() == original
    assert (
        len(read_journal(tmp_path / "vllm" / "generation_batches.jsonl", TextGenerationBatch)) == 1
    )
    assert summarize_generation(second).generation_batches == 1
    with pytest.raises(ValueError, match="within the dataset"):
        EmotionalGenerationConfig.model_validate_json(
            first.model_copy(update={"trace_subdirectory": Path("../outside")}).model_dump_json()
        )


def test_vllm_runtime_has_bounded_node_defaults_and_validates_resource_limits() -> None:
    runtime = VllmRuntimeConfig()
    assert runtime.max_num_seqs == 32
    assert runtime.gpu_memory_utilization == 0.75
    assert VllmRuntimeConfig.model_validate_json(runtime.model_dump_json()) == runtime
    with pytest.raises(ValueError):
        VllmRuntimeConfig(max_num_seqs=0)
    with pytest.raises(ValueError):
        VllmRuntimeConfig(gpu_memory_utilization=1.1)
