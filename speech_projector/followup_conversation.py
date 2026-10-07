"""Matched words-only conversation baselines alongside retained speech histories."""

from __future__ import annotations

import gc
import hashlib
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from pydantic import Field
from safetensors.torch import load_file
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

from scripts.inventory_results import write_record
from speech_projector.data import load_audio
from speech_projector.followup_evaluation import bind_provenance, file_artifact
from speech_projector.generation import CompletedGeneration
from speech_projector.inputs import TranscriptInput
from speech_projector.journal import append_record, read_journal
from speech_projector.launcher import initialize_projector
from speech_projector.llm import ConversationRequest, FrozenQwen, example_prompt
from speech_projector.models import (
    AsrTranscript,
    EvaluationCondition,
    FileArtifact,
    Record,
    Role,
    RunResult,
    Turn,
)
from speech_projector.overnight_conversation import DelayedCueFixtures, DelayedCueScenario
from speech_projector.overnight_conversation_execution import (
    ControlledConversationReply,
    ConversationHistoryMode,
    RolloutConversationReply,
    SavedConversationReply,
    _assert_history_retained,
    evaluate_conversations,
    reply_identifier,
    response_text,
)


class FollowupAsrConfig(Record):
    model_name: str = "openai/whisper-small"
    revision: str = Field(min_length=7)
    audio_root: Path


class FollowupAsrProvenance(Record):
    configuration: FollowupAsrConfig
    audio: tuple[FileArtifact, ...]
    fixtures_sha256: str


class PipelineConversationReply(Record):
    condition: EvaluationCondition
    user_words: tuple[str, ...]
    reply: SavedConversationReply


class ConversationBaselineProvenance(Record):
    source_git_commit: str
    condition: EvaluationCondition
    configuration_sha256: str
    fixtures_sha256: str
    transcripts: tuple[AsrTranscript, ...]


class PipelineConversationSummary(Record):
    condition: EvaluationCondition
    controlled_responses: int
    rollout_responses: int
    completed: int
    token_limited: int
    generation_seconds: float


class FollowupConversationConfig(Record):
    run_result: Path
    fixtures: Path
    asr_transcripts: Path
    followup_asr: FollowupAsrConfig
    output_directory: Path
    source_git_commit: str = Field(min_length=7)
    history_tokens: int = Field(default=8192, ge=1)


class FollowupConversationProvenance(Record):
    configuration: FollowupConversationConfig
    inputs: tuple[FileArtifact, ...]


def user_turns(
    scenario: DelayedCueScenario,
    branch: int,
    condition: EvaluationCondition,
    transcripts: Sequence[AsrTranscript],
) -> tuple[str, ...]:
    initial = scenario.initial[branch]
    if condition == EvaluationCondition.TEXT:
        return (initial.user_text,) + tuple(item.clip.case.text for item in scenario.followups)
    if condition != EvaluationCondition.ASR:
        raise ValueError("Words-only conversation baseline requires text or ASR")
    by_id = {row.example_id: row.text for row in transcripts}
    identifiers = (initial.example_id,) + tuple(
        item.clip.case.case_id for item in scenario.followups
    )
    if set(identifiers) - by_id.keys():
        raise ValueError("ASR conversation baseline lacks a current or previous user transcription")
    return tuple(by_id[identifier] for identifier in identifiers)


def conversation_history(
    users: Sequence[str], assistants: Sequence[Turn], turn_index: int
) -> tuple[Turn, ...]:
    return tuple(
        item
        for index in range(turn_index)
        for item in (Turn(role=Role.USER, text=users[index]), assistants[index])
    )


def summary(
    condition: EvaluationCondition, replies: Sequence[PipelineConversationReply]
) -> PipelineConversationSummary:
    return PipelineConversationSummary(
        condition=condition,
        controlled_responses=sum(
            isinstance(row.reply, ControlledConversationReply) for row in replies
        ),
        rollout_responses=sum(isinstance(row.reply, RolloutConversationReply) for row in replies),
        completed=sum(isinstance(row.reply.generation, CompletedGeneration) for row in replies),
        token_limited=sum(
            not isinstance(row.reply.generation, CompletedGeneration) for row in replies
        ),
        generation_seconds=sum(row.reply.generation_seconds for row in replies),
    )


@torch.no_grad()
def evaluate_conversation_baseline(
    wrapper: FrozenQwen,
    fixtures: DelayedCueFixtures,
    condition: EvaluationCondition,
    transcripts: Sequence[AsrTranscript],
    output_directory: Path,
    *,
    source_git_commit: str,
) -> tuple[PipelineConversationReply, ...]:
    if condition not in (EvaluationCondition.TEXT, EvaluationCondition.ASR):
        raise ValueError("Conversation baseline requires true words or recognized words")
    provenance = ConversationBaselineProvenance(
        source_git_commit=source_git_commit,
        condition=condition,
        configuration_sha256=hashlib.sha256(wrapper.config.model_dump_json().encode()).hexdigest(),
        fixtures_sha256=hashlib.sha256(fixtures.model_dump_json().encode()).hexdigest(),
        transcripts=tuple(transcripts) if condition == EvaluationCondition.ASR else (),
    )
    bind_provenance(output_directory / "provenance.json", provenance)
    journal = output_directory / "replies.jsonl"
    replies = list(read_journal(journal, PipelineConversationReply))
    saved = {reply_identifier(row.reply): row for row in replies}
    if len(saved) != len(replies):
        raise ValueError("Conversation baseline has duplicate responses")
    expected: set[str] = set()
    for scenario in fixtures.scenarios:
        for branch in range(2):
            initial = scenario.initial[branch]
            users = user_turns(scenario, branch, condition, transcripts)
            truth = user_turns(scenario, branch, EvaluationCondition.TEXT, ())
            for turn_index in (2, 3):
                identifier = f"controlled:{initial.example_id}:text_history:{turn_index}"
                expected.add(identifier)
                if identifier in saved:
                    continue
                request = ConversationRequest(
                    identifier=identifier,
                    prompt=example_prompt(initial, wrapper.config),
                    history=conversation_history(users, scenario.fixed_assistants, turn_index),
                    current=TranscriptInput(users[turn_index]),
                )
                _assert_history_retained(wrapper, request)
                started = time.perf_counter()
                generation = wrapper.generate_conversation(request, wrapper.config.max_new_tokens)
                reply = ControlledConversationReply(
                    scenario_base_id=scenario.base_id,
                    initial_example_id=initial.example_id,
                    current_case_id=scenario.followups[turn_index - 1].clip.case.case_id,
                    history_for_judge=conversation_history(
                        truth, scenario.fixed_assistants, turn_index
                    ),
                    history_mode=ConversationHistoryMode.TEXT_HISTORY,
                    turn_index=turn_index,
                    generation=generation,
                    generation_seconds=time.perf_counter() - started,
                )
                row = PipelineConversationReply(condition=condition, user_words=users, reply=reply)
                append_record(journal, row)
                replies.append(row)
                saved[identifier] = row
            assistants: list[Turn] = []
            for turn_index in range(4):
                identifier = f"rollout:{initial.example_id}:{turn_index}"
                expected.add(identifier)
                existing = saved.get(identifier)
                if existing is None:
                    request = ConversationRequest(
                        identifier=identifier,
                        prompt=example_prompt(initial, wrapper.config),
                        history=conversation_history(users, assistants, turn_index),
                        current=TranscriptInput(users[turn_index]),
                    )
                    _assert_history_retained(wrapper, request)
                    started = time.perf_counter()
                    generation = wrapper.generate_conversation(
                        request, wrapper.config.max_new_tokens
                    )
                    reply = RolloutConversationReply(
                        scenario_base_id=scenario.base_id,
                        initial_example_id=initial.example_id,
                        current_case_id=initial.example_id
                        if turn_index == 0
                        else scenario.followups[turn_index - 1].clip.case.case_id,
                        history_for_judge=conversation_history(truth, assistants, turn_index),
                        turn_index=turn_index,
                        generation=generation,
                        generation_seconds=time.perf_counter() - started,
                    )
                    existing = PipelineConversationReply(
                        condition=condition, user_words=users, reply=reply
                    )
                    append_record(journal, existing)
                    replies.append(existing)
                    saved[identifier] = existing
                assistants.append(
                    Turn(role=Role.ASSISTANT, text=response_text(existing.reply.generation))
                )
    if set(saved) != expected:
        raise ValueError("Conversation baseline journal has unexpected fixture coverage")
    write_record(output_directory / "summary.json", summary(condition, replies))
    return tuple(replies)


@torch.no_grad()
def transcribe_followups(
    fixtures: DelayedCueFixtures, configuration: FollowupAsrConfig, output_directory: Path
) -> tuple[AsrTranscript, ...]:
    clips = fixtures.scenarios[0].followups
    if any(scenario.followups != clips for scenario in fixtures.scenarios):
        raise ValueError("Reference comparison requires shared neutral follow-up clips")
    paths = tuple(configuration.audio_root / item.clip.audio_path for item in clips)
    artifacts = tuple(file_artifact(path) for path in paths)
    if tuple(row.sha256 for row in artifacts) != tuple(item.clip.sha256 for item in clips):
        raise ValueError("Follow-up waveform hashes differ from fixed fixture records")
    provenance = FollowupAsrProvenance(
        configuration=configuration,
        audio=artifacts,
        fixtures_sha256=hashlib.sha256(fixtures.model_dump_json().encode()).hexdigest(),
    )
    bind_provenance(output_directory / "provenance.json", provenance)
    journal = output_directory / "transcripts.jsonl"
    saved = read_journal(journal, AsrTranscript)
    identifiers = tuple(item.clip.case.case_id for item in clips)
    if saved:
        if tuple(row.example_id for row in saved) != identifiers:
            raise ValueError("Follow-up ASR journal has incomplete or unexpected coverage")
        return saved
    model = (
        WhisperForConditionalGeneration.from_pretrained(
            configuration.model_name, revision=configuration.revision, local_files_only=True
        )
        .to("cuda")
        .eval()
    )
    extractor = WhisperFeatureExtractor.from_pretrained(
        configuration.model_name, revision=configuration.revision, local_files_only=True
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        configuration.model_name, revision=configuration.revision, local_files_only=True
    )
    audio = [load_audio(path) for path in paths]
    batch = extractor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    tokens = model.generate(
        batch.input_features.to("cuda"),
        attention_mask=batch.attention_mask.to("cuda"),
        language="en",
        task="transcribe",
        max_new_tokens=128,
    )
    text = tokenizer.batch_decode(
        tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    records = tuple(
        AsrTranscript(example_id=identifier, text=words)
        for identifier, words in zip(identifiers, text, strict=True)
    )
    partial = journal.with_suffix(".part")
    partial.write_text("".join(row.model_dump_json() + "\n" for row in records), encoding="utf-8")
    partial.replace(journal)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return records


def render_pipeline_conversations(replies: Sequence[PipelineConversationReply]) -> str:
    lines = [
        "# Matched conversation pipeline comparison",
        "",
        "Text and ASR contain words at every user turn; speech retains audio embeddings. "
        "The controlled assistant turns are shared; "
        "rollout assistants are each pipeline's own replies.",
        "",
    ]
    for row in replies:
        lines.extend(
            [
                f"## {row.condition.value} / {reply_identifier(row.reply)}",
                "",
                f"User words (speech audit only): {row.user_words}",
                "",
                f"Actual response: {response_text(row.reply.generation)}",
                "",
                f"Termination: {row.reply.generation.kind}",
                "",
            ]
        )
    return "\n".join(lines)


def matched_speech_replies(
    replies: Sequence[SavedConversationReply], fixtures: DelayedCueFixtures
) -> tuple[PipelineConversationReply, ...]:
    by_id = {
        row.example_id: (scenario, branch)
        for scenario in fixtures.scenarios
        for branch, row in enumerate(scenario.initial)
    }
    selected = tuple(
        row
        for row in replies
        if isinstance(row, RolloutConversationReply)
        or row.history_mode == ConversationHistoryMode.RETAINED_AUDIO
    )
    return tuple(
        PipelineConversationReply(
            condition=EvaluationCondition.SPEECH,
            user_words=user_turns(*by_id[row.initial_example_id], EvaluationCondition.TEXT, ()),
            reply=row,
        )
        for row in selected
    )


def run_followup_conversation(configuration: FollowupConversationConfig) -> None:
    result = RunResult.model_validate_json(configuration.run_result.read_bytes())
    fixtures = DelayedCueFixtures.model_validate_json(configuration.fixtures.read_bytes())
    provenance = FollowupConversationProvenance(
        configuration=configuration,
        inputs=tuple(
            file_artifact(path)
            for path in (
                configuration.run_result,
                result.checkpoint_path,
                configuration.fixtures,
                configuration.asr_transcripts,
            )
        ),
    )
    bind_provenance(configuration.output_directory / "provenance.json", provenance)
    followups = transcribe_followups(
        fixtures, configuration.followup_asr, configuration.output_directory / "followup_asr"
    )
    initial = read_journal(configuration.asr_transcripts, AsrTranscript)
    transcripts = initial + followups
    if len({row.example_id for row in transcripts}) != len(transcripts):
        raise ValueError("Initial and neutral follow-up ASR journals have overlapping IDs")
    config = result.config.model_copy(
        update={
            "history_turns": 6,
            "max_history_tokens": configuration.history_tokens,
        }
    )
    wrapper = FrozenQwen(config, torch.device("cuda"))
    projector = initialize_projector(config, wrapper.device)
    projector.load_state_dict(load_file(str(result.checkpoint_path), device=str(wrapper.device)))
    speech = evaluate_conversations(
        wrapper,
        projector,
        fixtures,
        configuration.output_directory / "speech",
        source_commit=configuration.source_git_commit,
    )
    replies = list(matched_speech_replies(speech, fixtures))
    write_record(
        configuration.output_directory / "speech" / "matched_summary.json",
        summary(EvaluationCondition.SPEECH, replies),
    )
    for condition in (EvaluationCondition.TEXT, EvaluationCondition.ASR):
        replies.extend(
            evaluate_conversation_baseline(
                wrapper,
                fixtures,
                condition,
                transcripts,
                configuration.output_directory / condition.value,
                source_git_commit=configuration.source_git_commit,
            )
        )
    destination = configuration.output_directory / "matched_replies.jsonl"
    partial = destination.with_suffix(".part")
    partial.write_text("".join(row.model_dump_json() + "\n" for row in replies), encoding="utf-8")
    partial.replace(destination)
    (configuration.output_directory / "matched_comparison.md").write_text(
        render_pipeline_conversations(replies), encoding="utf-8"
    )
