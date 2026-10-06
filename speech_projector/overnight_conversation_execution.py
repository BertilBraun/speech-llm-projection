"""Final-only controlled delayed-cue responses and separate acoustic conversation rollouts."""

from __future__ import annotations

import hashlib
import time
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

import torch
from pydantic import Field, TypeAdapter
from torch import Tensor

from speech_projector.generation import (
    CompletedGeneration,
    GenerationResult,
    TokenLimitedGeneration,
)
from speech_projector.inputs import SpeechInput
from speech_projector.journal import append_record, read_journal
from speech_projector.llm import ConversationRequest, FrozenQwen, SpeechHistoryTurn, example_prompt
from speech_projector.models import Record, Role, RunConfig, Turn
from speech_projector.overnight_conversation import DelayedCueFixtures, DelayedCueScenario
from speech_projector.projectors import Projector
from speech_projector.teacher_evaluation import projector_digest


class ConversationHistoryMode(str, Enum):
    RETAINED_AUDIO = "retained_audio"
    TEXT_HISTORY = "text_history"
    OMIT_INITIAL = "omit_initial"


class ConversationReply(Record):
    scenario_base_id: str
    initial_example_id: str
    current_case_id: str
    history_for_judge: tuple[Turn, ...]
    generation: GenerationResult
    generation_seconds: float = Field(ge=0)


class ControlledConversationReply(ConversationReply):
    kind: Literal["controlled"] = "controlled"
    history_mode: ConversationHistoryMode
    turn_index: Literal[2, 3]


class RolloutConversationReply(ConversationReply):
    kind: Literal["rollout"] = "rollout"
    turn_index: int = Field(ge=0, le=3)


SavedConversationReply: TypeAlias = Annotated[
    ControlledConversationReply | RolloutConversationReply, Field(discriminator="kind")
]


class ConversationEvaluationProvenance(Record):
    source_commit: str
    configuration: RunConfig
    projector_weights_sha256: str
    fixtures_sha256: str


class ConversationEvaluationSummary(Record):
    scenarios: int
    controlled_responses: int
    rollouts: int
    rollout_responses: int
    completed_responses: int
    token_limited_responses: int
    generation_seconds: float


def reply_identifier(reply: SavedConversationReply) -> str:
    match reply:
        case ControlledConversationReply():
            return (
                f"controlled:{reply.initial_example_id}:"
                f"{reply.history_mode.value}:{reply.turn_index}"
            )
        case RolloutConversationReply():
            return f"rollout:{reply.initial_example_id}:{reply.turn_index}"


def response_text(response: GenerationResult) -> str:
    match response:
        case CompletedGeneration(text=text) | TokenLimitedGeneration(partial_text=text):
            return text


def _assert_history_retained(wrapper: FrozenQwen, request: ConversationRequest) -> None:
    if len(request.history) > wrapper.config.history_turns:
        raise ValueError(
            "Conversation history would lose turns under the recorded evaluation budget"
        )
    tokens = 0
    for turn in request.history:
        match turn:
            case SpeechHistoryTurn(utterance=SpeechInput(embeddings=embeddings)):
                role = Role.USER
                body_tokens = embeddings.shape[0]
            case Turn(role=role, text=text):
                body_tokens = len(wrapper.tokenizer.encode(text, add_special_tokens=False))
        tokens += (
            body_tokens
            + len(wrapper.tokenizer.encode(f"<|im_start|>{role.value}\n", add_special_tokens=False))
            + len(wrapper.tokenizer.encode("<|im_end|>\n", add_special_tokens=False))
        )
    if tokens > wrapper.config.max_history_tokens:
        raise ValueError(
            "Conversation history would truncate a cue under its evaluation token budget"
        )


def _truth_history(
    scenario: DelayedCueScenario, branch: int, turn_index: int, assistants: tuple[Turn, ...]
) -> tuple[Turn, ...]:
    users = (scenario.initial[branch].user_text,) + tuple(
        item.clip.case.text for item in scenario.followups
    )
    return tuple(
        item
        for index in range(turn_index)
        for item in (Turn(role=Role.USER, text=users[index]), assistants[index])
    )


def _controlled_history(
    speeches: tuple[SpeechInput, ...],
    truth: tuple[Turn, ...],
    mode: ConversationHistoryMode,
) -> tuple[Turn | SpeechHistoryTurn, ...]:
    if mode == ConversationHistoryMode.TEXT_HISTORY:
        return truth
    start = 1 if mode == ConversationHistoryMode.OMIT_INITIAL else 0
    return tuple(
        item
        for index in range(start, len(truth) // 2)
        for item in (SpeechHistoryTurn(speeches[index]), truth[2 * index + 1])
    )


@torch.no_grad()
def evaluate_conversations(
    wrapper: FrozenQwen,
    projector: Projector,
    fixtures: DelayedCueFixtures,
    output_directory: Path,
    *,
    source_commit: str,
) -> tuple[SavedConversationReply, ...]:
    wrapper.model.eval()
    projector.eval()
    output_directory.mkdir(parents=True, exist_ok=True)
    provenance = ConversationEvaluationProvenance(
        source_commit=source_commit,
        configuration=wrapper.config,
        projector_weights_sha256=projector_digest(projector),
        fixtures_sha256=hashlib.sha256(fixtures.model_dump_json().encode()).hexdigest(),
    )
    provenance_path = output_directory / "provenance.json"
    journal = output_directory / "replies.jsonl"
    if provenance_path.exists():
        if (
            ConversationEvaluationProvenance.model_validate_json(provenance_path.read_bytes())
            != provenance
        ):
            raise ValueError("Conversation output differs in inference config, weights or fixtures")
    else:
        if journal.exists():
            raise ValueError("Conversation journal lacks required provenance")
        pending_path = provenance_path.with_suffix(".json.part")
        pending_path.write_text(provenance.model_dump_json(indent=2) + "\n", encoding="utf-8")
        pending_path.replace(provenance_path)
    replies = list(read_journal(journal, TypeAdapter(SavedConversationReply)))
    saved = {reply_identifier(item): item for item in replies}
    if len(saved) != len(replies):
        raise ValueError("Conversation journal has duplicate response IDs")
    expected: set[str] = set()

    def project(path: Path) -> SpeechInput:
        features: Tensor = torch.load(path, map_location="cpu", weights_only=True)
        return SpeechInput(projector(features.to(wrapper.device)))

    for scenario in fixtures.scenarios:
        followups = tuple(project(item.feature_path) for item in scenario.followups)
        for branch in range(2):
            initial = scenario.initial[branch]
            speeches = (project(initial.feature_path),) + followups
            prompt = example_prompt(initial, wrapper.config)
            for turn_index in (2, 3):
                truth = _truth_history(scenario, branch, turn_index, scenario.fixed_assistants)
                for mode in ConversationHistoryMode:
                    identifier = f"controlled:{initial.example_id}:{mode.value}:{turn_index}"
                    expected.add(identifier)
                    if identifier in saved:
                        continue
                    request = ConversationRequest(
                        identifier=identifier,
                        prompt=prompt,
                        history=_controlled_history(speeches, truth, mode),
                        current=speeches[turn_index],
                    )
                    _assert_history_retained(wrapper, request)
                    started = time.perf_counter()
                    generation = wrapper.generate_conversation(
                        request, wrapper.config.max_new_tokens
                    )
                    reply = ControlledConversationReply(
                        scenario_base_id=scenario.base_id,
                        initial_example_id=initial.example_id,
                        current_case_id=scenario.followups[turn_index - 1].clip.case.case_id,
                        history_for_judge=truth,
                        history_mode=mode,
                        turn_index=turn_index,
                        generation=generation,
                        generation_seconds=time.perf_counter() - started,
                    )
                    append_record(journal, reply)
                    replies.append(reply)
                    saved[identifier] = reply
            assistants: list[Turn] = []
            for turn_index in range(4):
                identifier = f"rollout:{initial.example_id}:{turn_index}"
                expected.add(identifier)
                truth = _truth_history(scenario, branch, turn_index, tuple(assistants))
                existing = saved.get(identifier)
                if existing is None:
                    request = ConversationRequest(
                        identifier=identifier,
                        prompt=prompt,
                        history=_controlled_history(
                            speeches, truth, ConversationHistoryMode.RETAINED_AUDIO
                        ),
                        current=speeches[turn_index],
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
                        history_for_judge=truth,
                        turn_index=turn_index,
                        generation=generation,
                        generation_seconds=time.perf_counter() - started,
                    )
                    append_record(journal, reply)
                    replies.append(reply)
                    saved[identifier] = reply
                else:
                    generation = existing.generation
                assistants.append(Turn(role=Role.ASSISTANT, text=response_text(generation)))
    if set(saved) != expected:
        raise ValueError(
            "Conversation journal contains responses outside its fixed fixture coverage"
        )
    summary = ConversationEvaluationSummary(
        scenarios=len(fixtures.scenarios),
        controlled_responses=sum(isinstance(item, ControlledConversationReply) for item in replies),
        rollouts=2 * len(fixtures.scenarios),
        rollout_responses=sum(isinstance(item, RolloutConversationReply) for item in replies),
        completed_responses=sum(
            isinstance(item.generation, CompletedGeneration) for item in replies
        ),
        token_limited_responses=sum(
            isinstance(item.generation, TokenLimitedGeneration) for item in replies
        ),
        generation_seconds=sum(item.generation_seconds for item in replies),
    )
    summary_path = output_directory / "summary.json"
    pending_path = summary_path.with_suffix(".json.part")
    pending_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    pending_path.replace(summary_path)
    return tuple(replies)
