"""Control histories, four-turn rollout coverage, exact resume and budget guards."""

from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn

from speech_projector.generation import CompletedGeneration, GenerationResult
from speech_projector.llm import ConversationRequest, FrozenQwen, SpeechHistoryTurn
from speech_projector.models import Example, RunConfig, Split, Turn
from speech_projector.overnight_conversation import (
    DelayedCueConfiguration,
    build_delayed_cue_fixtures,
)
from speech_projector.overnight_conversation_execution import (
    ControlledConversationReply,
    RolloutConversationReply,
    evaluate_conversations,
)
from speech_projector.overnight_conversation_reporting import (
    conversation_judge_requests,
    render_conversations,
)
from speech_projector.overnight_data import NeuEmotionalExampleSource
from speech_projector.projectors import Projector
from speech_projector.tts_pilot import PilotEmotion
from tests.test_overnight_conversation import followups
from tests.test_overnight_evaluation import TestProjector, candidate, example


class TokenCounter:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [1] * len(text.split())


class ConversationWrapper:
    def __init__(self, configuration: RunConfig) -> None:
        self.config = configuration
        self.device = torch.device("cpu")
        self.model = nn.Linear(1, 1)
        self.tokenizer = TokenCounter()
        self.calls: list[ConversationRequest] = []

    def generate_conversation(
        self, request: ConversationRequest, max_new_tokens: int
    ) -> GenerationResult:
        self.calls.append(request)
        return CompletedGeneration(text="Actual generated assistant text.", token_ids=(1, 2))


def test_controls_retain_only_intended_acoustic_history_and_rollouts_resume(tmp_path: Path) -> None:
    examples: list[Example] = []
    sources: list[NeuEmotionalExampleSource] = []
    for tone, value in ((PilotEmotion.HAPPY, 1), (PilotEmotion.ANGRY, 3)):
        identifier = f"base_{tone.value}"
        feature_path = tmp_path / f"{identifier}.pt"
        torch.save(torch.full((2, 768), value, dtype=torch.float32), feature_path)
        examples.append(
            example(identifier, "The deadline changed again.").model_copy(
                update={"split": Split.TEST, "feature_path": feature_path}
            )
        )
        sources.append(
            NeuEmotionalExampleSource(
                example_id=identifier,
                source_manifest=Path("source"),
                source_example_id=identifier,
                base_id="base",
                family_id="family",
                emotion=tone,
            )
        )
    shared = tuple(
        item.model_copy(update={"feature_path": tmp_path / item.feature_path})
        for item in followups()
    )
    for item in shared:
        torch.save(torch.ones((2, 768)), item.feature_path)
    fixtures = build_delayed_cue_fixtures(
        examples, sources, (shared[0], shared[1], shared[2]), DelayedCueConfiguration(scenarios=1)
    )
    config = candidate("conversation", 1, 1, 0.1, 10).configuration.model_copy(
        update={"history_turns": 6, "max_history_tokens": 2048}
    )
    wrapper = ConversationWrapper(config)
    projector = TestProjector()
    replies = evaluate_conversations(
        cast(FrozenQwen, wrapper),
        cast(Projector, projector),
        fixtures,
        tmp_path / "evaluation",
        source_commit="source",
    )
    assert len(replies) == len(wrapper.calls) == 20
    assert len(conversation_judge_requests(replies, fixtures)) == 12
    assert len(conversation_judge_requests(replies, fixtures, include_rollout_next_step=True)) == 14
    readable = render_conversations(replies, fixtures)
    assert readable.count("Actual termination:") == 20
    assert "deliberately deprived" in readable and "controlled / omit_initial" in readable
    assert sum(isinstance(item, ControlledConversationReply) for item in replies) == 12
    assert sum(isinstance(item, RolloutConversationReply) for item in replies) == 8
    by_id = {item.identifier: item for item in wrapper.calls}
    retained = by_id[f"controlled:{examples[0].example_id}:retained_audio:2"]
    text = by_id[f"controlled:{examples[0].example_id}:text_history:2"]
    omitted = by_id[f"controlled:{examples[0].example_id}:omit_initial:2"]
    assert len(retained.history) == len(text.history) == 4
    assert len(omitted.history) == 2
    assert isinstance(retained.history[0], SpeechHistoryTurn)
    assert all(isinstance(item, Turn) for item in text.history)
    assert retained.current == text.current == omitted.current
    rollout = by_id[f"rollout:{examples[0].example_id}:2"]
    assert isinstance(rollout.history[1], Turn)
    assert rollout.history[1].text == "Actual generated assistant text."
    controlled = tuple(item for item in replies if isinstance(item, ControlledConversationReply))
    for current in controlled:
        assert current.history_for_judge[1] == fixtures.scenarios[0].fixed_assistants[0]
    assert (
        evaluate_conversations(
            cast(FrozenQwen, wrapper),
            cast(Projector, projector),
            fixtures,
            tmp_path / "evaluation",
            source_commit="source",
        )
        == replies
    )
    assert len(wrapper.calls) == 20
    too_small = ConversationWrapper(config.model_copy(update={"max_history_tokens": 1}))
    with pytest.raises(ValueError, match="truncate a cue"):
        evaluate_conversations(
            cast(FrozenQwen, too_small),
            cast(Projector, projector),
            fixtures,
            tmp_path / "small",
            source_commit="source",
        )
    assert not too_small.calls
