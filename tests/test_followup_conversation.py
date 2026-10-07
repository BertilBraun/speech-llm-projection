"""Conversation baselines use recognized words at every user turn, including past turns."""

from pathlib import Path
from typing import cast

from speech_projector.followup_conversation import evaluate_conversation_baseline
from speech_projector.inputs import TranscriptInput
from speech_projector.llm import FrozenQwen
from speech_projector.models import AsrTranscript, EvaluationCondition, Split, Turn
from speech_projector.overnight_conversation import (
    DelayedCueConfiguration,
    build_delayed_cue_fixtures,
)
from speech_projector.overnight_data import NeuEmotionalExampleSource
from speech_projector.tts_pilot import PilotEmotion
from tests.test_overnight_conversation import followups
from tests.test_overnight_conversation_execution import ConversationWrapper
from tests.test_overnight_evaluation import candidate, example


def test_asr_history_current_input_and_own_rollouts_are_fully_matched(tmp_path: Path) -> None:
    examples = tuple(
        example(f"base_{tone.value}", "The meeting moved to five.").model_copy(
            update={"split": Split.TEST}
        )
        for tone in (PilotEmotion.HAPPY, PilotEmotion.ANGRY)
    )
    sources = tuple(
        NeuEmotionalExampleSource(
            example_id=row.example_id,
            source_manifest=Path("source"),
            source_example_id=row.example_id,
            base_id="base",
            family_id="family",
            emotion=tone,
        )
        for row, tone in zip(examples, (PilotEmotion.HAPPY, PilotEmotion.ANGRY), strict=True)
    )
    fixtures = build_delayed_cue_fixtures(
        examples, sources, followups(), DelayedCueConfiguration(scenarios=1)
    )
    recognized = tuple(
        AsrTranscript(example_id=row.example_id, text="The meeting moved to nine.")
        for row in examples
    ) + tuple(
        AsrTranscript(example_id=row.clip.case.case_id, text=f"Recognized follow-up {index}.")
        for index, row in enumerate(fixtures.scenarios[0].followups)
    )
    config = candidate("test", 1, 1, 0.1, 10).configuration.model_copy(
        update={"history_turns": 6, "max_history_tokens": 2048}
    )
    wrapper = ConversationWrapper(config)
    replies = evaluate_conversation_baseline(
        cast(FrozenQwen, wrapper),
        fixtures,
        EvaluationCondition.ASR,
        recognized,
        tmp_path,
        source_git_commit="commit42",
    )
    assert len(replies) == len(wrapper.calls) == 12
    controlled = next(row for row in wrapper.calls if row.identifier.endswith("text_history:2"))
    assert controlled.current == TranscriptInput("Recognized follow-up 1.")
    assert controlled.history[0] == Turn(role="user", text="The meeting moved to nine.")
    assert all(isinstance(row, Turn) for row in controlled.history)
    assert replies[0].reply.history_for_judge[0].text == "The meeting moved to five."
    rollout = next(
        row
        for row in wrapper.calls
        if row.identifier.startswith("rollout:") and row.identifier.endswith(":2")
    )
    assert rollout.history[1].text == "Actual generated assistant text."
    evaluate_conversation_baseline(
        cast(FrozenQwen, wrapper),
        fixtures,
        EvaluationCondition.ASR,
        recognized,
        tmp_path,
        source_git_commit="commit42",
    )
    assert len(wrapper.calls) == 12
