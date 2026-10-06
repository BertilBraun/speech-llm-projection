"""Conversation fixtures preserve paired words and make no outcome-driven selection."""

from pathlib import Path

import pytest

from scripts.prepare_overnight_followups import followup_manifest
from speech_projector.judge import JudgeRequest
from speech_projector.models import Role, Split, Turn
from speech_projector.overnight_conversation import (
    DelayedCueConfiguration,
    DelayedCueFixtures,
    FollowupAudio,
    build_delayed_cue_fixtures,
)
from speech_projector.overnight_data import NeuEmotionalExampleSource
from speech_projector.overnight_judge import (
    ConversationToneJudgeRequest,
    PastUserDelivery,
    tone_prompt,
)
from speech_projector.tts_pilot import PilotEmotion, PilotTermination, TtsPilotClip
from tests.test_overnight_evaluation import example


def followups() -> tuple[FollowupAudio, FollowupAudio, FollowupAudio]:
    clips = tuple(
        FollowupAudio(
            clip=TtsPilotClip(
                case=case,
                audio_path=case.case_id + ".wav",
                sha256="a" * 64,
                sample_rate=24000,
                audio_seconds=2,
                generation_seconds=1,
                real_time_factor=0.5,
                termination=PilotTermination.STOP,
            ),
            feature_path=Path(case.case_id + ".pt"),
        )
        for case in followup_manifest().cases
        if case.case_id != "followup_summary"
    )
    return clips[0], clips[1], clips[2]


def test_fixture_selection_is_heldout_repeatable_and_keeps_shared_assistants_audio() -> None:
    examples = []
    sources = []
    for base in range(5):
        for tone in (PilotEmotion.HAPPY, PilotEmotion.ANGRY):
            identifier = f"base_{base}_{tone.value}"
            examples.append(
                example(identifier, f"The deadline for project {base} changed again.").model_copy(
                    update={"split": Split.TEST}
                )
            )
            sources.append(
                NeuEmotionalExampleSource(
                    example_id=identifier,
                    source_manifest=Path("neu"),
                    source_example_id=identifier,
                    base_id=f"base_{base}",
                    family_id=f"family_{base}",
                    emotion=tone,
                )
            )
    configuration = DelayedCueConfiguration()
    fixtures = build_delayed_cue_fixtures(examples, sources, followups(), configuration)
    assert len(fixtures.scenarios) == 3
    assert fixtures == build_delayed_cue_fixtures(examples, sources, followups(), configuration)
    assert DelayedCueFixtures.model_validate_json(fixtures.model_dump_json()) == fixtures
    for scenario in fixtures.scenarios:
        assert scenario.initial[0].user_text == scenario.initial[1].user_text
        assert all(item.clip.case.emotion == PilotEmotion.NEUTRAL for item in scenario.followups)
        assert scenario.fixed_assistants == fixtures.scenarios[0].fixed_assistants
    with pytest.raises(ValueError, match="held-out"):
        DelayedCueConfiguration(split=Split.TRAIN)
    bad_followups = followups()[0].model_copy(
        update={"clip": followups()[0].clip.model_copy(update={"audio_seconds": 31})}
    )
    with pytest.raises(ValueError, match="30-second"):
        build_delayed_cue_fixtures(
            examples, sources, (bad_followups, followups()[1], followups()[2]), configuration
        )


def test_conversation_judge_distinguishes_past_delivery_from_current_neutral_speech() -> None:
    request = JudgeRequest(
        example_id="hidden",
        dialogue_id="family",
        history=(
            Turn(role=Role.USER, text="The deadline changed again."),
            Turn(role=Role.ASSISTANT, text="I hear you."),
        ),
        transcript="What would be a sensible next step?",
        candidate_response="You could ask for a stable deadline.",
    )
    contextual = ConversationToneJudgeRequest(
        utterance=request,
        intended_tone=PilotEmotion.NEUTRAL,
        history_tones=(PastUserDelivery(history_index=0, intended_tone=PilotEmotion.ANGRY),),
    )
    rendered = tone_prompt(contextual)
    assert "angry" in rendered and "neutral" in rendered
    assert "hidden" not in rendered and "family" not in rendered
    with pytest.raises(ValueError, match="actual user history"):
        ConversationToneJudgeRequest(
            utterance=request,
            intended_tone=PilotEmotion.NEUTRAL,
            history_tones=(PastUserDelivery(history_index=1, intended_tone=PilotEmotion.ANGRY),),
        )
