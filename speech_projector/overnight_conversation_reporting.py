"""Readable four-turn evidence and blinded judge inputs from actual saved replies."""

from collections.abc import Sequence

from speech_projector.judge import JudgeRequest
from speech_projector.overnight_conversation import DelayedCueFixtures
from speech_projector.overnight_conversation_execution import (
    ControlledConversationReply,
    RolloutConversationReply,
    SavedConversationReply,
    reply_identifier,
    response_text,
)
from speech_projector.overnight_judge import ConversationToneJudgeRequest, PastUserDelivery
from speech_projector.tts_pilot import PilotEmotion


def conversation_judge_requests(
    replies: Sequence[SavedConversationReply],
    fixtures: DelayedCueFixtures,
    *,
    include_rollout_next_step: bool = False,
) -> tuple[ConversationToneJudgeRequest, ...]:
    scenarios = {item.base_id: item for item in fixtures.scenarios}
    requests: list[ConversationToneJudgeRequest] = []
    for reply in replies:
        match reply:
            case ControlledConversationReply():
                pass
            case RolloutConversationReply():
                if not include_rollout_next_step or reply.turn_index != 2:
                    continue
        scenario = scenarios[reply.scenario_base_id]
        branch = next(
            index
            for index, item in enumerate(scenario.initial)
            if item.example_id == reply.initial_example_id
        )
        requests.append(
            ConversationToneJudgeRequest(
                utterance=JudgeRequest(
                    example_id=reply_identifier(reply),
                    dialogue_id=scenario.family_id,
                    history=reply.history_for_judge,
                    transcript=scenario.followups[reply.turn_index - 1].clip.case.text,
                    candidate_response=response_text(reply.generation),
                ),
                intended_tone=PilotEmotion.NEUTRAL,
                history_tones=(
                    PastUserDelivery(
                        history_index=0, intended_tone=scenario.sources[branch].emotion
                    ),
                ),
            )
        )
    if len({item.utterance.example_id for item in requests}) != len(requests):
        raise ValueError("Conversation judge inputs have duplicate recorded reply IDs")
    return tuple(requests)


def render_conversations(
    replies: Sequence[SavedConversationReply], fixtures: DelayedCueFixtures
) -> str:
    scenarios = {item.base_id: item for item in fixtures.scenarios}
    lines = [
        "# Four-turn acoustic-history evidence",
        "",
        "Controlled replies use the same fixed assistant messages across paired initial tones. "
        "Retained-audio, words-only prior history, and omitted initial cue are separate inputs. "
        "Practical rollouts reuse their own generated assistant text, which can leak tone. "
        "No tone annotations or current transcripts were supplied to speech inference.",
        "",
        "The quality judge sees the same full true textual history and intended initial delivery "
        "for all controlled conditions, including the deliberately deprived omitted-cue input. "
        "Later user delivery is neutral. These few synthetic families do not establish general "
        "conversation or human emotion recognition; intended annotation is not listening proof.",
        "",
    ]
    for reply in replies:
        scenario = scenarios[reply.scenario_base_id]
        branch = next(
            index
            for index, item in enumerate(scenario.initial)
            if item.example_id == reply.initial_example_id
        )
        match reply:
            case ControlledConversationReply():
                condition = "controlled / " + reply.history_mode.value
            case RolloutConversationReply():
                condition = "practical retained-audio rollout"
        current = (
            scenario.initial[branch].user_text
            if reply.turn_index == 0
            else scenario.followups[reply.turn_index - 1].clip.case.text
        )
        lines.extend(
            [
                f"## {reply.initial_example_id} / turn {reply.turn_index + 1} / {condition}",
                "",
                f"Initial intended delivery: {scenario.sources[branch].emotion.value}.",
                "",
                "True prior conversation:",
                "",
                *(f"- {turn.role.value}: {turn.text}" for turn in reply.history_for_judge),
                "",
                f"Current user words: {current}",
                "",
                "> " + response_text(reply.generation).replace("\n", "\n> "),
                "",
                f"Actual termination: {reply.generation.kind.value}; "
                f"generation wall time {reply.generation_seconds:.3f}s.",
                "",
            ]
        )
    return "\n".join(lines)
