"""Literal screening flags novel clock values without demanding transcript copying."""

from speech_projector.content_review import LiteralKind, review_content
from speech_projector.models import EvaluationCondition, SampleGeneration
from speech_projector.overnight_data import Cohort


def sample(response: str) -> SampleGeneration:
    return SampleGeneration(
        example_id="case",
        dialogue_id="dialogue",
        condition=EvaluationCondition.SPEECH,
        history=(),
        user_transcript="I'll send it at five PM.",
        gold_response="unused",
        generated_response=response,
        duration=5,
    )


def test_time_reversal_is_a_review_flag_but_formatting_and_generic_replies_are_not() -> None:
    reversal = review_content(
        "speech",
        sample("I'll send it at three p.m."),
        "I'll send it at five PM.",
        "",
        Cohort.NEU_EMOTIONAL,
    )
    assert [(row.kind, row.normalized) for row in reversal.novel_response_literals] == [
        (LiteralKind.CLOCK, "15:00")
    ]
    assert reversal.first_person_action_overlap == ("send",)
    formatted = review_content(
        "speech",
        sample("Five PM, understood."),
        "I'll send it at 5:00 p.m.",
        "",
        Cohort.NEU_EMOTIONAL,
    )
    assert formatted.novel_response_literals == ()
    generic = review_content(
        "speech", sample("Understood."), "I'll send it at five PM.", "", Cohort.NEU_EMOTIONAL
    )
    assert generic.novel_response_literals == ()
    assert generic.first_person_action_overlap == ()
