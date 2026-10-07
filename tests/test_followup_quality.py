"""Blind rating aggregation preserves ties and separates tone gain from factual loss."""

from pathlib import Path

import pytest

from scripts.inventory_results import write_record
from speech_projector.followup_quality import (
    BlindCards,
    QualitySlotMapping,
    freeze_quality_ratings,
    summarize_ratings,
)
from speech_projector.overnight_data import Cohort
from speech_projector.response_quality import (
    BlindCase,
    CaseRating,
    RatingConfidence,
    ResponseRating,
)


def card(identifier: str) -> BlindCase:
    return BlindCase(
        example_id=identifier,
        cohort=Cohort.NEU_EMOTIONAL,
        base_id=identifier,
        family_id=identifier,
        intended_delivery="happy",
        user_text="The appointment moved to five.",
        history=(),
        response_a="That's good news; I'll meet you at three.",
        response_b="Okay.",
    )


def test_joint_tone_gain_and_grounding_loss_are_not_hidden_by_one_score(tmp_path: Path) -> None:
    cards = (card("a"), card("b"))
    ratings = (
        CaseRating(
            example_id="a",
            response_a=ResponseRating(
                tone_helpfulness=2, literal_grounding=0, reason="warm but invents time"
            ),
            response_b=ResponseRating(
                tone_helpfulness=1, literal_grounding=2, reason="generic and faithful"
            ),
            confidence=RatingConfidence.HIGH,
            ambiguity_note="",
        ),
        CaseRating(
            example_id="b",
            response_a=ResponseRating(tone_helpfulness=1, literal_grounding=2, reason="same"),
            response_b=ResponseRating(tone_helpfulness=1, literal_grounding=2, reason="same"),
            confidence=RatingConfidence.HIGH,
            ambiguity_note="",
        ),
    )
    mapping = tuple(
        QualitySlotMapping(example_id=row.example_id, source_a="speech", source_b="asr")
        for row in cards
    )
    result = summarize_ratings(cards, ratings, mapping, "speech")
    assert result.tone_better_grounding_worse == 1
    assert result.tone_better_grounding_not_worse == 0
    assert result.tone.half_credit_win_fraction.estimate == 0.75
    assert result.grounding.first_minus_second.estimate == -1
    cards_path, ratings_path, locked = (
        tmp_path / "cards.json",
        tmp_path / "ratings.jsonl",
        tmp_path / "locked.jsonl",
    )
    write_record(cards_path, BlindCards(seed=42, cases=cards))
    ratings_path.write_text(
        "".join(row.model_dump_json() + "\n" for row in ratings), encoding="utf-8"
    )
    freeze_quality_ratings(cards_path, ratings_path, locked)
    with pytest.raises(ValueError, match="cannot be overwritten"):
        freeze_quality_ratings(cards_path, ratings_path, locked)
    with pytest.raises(ValueError, match="cover every"):
        summarize_ratings(cards, ratings[:1], mapping, "speech")
