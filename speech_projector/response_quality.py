"""Blinded model-assisted response ratings, separate from teacher-response fidelity."""

from enum import Enum

from pydantic import Field

from speech_projector.models import EvaluationCondition, Record
from speech_projector.overnight_data import Cohort


class BlindCase(Record):
    example_id: str
    cohort: Cohort
    base_id: str
    family_id: str
    intended_delivery: str
    user_text: str
    history: tuple[tuple[str, str], ...]
    response_a: str
    response_b: str


class SlotMapping(Record):
    example_id: str
    condition_a: EvaluationCondition
    condition_b: EvaluationCondition


class RatingConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResponseRating(Record):
    tone_helpfulness: int = Field(ge=0, le=2)
    literal_grounding: int = Field(ge=0, le=2)
    reason: str = Field(min_length=1)


class CaseRating(Record):
    example_id: str
    response_a: ResponseRating
    response_b: ResponseRating
    confidence: RatingConfidence
    ambiguity_note: str
