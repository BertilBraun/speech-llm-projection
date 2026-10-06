"""Canonical completed and token-limited LLM responses."""

from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from speech_projector.models import GenerationKind as GenerationKind
from speech_projector.models import Record

INITIAL_TEACHER_TOKEN_CAP = 2048
RETRY_TEACHER_TOKEN_CAP = 4096


class CompletedGeneration(Record):
    kind: Literal[GenerationKind.COMPLETED] = GenerationKind.COMPLETED
    text: str
    token_ids: tuple[int, ...]


class TokenLimitedGeneration(Record):
    kind: Literal[GenerationKind.TOKEN_LIMIT] = GenerationKind.TOKEN_LIMIT
    partial_text: str
    token_ids: tuple[int, ...]


GenerationResult: TypeAlias = Annotated[
    CompletedGeneration | TokenLimitedGeneration, Field(discriminator="kind")
]
