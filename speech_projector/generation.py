"""Canonical completed and token-limited LLM responses."""

from enum import Enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from speech_projector.models import Record


class GenerationKind(str, Enum):
    COMPLETED = "completed"
    TOKEN_LIMIT = "token_limit"


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
