"""Canonical alternatives for the current utterance supplied to the language model."""

from dataclasses import dataclass
from typing import TypeAlias

from torch import Tensor


@dataclass(frozen=True)
class SpeechInput:
    embeddings: Tensor


@dataclass(frozen=True)
class TranscriptInput:
    text: str


UtteranceInput: TypeAlias = SpeechInput | TranscriptInput
