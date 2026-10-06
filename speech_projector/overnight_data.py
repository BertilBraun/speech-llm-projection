"""Canonical cohort and paired-source identities for the combined experiment."""

from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from speech_projector.emotion_preview import Delivery
from speech_projector.models import Record
from speech_projector.tts_pilot import PilotEmotion


class Cohort(str, Enum):
    ORDINARY = "ordinary"
    QWEN_EMOTIONAL = "qwen_emotional"
    NEU_EMOTIONAL = "neu_emotional"


class ExampleSource(Record):
    example_id: str
    source_manifest: Path
    source_example_id: str


class OrdinaryExampleSource(ExampleSource):
    cohort: Literal[Cohort.ORDINARY] = Cohort.ORDINARY


class EmotionalExampleSource(ExampleSource):
    base_id: str
    family_id: str


class QwenEmotionalExampleSource(EmotionalExampleSource):
    cohort: Literal[Cohort.QWEN_EMOTIONAL] = Cohort.QWEN_EMOTIONAL
    emotion: Delivery


class NeuEmotionalExampleSource(EmotionalExampleSource):
    cohort: Literal[Cohort.NEU_EMOTIONAL] = Cohort.NEU_EMOTIONAL
    emotion: PilotEmotion


SourceSidecar: TypeAlias = Annotated[
    OrdinaryExampleSource | QwenEmotionalExampleSource | NeuEmotionalExampleSource,
    Field(discriminator="cohort"),
]
