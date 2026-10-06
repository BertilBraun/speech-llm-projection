"""Outcome-independent delayed acoustic cue fixtures for final-only conversation checks."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field, model_validator

from speech_projector.models import Example, Record, Role, Split, Turn
from speech_projector.overnight_data import Cohort, NeuEmotionalExampleSource, SourceSidecar
from speech_projector.overnight_evaluation import build_emotion_pairs
from speech_projector.tts_pilot import PilotEmotion, TtsPilotClip


class FollowupAudio(Record):
    clip: TtsPilotClip
    feature_path: Path


class DelayedCueConfiguration(Record):
    scenarios: int = Field(default=3, ge=1)
    seed: int = 42
    split: Split = Split.TEST

    @model_validator(mode="after")
    def heldout_only(self) -> DelayedCueConfiguration:
        if self.split == Split.TRAIN:
            raise ValueError("Conversation fixtures require held-out examples")
        return self


class DelayedCueScenario(Record):
    base_id: str
    family_id: str
    initial: tuple[Example, Example]
    sources: tuple[NeuEmotionalExampleSource, NeuEmotionalExampleSource]
    followups: tuple[FollowupAudio, FollowupAudio, FollowupAudio]
    fixed_assistants: tuple[Turn, Turn, Turn]

    @model_validator(mode="after")
    def validate_same_words_and_neutral_followups(self) -> DelayedCueScenario:
        first, second = self.initial
        first_source, second_source = self.sources
        if (
            first.user_text != second.user_text
            or first.history != second.history
            or first.prompt != second.prompt
            or first.split != second.split
            or first.split == Split.TRAIN
        ):
            raise ValueError("Initial paired conversation cues need identical words/context/policy")
        if (
            first_source.example_id != first.example_id
            or second_source.example_id != second.example_id
            or first_source.base_id != self.base_id
            or second_source.base_id != self.base_id
            or first_source.family_id != self.family_id
            or second_source.family_id != self.family_id
            or first_source.emotion == second_source.emotion
        ):
            raise ValueError("Conversation cue source identities or intended contrast differ")
        if any(item.clip.case.emotion != PilotEmotion.NEUTRAL for item in self.followups):
            raise ValueError("Later conversation speech must be shared neutral recordings")
        if any(item.duration > 30 for item in self.initial) or any(
            item.clip.audio_seconds > 30 for item in self.followups
        ):
            raise ValueError("Conversation speech must fit the frozen encoder's 30-second window")
        if len({item.clip.case.case_id for item in self.followups}) != 3:
            raise ValueError("Conversation follows three distinct shared recordings")
        if any(item.role != Role.ASSISTANT for item in self.fixed_assistants):
            raise ValueError("Fixed conversation replies must be assistant text")
        return self


class DelayedCueFixtures(Record):
    configuration: DelayedCueConfiguration
    scenarios: tuple[DelayedCueScenario, ...]

    @model_validator(mode="after")
    def validate_coverage(self) -> DelayedCueFixtures:
        if len(self.scenarios) != self.configuration.scenarios:
            raise ValueError("Fixture scenario count does not match its configuration")
        if len({item.family_id for item in self.scenarios}) != len(self.scenarios):
            raise ValueError("Conversation fixtures require distinct held-out families")
        if any(item.initial[0].split != self.configuration.split for item in self.scenarios):
            raise ValueError("Fixture split differs from its selected held-out examples")
        return self


def build_delayed_cue_fixtures(
    examples: Sequence[Example],
    sources: Sequence[SourceSidecar],
    followups: tuple[FollowupAudio, FollowupAudio, FollowupAudio],
    configuration: DelayedCueConfiguration,
) -> DelayedCueFixtures:
    pairs = list(build_emotion_pairs(examples, sources, Cohort.NEU_EMOTIONAL, configuration.split))
    random.Random(configuration.seed).shuffle(pairs)
    selected = []
    families: set[str] = set()
    for pair in pairs:
        if pair.family_id in families:
            continue
        selected.append(pair)
        families.add(pair.family_id)
        if len(selected) == configuration.scenarios:
            break
    if len(selected) != configuration.scenarios:
        raise ValueError("Insufficient distinct held-out Neu families for conversation fixtures")
    by_id = {
        item.example_id: item for item in sources if isinstance(item, NeuEmotionalExampleSource)
    }
    fixed = (
        Turn(role=Role.ASSISTANT, text="I hear you. We can talk through this."),
        Turn(role=Role.ASSISTANT, text="I'll keep it brief."),
        Turn(role=Role.ASSISTANT, text="We can consider the next step together."),
    )
    return DelayedCueFixtures(
        configuration=configuration,
        scenarios=tuple(
            DelayedCueScenario(
                base_id=pair.base_id,
                family_id=pair.family_id,
                initial=(pair.first, pair.second),
                sources=(by_id[pair.first.example_id], by_id[pair.second.example_id]),
                followups=followups,
                fixed_assistants=fixed,
            )
            for pair in selected
        ),
    )
