"""Same-word audio preferences and validation-only overnight sweep selection."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from pydantic import ConfigDict, Field, model_validator
from torch import Tensor

from speech_projector.evaluation import (
    EvaluationOutcome,
    LossObservation,
    match_feature_length,
    summarize_losses,
)
from speech_projector.inputs import SpeechInput
from speech_projector.journal import append_record, read_journal
from speech_projector.judge import (
    BootstrapInterval,
    PairedMetricObservation,
    paired_dialogue_bootstrap,
)
from speech_projector.models import (
    EvaluationCondition,
    EvaluationMetrics,
    Example,
    GenerationKind,
    Record,
    RunConfig,
    Split,
)
from speech_projector.overnight_data import (
    Cohort,
    NeuEmotionalExampleSource,
    QwenEmotionalExampleSource,
    SourceSidecar,
)
from speech_projector.teacher_evaluation import projector_digest

if TYPE_CHECKING:
    from speech_projector.llm import FrozenQwen
    from speech_projector.projectors import Projector


@dataclass(frozen=True)
class EmotionEvaluationPair:
    base_id: str
    family_id: str
    first: Example
    second: Example


class PairedEmotionLoss(Record):
    model_config = ConfigDict(allow_inf_nan=False)
    base_id: str
    family_id: str
    first_example_id: str
    second_example_id: str
    first_target_tokens: int = Field(gt=0)
    second_target_tokens: int = Field(gt=0)
    targets_identical: bool
    first_audio_first_target: float
    second_audio_first_target: float
    second_audio_second_target: float
    first_audio_second_target: float
    resized_second_audio_first_target: float
    resized_first_audio_second_target: float

    @property
    def matching_margin(self) -> float:
        return 0.5 * (
            self.second_audio_first_target
            - self.first_audio_first_target
            + self.first_audio_second_target
            - self.second_audio_second_target
        )

    @property
    def resized_matching_margin(self) -> float:
        return 0.5 * (
            self.resized_second_audio_first_target
            - self.first_audio_first_target
            + self.resized_first_audio_second_target
            - self.second_audio_second_target
        )


class PairedEmotionProvenance(Record):
    configuration: RunConfig
    projector_weights_sha256: str
    pairs_sha256: str


class PairPreferenceSummary(Record):
    pairs: int = Field(gt=0)
    distinct_target_pairs: int = Field(ge=0)
    family_clusters: int = Field(ge=2)
    matching_margin: BootstrapInterval
    resized_matching_margin: BootstrapInterval
    matching_win_rate: BootstrapInterval
    tie_rate: float = Field(ge=0, le=1)
    first_direction_win_rate: float = Field(ge=0, le=1)
    second_direction_win_rate: float = Field(ge=0, le=1)
    numerical_tie_tolerance: float = Field(ge=0)


class ValidationCohorts(Record):
    old_ordinary: EvaluationMetrics
    old_emotional: EvaluationMetrics
    new_neu_emotional: EvaluationMetrics

    @model_validator(mode="after")
    def validate_cohort_coverage(self) -> ValidationCohorts:
        for item in (self.old_ordinary, self.old_emotional, self.new_neu_emotional):
            if item.examples < 1 or item.target_tokens < 1 or not math.isfinite(item.cross_entropy):
                raise ValueError("Each validation cohort needs nonempty finite scored coverage")
        return self

    @property
    def macro_cross_entropy(self) -> float:
        return (
            self.old_ordinary.cross_entropy
            + self.old_emotional.cross_entropy
            + self.new_neu_emotional.cross_entropy
        ) / 3


class SweepCandidate(Record):
    configuration: RunConfig
    validation: ValidationCohorts
    old_emotional_preference: PairPreferenceSummary
    new_neu_preference: PairPreferenceSummary
    pseudo_tokens_per_second: float = Field(gt=0)
    training_seconds_per_update: float = Field(gt=0)

    @property
    def robust_neu_margin(self) -> float:
        return min(
            self.new_neu_preference.matching_margin.estimate,
            self.new_neu_preference.resized_matching_margin.estimate,
        )

    @model_validator(mode="after")
    def validate_finite_metrics(self) -> SweepCandidate:
        values = (
            self.validation.old_ordinary.cross_entropy,
            self.validation.old_emotional.cross_entropy,
            self.validation.new_neu_emotional.cross_entropy,
            self.new_neu_preference.matching_margin.estimate,
            self.new_neu_preference.resized_matching_margin.estimate,
            self.pseudo_tokens_per_second,
            self.training_seconds_per_update,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Sweep selection requires finite validation and timing metrics")
        return self


class SweepSelectionPolicy(Record):
    ordinary_ce_tolerance: float = Field(default=0.10, ge=0)
    macro_ce_tie_band: float = Field(default=0.03, ge=0)
    compact_macro_ce_tolerance: float = Field(default=0.08, ge=0)
    minimum_token_rate_reduction: float = Field(default=2.0, gt=1)


class SweepDecision(Record):
    policy: SweepSelectionPolicy
    candidates: tuple[SweepCandidate, ...]
    quality_leader: str
    compact_winners: tuple[str, ...]
    ordinary_guard_eligible: tuple[str, ...]
    rationale: str


class FixedValidationConfig(Record):
    seed: int = 42
    ordinary_examples: int = Field(default=48, gt=0)
    pairs_per_emotional_cohort: int = Field(default=20, ge=2)
    ordinary_generations: int = Field(default=8, gt=0)
    generated_pairs_per_cohort: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def validate_generation_coverage(self) -> FixedValidationConfig:
        if self.ordinary_generations > self.ordinary_examples:
            raise ValueError("Ordinary generations exceed selected validation examples")
        if self.generated_pairs_per_cohort > self.pairs_per_emotional_cohort:
            raise ValueError("Generated emotion pairs exceed selected validation pairs")
        return self


class FixedValidationSelection(Record):
    configuration: FixedValidationConfig
    examples: tuple[Example, ...]
    sources: tuple[SourceSidecar, ...]
    generation_example_ids: tuple[str, ...]


def build_emotion_pairs(
    examples: Sequence[Example],
    sources: Sequence[SourceSidecar],
    cohort: Cohort,
    split: Split,
) -> tuple[EmotionEvaluationPair, ...]:
    if cohort == Cohort.ORDINARY:
        raise ValueError("Ordinary examples have no same-text emotion pairs")
    examples_by_id = {item.example_id: item for item in examples}
    groups: defaultdict[
        str, list[tuple[Example, QwenEmotionalExampleSource | NeuEmotionalExampleSource]]
    ]
    groups = defaultdict(list)
    for source in sources:
        if source.cohort != cohort:
            continue
        example = examples_by_id[source.example_id]
        if example.split != split:
            continue
        match source:
            case QwenEmotionalExampleSource() | NeuEmotionalExampleSource():
                groups[source.base_id].append((example, source))
    pairs: list[EmotionEvaluationPair] = []
    for base_id, group in sorted(groups.items()):
        if len(group) != 2:
            raise ValueError(f"Emotion base {base_id} does not have exactly two variants")
        ordered = sorted(group, key=lambda item: item[1].emotion.value)
        (first, first_source), (second, second_source) = ordered
        if (
            first_source.family_id != second_source.family_id
            or first_source.emotion.value == second_source.emotion.value
        ):
            raise ValueError("Emotion pair family/contrasting-delivery identity differs")
        pairs.append(EmotionEvaluationPair(base_id, first_source.family_id, first, second))
    validate_pairs(pairs)
    return tuple(pairs)


def _ordinary_subset(examples: Sequence[Example], count: int, seed: int) -> tuple[Example, ...]:
    if len(examples) < count:
        raise ValueError("Insufficient ordinary held-out examples for the fixed selection")
    empty = sorted(
        (item for item in examples if not item.history), key=lambda item: item.example_id
    )
    present = sorted((item for item in examples if item.history), key=lambda item: item.example_id)
    generator = random.Random(seed)
    generator.shuffle(empty)
    generator.shuffle(present)
    empty_count = min((count + 1) // 2, len(empty))
    present_count = min(count // 2, len(present))
    selected = empty[:empty_count] + present[:present_count]
    remaining = empty[empty_count:] + present[present_count:]
    generator.shuffle(remaining)
    selected.extend(remaining[: count - len(selected)])
    generator.shuffle(selected)
    return tuple(selected)


def _balanced_pairs(
    pairs: Sequence[EmotionEvaluationPair],
    sources: Mapping[str, SourceSidecar],
    count: int,
    seed: int,
) -> tuple[EmotionEvaluationPair, ...]:
    if len(pairs) < count:
        raise ValueError("Insufficient complete emotion pairs for the fixed selection")
    groups: defaultdict[tuple[str, str], list[EmotionEvaluationPair]] = defaultdict(list)
    for pair in pairs:
        first_source = sources[pair.first.example_id]
        second_source = sources[pair.second.example_id]
        match first_source, second_source:
            case (QwenEmotionalExampleSource(), QwenEmotionalExampleSource()) | (
                NeuEmotionalExampleSource(),
                NeuEmotionalExampleSource(),
            ):
                groups[(first_source.emotion.value, second_source.emotion.value)].append(pair)
            case _:
                raise ValueError("Emotion pair source records do not share a cohort")
    generator = random.Random(seed)
    for group in groups.values():
        generator.shuffle(group)
    selected: list[EmotionEvaluationPair] = []
    families: set[str] = set()
    while len(selected) < count:
        for contrast in sorted(groups):
            group = groups[contrast]
            if not group or len(selected) == count:
                continue
            index = next(
                (index for index, pair in enumerate(group) if pair.family_id not in families), 0
            )
            pair = group.pop(index)
            selected.append(pair)
            families.add(pair.family_id)
    return tuple(selected)


def select_fixed_validation(
    examples: Sequence[Example],
    sources: Sequence[SourceSidecar],
    configuration: FixedValidationConfig,
) -> FixedValidationSelection:
    examples_by_id = {item.example_id: item for item in examples}
    sources_by_id = {item.example_id: item for item in sources}
    if (
        len(examples_by_id) != len(examples)
        or len(sources_by_id) != len(sources)
        or set(examples_by_id) != set(sources_by_id)
    ):
        raise ValueError("Combined examples and sources require exact unique identity coverage")
    ordinary = _ordinary_subset(
        tuple(
            item
            for item in examples
            if item.split == Split.VALIDATION
            and sources_by_id[item.example_id].cohort == Cohort.ORDINARY
        ),
        configuration.ordinary_examples,
        configuration.seed,
    )
    ordinary_generations = _ordinary_subset(
        ordinary, configuration.ordinary_generations, configuration.seed + 1
    )
    emotional = tuple(
        _balanced_pairs(
            build_emotion_pairs(examples, sources, cohort, Split.VALIDATION),
            sources_by_id,
            configuration.pairs_per_emotional_cohort,
            configuration.seed,
        )
        for cohort in (Cohort.QWEN_EMOTIONAL, Cohort.NEU_EMOTIONAL)
    )
    generated = ordinary_generations + tuple(
        example
        for cohort_pairs in emotional
        for pair in cohort_pairs[: configuration.generated_pairs_per_cohort]
        for example in (pair.first, pair.second)
    )
    generated_ids = {item.example_id for item in generated}
    all_selected = ordinary + tuple(
        example
        for cohort_pairs in emotional
        for pair in cohort_pairs
        for example in (pair.first, pair.second)
    )
    ordered = generated + tuple(
        item for item in all_selected if item.example_id not in generated_ids
    )
    return FixedValidationSelection(
        configuration=configuration,
        examples=ordered,
        sources=tuple(sources_by_id[item.example_id] for item in ordered),
        generation_example_ids=tuple(item.example_id for item in generated),
    )


def select_sweep(
    candidates: Sequence[SweepCandidate], policy: SweepSelectionPolicy
) -> SweepDecision:
    if not candidates or len({item.configuration.name for item in candidates}) != len(candidates):
        raise ValueError("Sweep candidates must be nonempty and have unique run names")
    ordinary_best = min(item.validation.old_ordinary.cross_entropy for item in candidates)
    eligible = tuple(
        item
        for item in candidates
        if item.validation.old_ordinary.cross_entropy
        <= ordinary_best + policy.ordinary_ce_tolerance
    )
    macro_best = min(item.validation.macro_cross_entropy for item in eligible)
    tied = tuple(
        item
        for item in eligible
        if item.validation.macro_cross_entropy <= macro_best + policy.macro_ce_tie_band
    )
    leader = min(
        tied,
        key=lambda item: (
            -item.robust_neu_margin,
            item.training_seconds_per_update,
            item.validation.macro_cross_entropy,
            item.configuration.name,
        ),
    )
    compact = tuple(
        item
        for item in eligible
        if item.configuration.name != leader.configuration.name
        and item.validation.macro_cross_entropy
        <= leader.validation.macro_cross_entropy + policy.compact_macro_ce_tolerance
        and item.pseudo_tokens_per_second * policy.minimum_token_rate_reduction
        <= leader.pseudo_tokens_per_second
    )
    compact_winners = ()
    if compact:
        winner = min(
            compact,
            key=lambda item: (
                item.pseudo_tokens_per_second,
                item.validation.macro_cross_entropy,
                -item.robust_neu_margin,
                item.training_seconds_per_update,
                item.configuration.name,
            ),
        )
        compact_winners = (winner.configuration.name,)
    return SweepDecision(
        policy=policy,
        candidates=tuple(candidates),
        quality_leader=leader.configuration.name,
        compact_winners=compact_winners,
        ordinary_guard_eligible=tuple(item.configuration.name for item in eligible),
        rationale=(
            "Validation only: ordinary-content CE guard, equal-cohort macro CE band, "
            "then the minimum of raw and length-resized new Neu same-word audio matching "
            "margins, followed by training speed. No positive-margin acceptance gate. "
            "Compact selection additionally requires the precommitted macro CE tolerance "
            "and token-rate reduction. Old emotional margins are reported separately. "
            "Root reviews actual outputs before accepting this automatic recommendation; "
            "no test metric enters selection."
        ),
    )


def summarize_cohort_validation(
    outcome: EvaluationOutcome,
    sources: Sequence[SourceSidecar],
    condition: EvaluationCondition,
) -> ValidationCohorts:
    """Aggregate stored records; cohort clocks remain unmeasured, whole-run clocks stay intact."""
    sources_by_id = {item.example_id: item for item in sources}
    if len(sources_by_id) != len(sources):
        raise ValueError("Cohort summaries require unique source example IDs")
    primary_losses = tuple(item for item in outcome.example_losses if item.condition == condition)
    if len({item.example_id for item in primary_losses}) != len(primary_losses) or {
        item.example_id for item in primary_losses
    } != set(sources_by_id):
        raise ValueError("Primary scored examples must exactly cover cohort source records")
    metrics: list[EvaluationMetrics] = []
    for cohort in (Cohort.ORDINARY, Cohort.QWEN_EMOTIONAL, Cohort.NEU_EMOTIONAL):
        selected = tuple(
            item for item in primary_losses if sources_by_id[item.example_id].cohort == cohort
        )
        cross_entropy, perplexity, tokens = summarize_losses(
            tuple(
                LossObservation(item.cross_entropy * item.target_tokens, item.target_tokens)
                for item in selected
            )
        )
        generations = tuple(
            item
            for item in outcome.samples
            if item.condition == condition and sources_by_id[item.example_id].cohort == cohort
        )
        similarities = tuple(
            item.semantic_similarity for item in generations if item.semantic_similarity is not None
        )
        metadata = tuple(item.generation for item in generations)
        completion_known = all(item is not None for item in metadata)
        metrics.append(
            EvaluationMetrics(
                examples=len(selected),
                target_tokens=tokens,
                cross_entropy=cross_entropy,
                perplexity=perplexity,
                semantic_similarity=sum(similarities) / len(similarities) if similarities else None,
                generated_examples=len(generations),
                generated_tokens=sum(len(item.token_ids) for item in metadata if item is not None),
                completed_generations=sum(
                    item.kind == GenerationKind.COMPLETED for item in metadata if item is not None
                )
                if completion_known
                else None,
                token_limited_generations=sum(
                    item.kind == GenerationKind.TOKEN_LIMIT for item in metadata if item is not None
                )
                if completion_known
                else None,
            )
        )
    return ValidationCohorts(
        old_ordinary=metrics[0], old_emotional=metrics[1], new_neu_emotional=metrics[2]
    )


def validate_pairs(pairs: Sequence[EmotionEvaluationPair]) -> None:
    if not pairs or len({item.base_id for item in pairs}) != len(pairs):
        raise ValueError("Emotion evaluation requires nonempty unique base pairs")
    identifiers = tuple(
        example.example_id for pair in pairs for example in (pair.first, pair.second)
    )
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Emotion pairs must contain unique audio example IDs")
    if len({item.first.split for item in pairs}) != 1:
        raise ValueError("One paired evaluation must use one held-out split")
    for pair in pairs:
        if pair.first.split == Split.TRAIN or pair.first.split != pair.second.split:
            raise ValueError("Emotion evaluation requires pairs within one held-out split")
        if (
            pair.first.user_text != pair.second.user_text
            or pair.first.history != pair.second.history
        ):
            raise ValueError(
                "Paired tone tests require identical literal words and textual history"
            )
        if pair.first.prompt != pair.second.prompt:
            raise ValueError("Paired tone tests require identical prompt policies")


@torch.no_grad()
def evaluate_emotion_pairs(
    wrapper: FrozenQwen,
    projector: Projector,
    pairs: Sequence[EmotionEvaluationPair],
    output_directory: Path,
) -> tuple[PairedEmotionLoss, ...]:
    validate_pairs(pairs)
    wrapper.model.eval()
    projector.eval()
    provenance = PairedEmotionProvenance(
        configuration=wrapper.config,
        projector_weights_sha256=projector_digest(projector),
        pairs_sha256=hashlib.sha256(
            "".join(
                pair.base_id
                + "\n"
                + pair.family_id
                + "\n"
                + pair.first.model_dump_json()
                + "\n"
                + pair.second.model_dump_json()
                + "\n"
                for pair in pairs
            ).encode("utf-8")
        ).hexdigest(),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    provenance_path = output_directory / "paired_emotion_provenance.json"
    journal = output_directory / "paired_emotion_losses.jsonl"
    if provenance_path.exists():
        if PairedEmotionProvenance.model_validate_json(provenance_path.read_bytes()) != provenance:
            raise ValueError("Paired emotion journal differs in config, weights or targets")
    else:
        if journal.exists():
            raise ValueError("Paired emotion journal has no matching provenance")
        partial = provenance_path.with_suffix(".json.part")
        partial.write_text(provenance.model_dump_json(indent=2) + "\n", encoding="utf-8")
        partial.replace(provenance_path)
    observations = list(read_journal(journal, PairedEmotionLoss))
    saved = {item.base_id for item in observations}
    if len(saved) != len(observations) or not saved.issubset({item.base_id for item in pairs}):
        raise ValueError("Paired emotion journal has duplicated or out-of-subset bases")
    for pair in pairs:
        if pair.base_id in saved:
            continue
        first: Tensor = torch.load(pair.first.feature_path, map_location="cpu", weights_only=True)
        second: Tensor = torch.load(pair.second.feature_path, map_location="cpu", weights_only=True)
        first = first.to(wrapper.device)
        second = second.to(wrapper.device)
        first_embeddings = projector(first)
        second_embeddings = projector(second)
        observation = PairedEmotionLoss(
            base_id=pair.base_id,
            family_id=pair.family_id,
            first_example_id=pair.first.example_id,
            second_example_id=pair.second.example_id,
            first_target_tokens=wrapper.target_token_count(pair.first),
            second_target_tokens=wrapper.target_token_count(pair.second),
            targets_identical=pair.first.target_text == pair.second.target_text,
            first_audio_first_target=float(wrapper.loss(pair.first, SpeechInput(first_embeddings))),
            second_audio_first_target=float(
                wrapper.loss(pair.first, SpeechInput(second_embeddings))
            ),
            second_audio_second_target=float(
                wrapper.loss(pair.second, SpeechInput(second_embeddings))
            ),
            first_audio_second_target=float(
                wrapper.loss(pair.second, SpeechInput(first_embeddings))
            ),
            resized_second_audio_first_target=float(
                wrapper.loss(
                    pair.first, SpeechInput(projector(match_feature_length(second, first.shape[0])))
                )
            ),
            resized_first_audio_second_target=float(
                wrapper.loss(
                    pair.second,
                    SpeechInput(projector(match_feature_length(first, second.shape[0]))),
                )
            ),
        )
        append_record(journal, observation)
        observations.append(observation)
    return tuple(observations)


def summarize_preferences(
    observations: Sequence[PairedEmotionLoss], numerical_tie_tolerance: float = 1e-6
) -> PairPreferenceSummary:
    if not observations or len({item.base_id for item in observations}) != len(observations):
        raise ValueError("Preference summaries require nonempty unique paired bases")
    if numerical_tie_tolerance < 0:
        raise ValueError("Numerical tie tolerance must be nonnegative")
    natural = tuple(
        PairedMetricObservation(
            example_id=item.base_id, dialogue_id=item.family_id, difference=item.matching_margin
        )
        for item in observations
    )
    resized = tuple(
        PairedMetricObservation(
            example_id=item.base_id,
            dialogue_id=item.family_id,
            difference=item.resized_matching_margin,
        )
        for item in observations
    )
    wins = tuple(
        PairedMetricObservation(
            example_id=item.base_id,
            dialogue_id=item.family_id,
            difference=float(item.matching_margin > numerical_tie_tolerance),
        )
        for item in observations
    )
    return PairPreferenceSummary(
        pairs=len(observations),
        distinct_target_pairs=sum(not item.targets_identical for item in observations),
        family_clusters=len({item.family_id for item in observations}),
        matching_margin=paired_dialogue_bootstrap(natural),
        resized_matching_margin=paired_dialogue_bootstrap(resized),
        matching_win_rate=paired_dialogue_bootstrap(wins),
        tie_rate=sum(abs(item.matching_margin) <= numerical_tie_tolerance for item in observations)
        / len(observations),
        first_direction_win_rate=sum(
            item.second_audio_first_target - item.first_audio_first_target > numerical_tie_tolerance
            for item in observations
        )
        / len(observations),
        second_direction_win_rate=sum(
            item.first_audio_second_target - item.second_audio_second_target
            > numerical_tie_tolerance
            for item in observations
        )
        / len(observations),
        numerical_tie_tolerance=numerical_tie_tolerance,
    )
