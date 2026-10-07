"""Prepare immutable source-blinded cards and aggregate locked paired ratings."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import mean

from pydantic import Field

from scripts.inventory_results import write_record
from speech_projector.followup_evaluation import bind_provenance, file_artifact
from speech_projector.judge import (
    BootstrapInterval,
    PairedMetricObservation,
    paired_dialogue_bootstrap,
)
from speech_projector.models import EvaluationCondition, FileArtifact, Record, SampleGeneration
from speech_projector.overnight_data import (
    Cohort,
    NeuEmotionalExampleSource,
    QwenEmotionalExampleSource,
)
from speech_projector.overnight_launcher import FinalEvaluationSelection, load_sources
from speech_projector.overnight_preparation import load_records
from speech_projector.response_quality import BlindCase, CaseRating, ResponseRating


class QualityResponseSource(Record):
    name: str = Field(min_length=1)
    generations: Path
    condition: EvaluationCondition


class QualityComparisonConfig(Record):
    selection: Path
    sources: Path
    first: QualityResponseSource
    second: QualityResponseSource
    cohorts: tuple[Cohort, ...]
    output_directory: Path
    seed: int = 42


class QualitySlotMapping(Record):
    example_id: str
    source_a: str
    source_b: str


class BlindCards(Record):
    seed: int
    cases: tuple[BlindCase, ...]


class QualityComparisonProvenance(Record):
    configuration: QualityComparisonConfig
    artifacts: tuple[FileArtifact, ...]


class QualityAxisComparison(Record):
    first_mean: float
    second_mean: float
    first_minus_second: BootstrapInterval
    wins: int
    ties: int
    losses: int
    half_credit_win_fraction: BootstrapInterval


class QualityComparisonSummary(Record):
    examples: int
    bases: int
    families: int
    tone: QualityAxisComparison
    grounding: QualityAxisComparison
    tone_better_grounding_not_worse: int
    tone_better_grounding_worse: int
    ambiguous_cases: int


class CohortQualityComparison(Record):
    cohort: Cohort
    comparison: QualityComparisonSummary


class QualityScoredReport(Record):
    provenance: QualityComparisonProvenance
    ratings: FileArtifact
    overall: QualityComparisonSummary
    cohorts: tuple[CohortQualityComparison, ...]


def render_cards(cases: Sequence[BlindCase]) -> str:
    lines = [
        "# Blinded response comparison",
        "",
        "Intended labels describe synthetic delivery settings; they have not been validated "
        "by human listening. Rate tone helpfulness and literal grounding separately, 0–2. "
        "No reference response, ASR transcript or condition mapping is shown.",
        "",
        "Tone: 0 incompatible/dismissive/unhelpful; 1 appropriate generic; "
        "2 specifically tone-sensitive and helpful without unsupported mental diagnosis. "
        "Grounding: 0 materially wrong/misses request; 1 partial/generic/minor unsupported; "
        "2 faithful and useful. A factual reversal penalizes grounding, not automatically tone.",
        "",
    ]
    for index, case in enumerate(cases):
        lines.extend(
            [
                f"## {index}: {case.example_id}",
                "",
                f"Cohort {case.cohort.value}; family {case.family_id}; "
                f"intended {case.intended_delivery}",
                "",
                f"USER: {case.user_text}",
                "",
            ]
        )
        lines.extend(f"History {role}: {text}" for role, text in case.history)
        lines.extend(["", "A:", "", case.response_a, "", "B:", "", case.response_b, ""])
    return "\n".join(lines)


def choose_root_cards(cases: Sequence[BlindCase], seed: int) -> tuple[BlindCase, ...]:
    pairs: dict[tuple[Cohort, str], list[BlindCase]] = defaultdict(list)
    for case in cases:
        pairs[(case.cohort, case.base_id)].append(case)
    if any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError("Quality cards require complete same-word delivery pairs")
    generator = random.Random(seed)
    old = [pair for pair in pairs.values() if pair[0].cohort == Cohort.QWEN_EMOTIONAL]
    generator.shuffle(old)
    selected: list[BlindCase] = []
    families: set[str] = set()
    for pair in old:
        if pair[0].family_id not in families:
            selected.extend(pair)
            families.add(pair[0].family_id)
        if len(selected) == 8:
            break
    if old and len(selected) != 8:
        raise ValueError("Root audit needs four old family-distinct pairs")
    pairs_per_contrast = 2 if old else 3
    for contrast in (("happy", "sad"), ("fearful", "happy"), ("angry", "happy"), ("angry", "sad")):
        matching = [
            pair
            for pair in pairs.values()
            if pair[0].cohort == Cohort.NEU_EMOTIONAL
            and tuple(sorted(row.intended_delivery for row in pair)) == contrast
        ]
        generator.shuffle(matching)
        chosen: list[list[BlindCase]] = []
        for pair in matching:
            if pair[0].family_id not in {item[0].family_id for item in chosen}:
                chosen.append(pair)
            if len(chosen) == pairs_per_contrast:
                break
        if len(chosen) != pairs_per_contrast:
            raise ValueError(f"Root audit lacks family-diverse pairs for {contrast}")
        selected.extend(row for pair in chosen for row in pair)
    assert len(selected) == 24
    return tuple(selected)


def prepare_quality_comparison(configuration: QualityComparisonConfig) -> BlindCards:
    if Cohort.ORDINARY in configuration.cohorts:
        raise ValueError("Emotional rating cards require genuine emotional source records")
    if configuration.first.name == configuration.second.name:
        raise ValueError("Comparison sources must have distinct names")
    selection = FinalEvaluationSelection.model_validate_json(configuration.selection.read_bytes())
    sources = {row.example_id: row for row in load_sources(configuration.sources)}
    examples = {row.example_id: row for row in selection.examples}
    first = {
        row.example_id: row
        for row in load_records(configuration.first.generations, SampleGeneration)
        if row.condition == configuration.first.condition
    }
    second = {
        row.example_id: row
        for row in load_records(configuration.second.generations, SampleGeneration)
        if row.condition == configuration.second.condition
    }
    cards: list[BlindCase] = []
    mapping: list[QualitySlotMapping] = []
    for identity in selection.generation_example_ids:
        source = sources[identity]
        if source.cohort not in configuration.cohorts:
            continue
        if identity not in first or identity not in second:
            raise ValueError(f"Comparison missing fixed response {identity}")
        example = examples[identity]
        if (
            first[identity].gold_response != example.target_text
            or second[identity].gold_response != example.target_text
        ):
            raise ValueError("Response sources refer to different target examples")
        match source:
            case QwenEmotionalExampleSource() | NeuEmotionalExampleSource():
                base, family, delivery = source.base_id, source.family_id, source.emotion.value
            case _:
                raise ValueError("Selected rating source is not emotional")
        flipped = bool(hashlib.sha256(f"{configuration.seed}:{identity}".encode()).digest()[0] % 2)
        response_a, response_b = (
            (second[identity], first[identity]) if flipped else (first[identity], second[identity])
        )
        cards.append(
            BlindCase(
                example_id=identity,
                cohort=source.cohort,
                base_id=base,
                family_id=family,
                intended_delivery=delivery,
                user_text=example.user_text,
                history=tuple((row.role.value, row.text) for row in example.history),
                response_a=response_a.generated_response,
                response_b=response_b.generated_response,
            )
        )
        mapping.append(
            QualitySlotMapping(
                example_id=identity,
                source_a=configuration.second.name if flipped else configuration.first.name,
                source_b=configuration.first.name if flipped else configuration.second.name,
            )
        )
    provenance = QualityComparisonProvenance(
        configuration=configuration,
        artifacts=tuple(
            file_artifact(path)
            for path in (
                configuration.selection,
                configuration.sources,
                configuration.first.generations,
                configuration.second.generations,
            )
        ),
    )
    directory = configuration.output_directory
    bind_provenance(directory / "provenance.json", provenance)
    full = BlindCards(seed=configuration.seed, cases=tuple(cards))
    subset = BlindCards(seed=configuration.seed, cases=choose_root_cards(cards, configuration.seed))
    write_record(directory / "blind_cases.json", full)
    write_record(directory / "root_blind24.json", subset)
    (directory / "blind_cases.md").write_text(render_cards(cards), encoding="utf-8")
    (directory / "root_blind24.md").write_text(render_cards(subset.cases), encoding="utf-8")
    (directory / "slot_mapping.jsonl").write_text(
        "".join(row.model_dump_json() + "\n" for row in mapping), encoding="utf-8"
    )
    return full


def summarize_ratings(
    cards: Sequence[BlindCase],
    ratings: Sequence[CaseRating],
    mapping: Sequence[QualitySlotMapping],
    first_name: str,
) -> QualityComparisonSummary:
    by_rating = {row.example_id: row for row in ratings}
    by_mapping = {row.example_id: row for row in mapping}
    identities = {row.example_id for row in cards}
    if (
        len(by_rating) != len(ratings)
        or len(by_mapping) != len(mapping)
        or identities != by_rating.keys()
        or identities != by_mapping.keys()
    ):
        raise ValueError("Locked ratings and slot mappings must cover every unique card")
    tone_first: list[float] = []
    tone_second: list[float] = []
    ground_first: list[float] = []
    ground_second: list[float] = []
    for card in cards:
        rating, slots = by_rating[card.example_id], by_mapping[card.example_id]
        first: ResponseRating
        second: ResponseRating
        if slots.source_a == first_name:
            first, second = rating.response_a, rating.response_b
        elif slots.source_b == first_name:
            first, second = rating.response_b, rating.response_a
        else:
            raise ValueError("Slot map lacks the named first comparison source")
        tone_first.append(first.tone_helpfulness)
        tone_second.append(second.tone_helpfulness)
        ground_first.append(first.literal_grounding)
        ground_second.append(second.literal_grounding)

    def axis(first: Sequence[float], second: Sequence[float]) -> QualityAxisComparison:
        differences = tuple(left - right for left, right in zip(first, second, strict=True))
        observations = tuple(
            PairedMetricObservation(
                example_id=row.example_id, dialogue_id=row.family_id, difference=delta
            )
            for row, delta in zip(cards, differences, strict=True)
        )
        wins = tuple(
            PairedMetricObservation(
                example_id=row.example_id,
                dialogue_id=row.family_id,
                difference=1.0 if delta > 0 else 0.5 if delta == 0 else 0.0,
            )
            for row, delta in zip(cards, differences, strict=True)
        )
        return QualityAxisComparison(
            first_mean=mean(first),
            second_mean=mean(second),
            first_minus_second=paired_dialogue_bootstrap(observations),
            wins=sum(delta > 0 for delta in differences),
            ties=sum(delta == 0 for delta in differences),
            losses=sum(delta < 0 for delta in differences),
            half_credit_win_fraction=paired_dialogue_bootstrap(wins),
        )

    return QualityComparisonSummary(
        examples=len(cards),
        bases=len({(row.cohort, row.base_id) for row in cards}),
        families=len({row.family_id for row in cards}),
        tone=axis(tone_first, tone_second),
        grounding=axis(ground_first, ground_second),
        tone_better_grounding_not_worse=sum(
            first_tone > second_tone and first_ground >= second_ground
            for first_tone, second_tone, first_ground, second_ground in zip(
                tone_first, tone_second, ground_first, ground_second, strict=True
            )
        ),
        tone_better_grounding_worse=sum(
            first_tone > second_tone and first_ground < second_ground
            for first_tone, second_tone, first_ground, second_ground in zip(
                tone_first, tone_second, ground_first, ground_second, strict=True
            )
        ),
        ambiguous_cases=sum(bool(row.ambiguity_note) for row in ratings),
    )


def freeze_quality_ratings(cards_path: Path, ratings_path: Path, destination: Path) -> None:
    cards = BlindCards.model_validate_json(cards_path.read_bytes())
    ratings = load_records(ratings_path, CaseRating)
    if len(ratings) != len(cards.cases) or {row.example_id for row in ratings} != {
        row.example_id for row in cards.cases
    }:
        raise ValueError("Freeze requires complete unique ratings for exactly these cards")
    if destination.exists():
        raise ValueError("Locked ratings cannot be overwritten")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    partial.write_text("".join(row.model_dump_json() + "\n" for row in ratings), encoding="utf-8")
    partial.replace(destination)


def report_quality_comparison(directory: Path, ratings_path: Path) -> QualityScoredReport:
    provenance = QualityComparisonProvenance.model_validate_json(
        (directory / "provenance.json").read_bytes()
    )
    cards = BlindCards.model_validate_json((directory / "blind_cases.json").read_bytes())
    ratings = load_records(ratings_path, CaseRating)
    mapping = load_records(directory / "slot_mapping.jsonl", QualitySlotMapping)
    first_name = provenance.configuration.first.name
    overall = summarize_ratings(cards.cases, ratings, mapping, first_name)
    cohorts: list[CohortQualityComparison] = []
    for cohort in provenance.configuration.cohorts:
        subset = tuple(row for row in cards.cases if row.cohort == cohort)
        identifiers = {row.example_id for row in subset}
        cohorts.append(
            CohortQualityComparison(
                cohort=cohort,
                comparison=summarize_ratings(
                    subset,
                    tuple(row for row in ratings if row.example_id in identifiers),
                    tuple(row for row in mapping if row.example_id in identifiers),
                    first_name,
                ),
            )
        )
    report = QualityScoredReport(
        provenance=provenance,
        ratings=file_artifact(ratings_path),
        overall=overall,
        cohorts=tuple(cohorts),
    )
    write_record(directory / "quality_summary.json", report)
    lines = [
        "# Model-assisted emotional response comparison",
        "",
        f"First: {first_name}; second: {provenance.configuration.second.name}.",
        "",
        "These are model-assisted ordinal judgments conditioned on intended synthetic labels, "
        "not human emotion accuracy or calibrated acceptability. Reviewers can know the wider "
        "experiment context; only individual response slots are blinded. Scores never compare "
        "teacher resemblance. Equal-step means are heuristic; intervals bootstrap shared design "
        "families with 2,000 draws and seed42. Ties receive half credit.",
        "",
        "| Cohort | n | Tone first−second [95% CI] | W/T/L | "
        "Grounding first−second [95% CI] | W/T/L |",
        "|---|---:|---|---|---|---|",
    ]
    for label, comparison in (("All", overall),) + tuple(
        (row.cohort.value, row.comparison) for row in cohorts
    ):
        tone, grounding = comparison.tone, comparison.grounding
        lines.append(
            f"| {label} | {comparison.examples} | "
            f"{tone.first_minus_second.estimate:+.3f} "
            f"[{tone.first_minus_second.lower:+.3f}, {tone.first_minus_second.upper:+.3f}] | "
            f"{tone.wins}/{tone.ties}/{tone.losses} | "
            f"{grounding.first_minus_second.estimate:+.3f} "
            f"[{grounding.first_minus_second.lower:+.3f}, "
            f"{grounding.first_minus_second.upper:+.3f}] | "
            f"{grounding.wins}/{grounding.ties}/{grounding.losses} |"
        )
    lines.extend(
        [
            "",
            f"Tone improves without grounding loss: {overall.tone_better_grounding_not_worse}; "
            f"with grounding loss: {overall.tone_better_grounding_worse}. "
            f"Prelocked ambiguity notes: {overall.ambiguous_cases}.",
            "",
        ]
    )
    by_id = {row.example_id: row for row in ratings}
    for card in cards.cases:
        rating = by_id[card.example_id]
        lines.extend(
            [
                f"## {card.example_id} / {card.intended_delivery}",
                "",
                f"USER: {card.user_text}",
                "",
                f"A: {card.response_a}",
                "",
                f"A tone/grounding: {rating.response_a.tone_helpfulness}/"
                f"{rating.response_a.literal_grounding}; {rating.response_a.reason}",
                "",
                f"B: {card.response_b}",
                "",
                f"B tone/grounding: {rating.response_b.tone_helpfulness}/"
                f"{rating.response_b.literal_grounding}; {rating.response_b.reason}",
                "",
                f"Confidence {rating.confidence.value}; "
                f"ambiguity: {rating.ambiguity_note or '[none]'}",
                "",
            ]
        )
    (directory / "quality_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report
