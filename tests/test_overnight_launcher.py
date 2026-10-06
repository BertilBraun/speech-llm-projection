from pathlib import Path

import pytest

from speech_projector.emotion_preview import Delivery
from speech_projector.models import Example, Split
from speech_projector.overnight_configuration import sweep_runs
from speech_projector.overnight_data import (
    Cohort,
    NeuEmotionalExampleSource,
    OrdinaryExampleSource,
    QwenEmotionalExampleSource,
    SourceSidecar,
)
from speech_projector.overnight_judge import FinalJudgingQuota
from speech_projector.overnight_launcher import (
    ContinuationBudget,
    OvernightSuiteConfig,
    continuation_fits_budget,
    final_evaluation_config,
    select_final_evaluation,
)
from speech_projector.tts_pilot import PilotEmotion


def final_rows() -> tuple[tuple[Example, ...], tuple[SourceSidecar, ...]]:
    examples: list[Example] = []
    sources: list[SourceSidecar] = []
    for cohort, count in (
        (Cohort.ORDINARY, 96),
        (Cohort.QWEN_EMOTIONAL, 40),
        (Cohort.NEU_EMOTIONAL, 40),
    ):
        for base in range(count):
            for member in range(1 if cohort == Cohort.ORDINARY else 2):
                identifier = f"{cohort.value}:{base}:{member}"
                examples.append(
                    Example(
                        example_id=identifier,
                        dialogue_id=f"{cohort.value}:{base}",
                        split=Split.TEST,
                        history=(),
                        user_text=f"Same words {base}.",
                        target_text=f"Reply {member}.",
                        audio_path=Path(identifier.replace(":", "_") + ".wav"),
                        duration=2,
                        domain="test",
                        emotion="",
                        feature_path=Path("feature.pt"),
                    )
                )
                match cohort:
                    case Cohort.ORDINARY:
                        source = OrdinaryExampleSource(
                            example_id=identifier,
                            source_manifest=Path("ordinary"),
                            source_example_id=identifier,
                        )
                    case Cohort.QWEN_EMOTIONAL:
                        source = QwenEmotionalExampleSource(
                            example_id=identifier,
                            source_manifest=Path("qwen"),
                            source_example_id=identifier,
                            base_id=f"q{base}",
                            family_id=f"q{base}",
                            emotion=(Delivery.HAPPY, Delivery.SAD)[member],
                        )
                    case Cohort.NEU_EMOTIONAL:
                        source = NeuEmotionalExampleSource(
                            example_id=identifier,
                            source_manifest=Path("neu"),
                            source_example_id=identifier,
                            base_id=f"n{base}",
                            family_id=f"n{base}",
                            emotion=(PilotEmotion.HAPPY, PilotEmotion.ANGRY)[member],
                        )
                sources.append(source)
    return tuple(examples), tuple(sources)


def test_final_generation_quotas_preserve_full_ce_population_and_training_config() -> None:
    examples, sources = final_rows()
    selection = select_final_evaluation(examples, sources, FinalJudgingQuota())
    assert len(selection.examples) == 256
    assert set(selection.examples) == set(examples)
    assert len(selection.generation_example_ids) == 136
    assert (
        tuple(example.example_id for example in selection.examples[:136])
        == selection.generation_example_ids
    )
    source_by_id = {source.example_id: source for source in sources}
    assert [
        sum(
            source_by_id[identifier].cohort == cohort
            for identifier in selection.generation_example_ids
        )
        for cohort in Cohort
    ] == [48, 24, 64]
    for cohort in (Cohort.QWEN_EMOTIONAL, Cohort.NEU_EMOTIONAL):
        selected_sources = [
            source_by_id[identifier]
            for identifier in selection.generation_example_ids
            if source_by_id[identifier].cohort == cohort
        ]
        for source in selected_sources:
            match source:
                case QwenEmotionalExampleSource() | NeuEmotionalExampleSource():
                    assert (
                        sum(candidate.base_id == source.base_id for candidate in selected_sources)
                        == 2
                    )
    configuration = sweep_runs(38194, 1)[0]
    evaluated = final_evaluation_config(configuration, selection)
    assert evaluated.qualitative_examples == evaluated.semantic_examples == 136
    assert configuration.qualitative_examples == configuration.semantic_examples == 24
    assert (
        evaluated.model_copy(update={"qualitative_examples": 24, "semantic_examples": 24})
        == configuration
    )
    assert selection == select_final_evaluation(examples, sources, FinalJudgingQuota())


def test_final_selection_rejects_missing_complete_pairs() -> None:
    examples, sources = final_rows()
    incomplete = tuple(
        example
        for example in examples
        if not example.example_id.startswith("neu_emotional:") or example.example_id.endswith(":0")
    )
    identifiers = {example.example_id for example in incomplete}
    with pytest.raises(ValueError, match="two variants"):
        select_final_evaluation(
            incomplete,
            tuple(source for source in sources if source.example_id in identifiers),
            FinalJudgingQuota(),
        )


def test_optional_compact_continuation_cannot_consume_minimum_winner_pass_budget() -> None:
    configuration = OvernightSuiteConfig(
        data_root=Path("data"),
        output_root=Path("results"),
        deadline_unix_time=18000,
        finalization_reserve_seconds=3600,
    )
    optional_without_minimum = ContinuationBudget(
        remaining_updates=2000,
        seconds_per_update=2,
        minimum_final_pass_reserve_seconds=0,
    )
    optional_with_minimum = ContinuationBudget(
        remaining_updates=2000,
        seconds_per_update=2,
        minimum_final_pass_reserve_seconds=(4775 - 4000) * 2 + 120,
    )
    winner_full_pass = ContinuationBudget(
        remaining_updates=4775 - 4000,
        seconds_per_update=2,
        minimum_final_pass_reserve_seconds=0,
    )
    assert continuation_fits_budget(configuration, optional_without_minimum, 10000)
    assert not continuation_fits_budget(configuration, optional_with_minimum, 10000)
    assert continuation_fits_budget(configuration, winner_full_pass, 10000)
    assert continuation_fits_budget(configuration, winner_full_pass, 12730)
    assert not continuation_fits_budget(configuration, winner_full_pass, 12731)
