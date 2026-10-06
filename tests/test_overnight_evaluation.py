"""Validate same-word preference cancellation and validation-only selection."""

from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor, nn

from speech_projector.emotion_preview import Delivery
from speech_projector.inputs import SpeechInput, UtteranceInput
from speech_projector.llm import FrozenQwen
from speech_projector.models import (
    EvaluationMetrics,
    Example,
    ExperimentStage,
    MlpProjectorConfig,
    RunConfig,
    Split,
)
from speech_projector.overnight_data import (
    Cohort,
    NeuEmotionalExampleSource,
    OrdinaryExampleSource,
    QwenEmotionalExampleSource,
    SourceSidecar,
)
from speech_projector.overnight_evaluation import (
    EmotionEvaluationPair,
    FixedValidationConfig,
    PairedEmotionLoss,
    SweepCandidate,
    SweepSelectionPolicy,
    ValidationCohorts,
    build_emotion_pairs,
    evaluate_emotion_pairs,
    select_fixed_validation,
    select_sweep,
    summarize_preferences,
)
from speech_projector.projectors import Projector
from speech_projector.tts_pilot import PilotEmotion


def pair_loss(base: str, family: str, margin: float) -> PairedEmotionLoss:
    return PairedEmotionLoss(
        base_id=base,
        family_id=family,
        first_example_id=base + "a",
        second_example_id=base + "b",
        first_target_tokens=10,
        second_target_tokens=20,
        targets_identical=False,
        first_audio_first_target=1,
        second_audio_first_target=1 + margin,
        second_audio_second_target=2,
        first_audio_second_target=2 + margin,
        resized_second_audio_first_target=1 + margin / 2,
        resized_first_audio_second_target=2 + margin / 2,
    )


def candidate(
    name: str, ordinary: float, macro: float, margin: float, rate: float
) -> SweepCandidate:
    def metrics(cross_entropy: float) -> EvaluationMetrics:
        return EvaluationMetrics(
            examples=40, target_tokens=100, cross_entropy=cross_entropy, perplexity=2
        )

    preference = summarize_preferences((pair_loss("a", "f1", margin), pair_loss("b", "f2", margin)))
    return SweepCandidate(
        configuration=RunConfig(
            name=name,
            stage=ExperimentStage.V2,
            train_examples=40000,
            epochs=1,
            learning_rate=0.001,
            projector=MlpProjectorConfig(compression_factor=5),
        ),
        validation=ValidationCohorts(
            old_ordinary=metrics(ordinary),
            old_emotional=metrics((3 * macro - ordinary) / 2),
            new_neu_emotional=metrics((3 * macro - ordinary) / 2),
        ),
        old_emotional_preference=preference,
        new_neu_preference=preference,
        pseudo_tokens_per_second=rate,
        training_seconds_per_update=1,
    )


def test_matching_margin_cancels_target_priors_and_clusters_by_family() -> None:
    original = pair_loss("a", "family1", 0.2)
    shifted = original.model_copy(
        update={
            "first_audio_first_target": original.first_audio_first_target + 100,
            "second_audio_first_target": original.second_audio_first_target + 100,
            "second_audio_second_target": original.second_audio_second_target + 200,
            "first_audio_second_target": original.first_audio_second_target + 200,
        }
    )
    assert shifted.matching_margin == pytest.approx(original.matching_margin)
    summary = summarize_preferences(
        (original, pair_loss("b", "family1", 0.3), pair_loss("c", "f2", 0))
    )
    assert summary.matching_margin.examples == 3
    assert summary.matching_margin.dialogues == summary.family_clusters == 2
    assert summary.matching_win_rate.estimate == pytest.approx(2 / 3)
    assert summary.tie_rate == pytest.approx(1 / 3)


def test_selection_preserves_content_then_new_tone_and_compact_guard() -> None:
    candidates = (
        candidate("lowest_ce", 1, 1, 0.1, 25),
        candidate("new_tone_leader", 1, 1.02, 0.2, 25),
        candidate("compact", 1, 1.06, 0.05, 10),
        candidate("too_weak", 1, 1.12, 0.5, 2.5),
        candidate("content_regression", 1.11, 0.8, 10, 25),
    )
    decision = select_sweep(candidates, SweepSelectionPolicy())
    assert decision.quality_leader == "new_tone_leader"
    assert decision.compact_winners == ("compact",)
    assert "content_regression" not in decision.ordinary_guard_eligible
    assert "no test metric" in decision.rationale.lower()
    assert decision == type(decision).model_validate_json(decision.model_dump_json())


def test_selection_does_not_invent_compact_winner() -> None:
    decision = select_sweep((candidate("only", 1, 1, 0, 10),), SweepSelectionPolicy())
    assert decision.compact_winners == ()
    with pytest.raises(ValueError, match="unique"):
        select_sweep(decision.candidates * 2, SweepSelectionPolicy())


def example(identifier: str, text: str) -> Example:
    return Example(
        example_id=identifier,
        dialogue_id=identifier,
        split=Split.VALIDATION,
        history=(),
        user_text=text,
        target_text="An appropriate reply.",
        audio_path=Path(identifier + ".wav"),
        duration=2,
        domain="conversation",
        emotion="",
        feature_path=Path(identifier + ".pt"),
    )


def test_fixed_selection_keeps_complete_pairs_and_balanced_generation_prefix() -> None:
    examples: list[Example] = []
    sources: list[SourceSidecar] = []
    for index in range(4):
        current = example(f"ordinary_{index}", "An ordinary utterance.")
        examples.append(current)
        sources.append(
            OrdinaryExampleSource(
                example_id=current.example_id,
                source_manifest=Path("ordinary.jsonl"),
                source_example_id=current.example_id,
            )
        )
    for cohort in (Cohort.QWEN_EMOTIONAL, Cohort.NEU_EMOTIONAL):
        for base in range(3):
            for index in range(2):
                identifier = f"{cohort.value}_{base}_{index}"
                examples.append(example(identifier, f"The same literal words for base {base}."))
                match cohort:
                    case Cohort.QWEN_EMOTIONAL:
                        source = QwenEmotionalExampleSource(
                            example_id=identifier,
                            source_manifest=Path("qwen.jsonl"),
                            source_example_id=identifier,
                            base_id=f"qwen_{base}",
                            family_id=f"qwen_family{base}",
                            emotion=(Delivery.HAPPY, Delivery.SAD)[index],
                        )
                    case Cohort.NEU_EMOTIONAL:
                        source = NeuEmotionalExampleSource(
                            example_id=identifier,
                            source_manifest=Path("neu.jsonl"),
                            source_example_id=identifier,
                            base_id=f"neu_{base}",
                            family_id=f"neu_family{base}",
                            emotion=(PilotEmotion.HAPPY, PilotEmotion.ANGRY)[index],
                        )
                    case _:
                        raise AssertionError("Test cohort is emotional")
                sources.append(source)
    configuration = FixedValidationConfig(
        ordinary_examples=4,
        pairs_per_emotional_cohort=2,
        ordinary_generations=2,
        generated_pairs_per_cohort=1,
    )
    selected = select_fixed_validation(examples, sources, configuration)
    assert len(selected.examples) == 12
    assert len(selected.generation_example_ids) == 6
    assert (
        tuple(item.example_id for item in selected.examples[:6]) == selected.generation_example_ids
    )
    assert tuple(item.cohort for item in selected.sources[:6]) == (
        Cohort.ORDINARY,
        Cohort.ORDINARY,
        Cohort.QWEN_EMOTIONAL,
        Cohort.QWEN_EMOTIONAL,
        Cohort.NEU_EMOTIONAL,
        Cohort.NEU_EMOTIONAL,
    )
    for cohort in (Cohort.QWEN_EMOTIONAL, Cohort.NEU_EMOTIONAL):
        pairs = build_emotion_pairs(selected.examples, selected.sources, cohort, Split.VALIDATION)
        assert len(pairs) == 2
        assert len({pair.family_id for pair in pairs}) == 2
    assert selected == select_fixed_validation(examples, sources, configuration)
    missing = sources[:-1]
    with pytest.raises(ValueError, match="coverage"):
        select_fixed_validation(examples, missing, configuration)


class TestProjector(nn.Module):
    __test__ = False

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([1.0]))

    def forward(self, features: Tensor) -> Tensor:
        return features * self.scale


class TestWrapper:
    __test__ = False

    def __init__(self, configuration: RunConfig) -> None:
        self.model = nn.Linear(1, 1)
        self.config = configuration
        self.device = torch.device("cpu")
        self.calls: list[tuple[str, str, int]] = []

    def target_token_count(self, example: Example) -> int:
        return len(example.target_text.split()) + 1

    def loss(self, example: Example, utterance: UtteranceInput) -> Tensor:
        match utterance:
            case SpeechInput(embeddings=embeddings):
                self.calls.append((example.example_id, example.target_text, embeddings.shape[0]))
                expected = 1 if example.target_text == "first" else 3
                return (embeddings.mean() - expected).square()
            case _:
                raise AssertionError("Speech pairing must never inject a transcript")


def test_pair_scoring_uses_same_targets_for_swaps_and_resumes_bound_weights(tmp_path: Path) -> None:
    first = example("first_audio", "Exactly the same literal words.").model_copy(
        update={"target_text": "first", "feature_path": tmp_path / "first.pt"}
    )
    second = example("second_audio", first.user_text).model_copy(
        update={"target_text": "second", "feature_path": tmp_path / "second.pt"}
    )
    torch.save(torch.ones((2, 768)), first.feature_path)
    torch.save(torch.full((4, 768), 3.0), second.feature_path)
    configuration = candidate("scoring", 1, 1, 0.1, 10).configuration
    wrapper = TestWrapper(configuration)
    projector = TestProjector()
    pair = EmotionEvaluationPair("base", "family", first, second)
    results = evaluate_emotion_pairs(
        cast(FrozenQwen, wrapper), cast(Projector, projector), (pair,), tmp_path / "evaluation"
    )
    assert results[0].matching_margin == results[0].resized_matching_margin == 4
    assert wrapper.calls == [
        (first.example_id, "first", 2),
        (first.example_id, "first", 4),
        (second.example_id, "second", 4),
        (second.example_id, "second", 2),
        (first.example_id, "first", 2),
        (second.example_id, "second", 4),
    ]
    assert (
        evaluate_emotion_pairs(
            cast(FrozenQwen, wrapper), cast(Projector, projector), (pair,), tmp_path / "evaluation"
        )
        == results
    )
    assert len(wrapper.calls) == 6
    with torch.no_grad():
        projector.scale.fill_(2)
    with pytest.raises(ValueError, match="weights or targets"):
        evaluate_emotion_pairs(
            cast(FrozenQwen, wrapper), cast(Projector, projector), (pair,), tmp_path / "evaluation"
        )
