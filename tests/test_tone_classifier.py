"""Classifier fit isolation, pair boundaries and deployable probability contracts."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from speech_projector.acoustic_tone_probe import LABELS, ToneProbeInput
from speech_projector.journal import read_journal
from speech_projector.models import Example, Split
from speech_projector.overnight_data import NeuEmotionalExampleSource
from speech_projector.tone_classifier import (
    ToneClassifierConfig,
    ToneClassifierPrediction,
    ToneProbability,
    classifier_feature,
    classifier_metric,
    fit_classifier,
    predict_features,
    select_classifier_inputs,
    train_tone_classifier,
)


def classifier_inputs(directory: Path) -> tuple[ToneProbeInput, ...]:
    rows: list[ToneProbeInput] = []
    for split in Split:
        for pair_index in range(2):
            base = f"{split.value}_{pair_index}"
            for tone in LABELS[pair_index * 2 : pair_index * 2 + 2]:
                identity = f"{base}_{tone.value}"
                example = Example(
                    example_id=identity,
                    dialogue_id=base,
                    split=split,
                    history=(),
                    user_text=f"Please check item {base} before the scheduled meeting tomorrow.",
                    target_text="I can help.",
                    audio_path=directory / f"{identity}.wav",
                    duration=0.5,
                    domain="work",
                    emotion=tone.value,
                    feature_path=directory / f"{identity}.pt",
                )
                source = NeuEmotionalExampleSource(
                    example_id=identity,
                    source_manifest=directory / "source.jsonl",
                    source_example_id=identity,
                    base_id=base,
                    family_id=base,
                    emotion=tone,
                )
                rows.append(ToneProbeInput(example, source))
    return tuple(rows)


def configuration(directory: Path) -> ToneClassifierConfig:
    return ToneClassifierConfig(
        manifest=directory / "examples.jsonl",
        sidecar=directory / "sources.jsonl",
        output_directory=directory / "classifier",
    )


def test_mean_std_and_temporal_order_are_preserved() -> None:
    hidden = torch.arange(8, dtype=torch.float32).view(8, 1).repeat(1, 768)
    vector = classifier_feature(hidden, 4)
    assert vector.shape == (4608,)
    assert vector[0] == 3.5
    assert vector[768] == pytest.approx(np.std(np.arange(8)))
    assert vector[1536::768].tolist() == [0.5, 2.5, 4.5, 6.5]


@pytest.mark.parametrize("hidden", [torch.empty(0, 768), torch.zeros(3, 768), torch.zeros(8, 3)])
def test_invalid_cache_shape_or_too_few_frames_fails(hidden: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        classifier_feature(hidden, 4)


@pytest.mark.parametrize("crossing", ["family", "text", "path"])
def test_cross_split_leakage_fails(directory: Path, crossing: str) -> None:
    rows = list(classifier_inputs(directory))
    training, heldout = rows[0], rows[4]
    match crossing:
        case "family":
            heldout = ToneProbeInput(
                heldout.example,
                heldout.source.model_copy(update={"family_id": training.source.family_id}),
            )
        case "text":
            heldout = ToneProbeInput(
                heldout.example.model_copy(update={"user_text": training.example.user_text}),
                heldout.source,
            )
        case "path":
            heldout = ToneProbeInput(
                heldout.example.model_copy(update={"feature_path": training.example.feature_path}),
                heldout.source,
            )
    rows[4] = heldout
    with pytest.raises(ValueError, match="crosses splits"):
        select_classifier_inputs([row.example for row in rows], [row.source for row in rows])


@pytest.fixture
def directory(tmp_path: Path) -> Path:
    return tmp_path


def test_scaler_and_fit_ignore_heldout_values(directory: Path) -> None:
    rows = classifier_inputs(directory)
    features = np.arange(len(rows) * 3, dtype=np.float64).reshape(len(rows), 3)
    changed = features.copy()
    changed[4:] += 100_000
    first = fit_classifier(features, rows, configuration(directory))
    second = fit_classifier(changed, rows, configuration(directory))
    scaler: StandardScaler = first["standardscaler"]
    assert np.allclose(scaler.mean_, features[:4].mean(axis=0))
    assert np.array_equal(first.predict_proba(features[:4]), second.predict_proba(features[:4]))


def test_probability_contract_catches_false_prediction() -> None:
    probabilities = tuple(ToneProbability(tone=tone, probability=0.25) for tone in LABELS)
    with pytest.raises(ValidationError, match="maximum probability"):
        ToneClassifierPrediction(
            example_id="case",
            predicted_tone=LABELS[1],
            probabilities=probabilities,
            confidence=0.25,
        )


def test_pair_correctness_requires_both_deliveries(directory: Path) -> None:
    rows = classifier_inputs(directory)
    predictions: list[ToneClassifierPrediction] = []
    for index, row in enumerate(rows[:4]):
        predicted = row.source.emotion if index else LABELS[1]
        predictions.append(
            ToneClassifierPrediction(
                example_id=row.example.example_id,
                predicted_tone=predicted,
                probabilities=tuple(
                    ToneProbability(tone=tone, probability=float(tone == predicted))
                    for tone in LABELS
                ),
                confidence=1.0,
            )
        )
    metric = classifier_metric(rows, predictions, Split.TRAIN)
    assert metric.accuracy == 0.75
    assert metric.both_deliveries_correct_pairs == 1
    assert metric.both_deliveries_correct_fraction == 0.5


def test_persisted_model_reproduces_predictions_without_reference_labels(directory: Path) -> None:
    rows = classifier_inputs(directory)
    config = configuration(directory)
    for row in rows:
        torch.save(
            torch.full((8, 768), float(LABELS.index(row.source.emotion))), row.example.feature_path
        )
    config.manifest.write_text("\n".join(row.example.model_dump_json() for row in rows) + "\n")
    config.sidecar.write_text("\n".join(row.source.model_dump_json() for row in rows) + "\n")
    report = train_tone_classifier(config, "test-source")
    with (config.output_directory / "classifier.pkl").open("rb") as stream:
        estimator: Pipeline = pickle.load(stream)
    validation = tuple(row for row in rows if row.example.split == Split.VALIDATION)
    features = np.stack(
        [
            classifier_feature(torch.load(row.example.feature_path, weights_only=True), 4)
            for row in validation
        ]
    )
    expected = predict_features(estimator, features, [row.example.example_id for row in validation])
    actual = read_journal(
        config.output_directory / "validation_predictions.jsonl", ToneClassifierPrediction
    )
    assert actual == expected
    assert "intended" not in actual[0].model_dump_json()
    assert tuple(metric.examples for metric in report.metrics) == (4, 4, 4)
    with pytest.raises(ValueError, match="already exists"):
        train_tone_classifier(config, "test-source")
