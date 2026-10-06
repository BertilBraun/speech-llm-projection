"""Pair/family boundaries, pooled-state arithmetic and a complete CPU diagnostic."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch

from speech_projector.acoustic_tone_probe import (
    LABELS,
    FeatureRepresentation,
    ToneProbeConfig,
    ToneProbeInput,
    acoustic_measurement,
    classify_features,
    mean_std_feature,
    pair_center_features,
    run_tone_probe,
    select_probe_inputs,
)
from speech_projector.models import Example, Split
from speech_projector.neu_dataset import EMOTION_PAIRS
from speech_projector.overnight_data import NeuEmotionalExampleSource


def inputs(directory: Path) -> tuple[ToneProbeInput, ...]:
    rows: list[ToneProbeInput] = []
    for split in Split:
        for index, pair in enumerate(EMOTION_PAIRS):
            base = f"{split.value}_{index}"
            for emotion in pair:
                identity = f"{base}_{emotion.value}"
                example = Example(
                    example_id=identity,
                    dialogue_id=base,
                    split=split,
                    history=(),
                    user_text=(
                        f"Please check the package numbered {base} before our appointment tomorrow."
                    ),
                    target_text="I can help you check that.",
                    audio_path=directory / f"{identity}.wav",
                    duration=0.1,
                    domain="home",
                    emotion=emotion.value,
                    feature_path=directory / f"{identity}.pt",
                )
                source = NeuEmotionalExampleSource(
                    example_id=identity,
                    source_manifest=directory / "source.jsonl",
                    source_example_id=identity,
                    base_id=base,
                    family_id=base,
                    emotion=emotion,
                )
                rows.append(ToneProbeInput(example, source))
    return tuple(rows)


def config(directory: Path) -> ToneProbeConfig:
    return ToneProbeConfig(
        manifest=directory / "examples.jsonl",
        sidecar=directory / "sources.jsonl",
        output_directory=directory / "probe",
        train_examples=8,
        validation_examples=8,
        test_examples=8,
    )


def test_selects_deterministic_complete_pairs_without_family_leakage(tmp_path: Path) -> None:
    rows = inputs(tmp_path)
    selected = select_probe_inputs(
        tuple(row.example for row in rows), tuple(row.source for row in rows), config(tmp_path)
    )
    assert selected == select_probe_inputs(
        tuple(row.example for row in reversed(rows)),
        tuple(row.source for row in rows),
        config(tmp_path),
    )
    assert len(selected) == 24
    assert all(
        selected[index].source.base_id == selected[index + 1].source.base_id
        for index in range(0, 24, 2)
    )
    leaking = replace(
        rows[-1],
        source=NeuEmotionalExampleSource(
            example_id=rows[-1].source.example_id,
            source_manifest=rows[-1].source.source_manifest,
            source_example_id=rows[-1].source.source_example_id,
            base_id=rows[-1].source.base_id,
            family_id=rows[0].source.family_id,
            emotion=rows[-1].source.emotion,
        ),
    )
    with pytest.raises(ValueError, match="family leaks"):
        select_probe_inputs(
            tuple(row.example for row in rows),
            tuple(row.source for row in rows[:-1]) + (leaking.source,),
            config(tmp_path),
        )
    with pytest.raises(ValueError, match="Incomplete"):
        select_probe_inputs(
            tuple(row.example for row in rows),
            tuple(row.source for row in rows[:-1]),
            config(tmp_path),
        )


def test_mean_std_retains_correct_population_variance_and_single_frame() -> None:
    hidden = torch.stack((torch.zeros(768), torch.full((768,), 2.0)))
    vector = mean_std_feature(hidden)
    np.testing.assert_array_equal(vector, np.ones(1536))
    np.testing.assert_array_equal(mean_std_feature(hidden[:1])[768:], np.zeros(768))
    with pytest.raises(ValueError, match="nonempty"):
        mean_std_feature(torch.empty(0, 768))
    with pytest.raises(ValueError, match="nonfinite"):
        mean_std_feature(torch.full((2, 768), float("nan")))


def test_centered_features_use_only_the_corresponding_same_text_pair(tmp_path: Path) -> None:
    rows = inputs(tmp_path)
    values = np.arange(len(rows) * 2, dtype=np.float64).reshape(len(rows), 2)
    centered = pair_center_features(values, rows)
    for index in range(0, len(rows), 2):
        np.testing.assert_array_equal(centered[index] + centered[index + 1], np.zeros(2))
        np.testing.assert_array_equal(
            centered[index] - centered[index + 1], values[index] - values[index + 1]
        )
    with pytest.raises(ValueError, match="both delivery"):
        pair_center_features(values[:-1], rows[:-1])


def test_training_only_classifier_generalizes_known_synthetic_signal(tmp_path: Path) -> None:
    rows = inputs(tmp_path)
    features = np.stack([np.eye(4)[LABELS.index(row.source.emotion)] for row in rows])
    metrics, predictions = classify_features(
        features, rows, FeatureRepresentation.RAW, config(tmp_path)
    )
    assert all(metric.accuracy == metric.balanced_accuracy == 1 for metric in metrics)
    assert all(metric.majority_balanced_accuracy == pytest.approx(0.25) for metric in metrics)
    assert all(row.predicted_emotion == row.intended_emotion for row in predictions)
    assert all(sum(sum(row) for row in metric.confusion) == metric.examples for metric in metrics)


def test_full_cpu_probe_writes_hashes_predictions_and_pair_acoustics(tmp_path: Path) -> None:
    rows = inputs(tmp_path)
    for row in rows:
        hidden = torch.zeros(4, 768)
        hidden[:, LABELS.index(row.source.emotion)] = 2
        torch.save(hidden, row.example.feature_path)
        samples = np.sin(2 * np.pi * 1000 * np.arange(1600) / 16000) * 0.2
        soundfile.write(row.example.audio_path, samples, 16000, subtype="FLOAT")
    configuration = config(tmp_path)
    configuration.manifest.write_text("".join(row.example.model_dump_json() + "\n" for row in rows))
    configuration.sidecar.write_text("".join(row.source.model_dump_json() + "\n" for row in rows))
    report = run_tone_probe(
        configuration, tuple(row.example for row in rows), tuple(row.source for row in rows)
    )
    assert report.feature_dimension == 1536 and report.cpu_threads == 2
    assert len(report.metrics) == 6
    assert (configuration.output_directory / "predictions.jsonl").read_text().count("\n") == 48
    assert (configuration.output_directory / "pair_acoustic_differences.jsonl").read_text().count(
        "\n"
    ) == 12
    assert report.pair_rms_difference.maximum == 0
    assert len(report.manifest_sha256) == len(report.sidecar_sha256) == 64
    measurement = acoustic_measurement(rows[0])
    assert measurement.rms == pytest.approx(0.2 / np.sqrt(2), rel=1e-5)
    assert measurement.spectral_centroid_hz == pytest.approx(1000, abs=0.01)
