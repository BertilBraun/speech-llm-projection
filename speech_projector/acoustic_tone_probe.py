"""CPU-only intended-delivery prediction from frozen Whisper mean/std features."""

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import soundfile
import torch
from numpy.typing import NDArray
from pydantic import Field, model_validator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.inventory_results import write_record
from scripts.neutts_pilot_state import digest
from speech_projector.cache import load_feature
from speech_projector.data import Distribution, distribution
from speech_projector.models import Example, Record, Split
from speech_projector.overnight_data import NeuEmotionalExampleSource
from speech_projector.tts_pilot import PilotEmotion

LABELS = (PilotEmotion.ANGRY, PilotEmotion.FEARFUL, PilotEmotion.HAPPY, PilotEmotion.SAD)


class ToneProbeConfig(Record):
    manifest: Path
    sidecar: Path
    output_directory: Path
    train_examples: int = Field(default=2000, gt=0)
    validation_examples: int = Field(default=256, gt=0)
    test_examples: int = Field(default=256, gt=0)
    seed: int = Field(default=42, ge=0)
    pair_centered: bool = True
    maximum_iterations: int = Field(default=1000, gt=0)

    @model_validator(mode="after")
    def validate_complete_pair_counts(self) -> "ToneProbeConfig":
        if any(
            count % 2
            for count in (self.train_examples, self.validation_examples, self.test_examples)
        ):
            raise ValueError("Probe counts must be even to retain complete same-text pairs")
        return self


@dataclass(frozen=True)
class ToneProbeInput:
    example: Example
    source: NeuEmotionalExampleSource


class FeatureRepresentation(str, Enum):
    RAW = "mean_std"
    PAIR_CENTERED = "pair_centered_mean_std"


class TonePrediction(Record):
    example_id: str
    base_id: str
    family_id: str
    split: Split
    intended_emotion: PilotEmotion
    predicted_emotion: PilotEmotion
    representation: FeatureRepresentation


class ToneProbeMetric(Record):
    representation: FeatureRepresentation
    split: Split
    examples: int
    families: int
    label_counts: tuple[tuple[PilotEmotion, int], ...]
    accuracy: float
    balanced_accuracy: float
    majority_label: PilotEmotion
    majority_accuracy: float
    majority_balanced_accuracy: float
    confusion: tuple[tuple[int, ...], ...]


class AcousticMeasurement(Record):
    example_id: str
    base_id: str
    intended_emotion: PilotEmotion
    duration_seconds: float
    rms: float
    absolute_peak: float
    spectral_centroid_hz: float


class PairAcousticDifference(Record):
    base_id: str
    example_ids: tuple[str, str]
    absolute_duration_difference_seconds: float
    absolute_rms_difference: float
    absolute_centroid_difference_hz: float


class ToneProbeReport(Record):
    configuration: ToneProbeConfig
    manifest_sha256: str
    sidecar_sha256: str
    selected_example_ids: tuple[str, ...]
    feature_dimension: int
    classifier_seconds: float
    total_wall_seconds: float
    metrics: tuple[ToneProbeMetric, ...]
    duration_seconds: Distribution
    rms: Distribution
    pair_rms_difference: Distribution
    pair_centroid_difference_hz: Distribution
    cpu_threads: int
    limitations: tuple[str, ...]


def select_probe_inputs(
    examples: Sequence[Example],
    sources: Sequence[NeuEmotionalExampleSource],
    config: ToneProbeConfig,
) -> tuple[ToneProbeInput, ...]:
    examples_by_id = {example.example_id: example for example in examples}
    if len(examples_by_id) != len(examples):
        raise ValueError("Probe manifest repeats an example ID")
    groups: dict[str, list[ToneProbeInput]] = {}
    family_splits: dict[str, Split] = {}
    observed_ids: set[str] = set()
    for source in sources:
        if source.example_id in observed_ids:
            raise ValueError("Neu sidecar repeats an example ID")
        observed_ids.add(source.example_id)
        if source.example_id not in examples_by_id:
            raise ValueError("Neu sidecar refers to an absent example")
        example = examples_by_id[source.example_id]
        if source.emotion not in LABELS:
            raise ValueError("Probe supports only the four new Neu delivery labels")
        previous = family_splits.setdefault(source.family_id, example.split)
        if previous != example.split:
            raise ValueError("A scenario family leaks across probe splits")
        groups.setdefault(source.base_id, []).append(ToneProbeInput(example, source))
    pairs: dict[Split, list[tuple[ToneProbeInput, ToneProbeInput]]] = {split: [] for split in Split}
    for base_id, rows in groups.items():
        if len(rows) != 2:
            raise ValueError(f"Incomplete same-text probe pair: {base_id}")
        first, second = rows
        if (
            first.example.split != second.example.split
            or first.example.user_text != second.example.user_text
            or first.source.family_id != second.source.family_id
            or first.source.emotion == second.source.emotion
        ):
            raise ValueError(f"Pair identity, literal words or split differs: {base_id}")
        pairs[first.example.split].append((first, second))
    selected: list[ToneProbeInput] = []
    for split, count in (
        (Split.TRAIN, config.train_examples),
        (Split.VALIDATION, config.validation_examples),
        (Split.TEST, config.test_examples),
    ):
        available = sorted(
            pairs[split],
            key=lambda pair: hashlib.sha256(
                f"{config.seed}:{pair[0].source.base_id}".encode()
            ).hexdigest(),
        )
        if len(available) * 2 < count:
            raise ValueError(f"Insufficient complete {split.value} pairs for requested probe count")
        selected.extend(row for pair in available[: count // 2] for row in pair)
    return tuple(selected)


def mean_std_feature(hidden: torch.Tensor) -> NDArray[np.float64]:
    if hidden.ndim != 2 or hidden.shape[0] == 0 or hidden.shape[1] != 768:
        raise ValueError("Whisper cache must be nonempty [frames,768] hidden states")
    hidden = hidden.float()
    if not torch.isfinite(hidden).all():
        raise ValueError("Whisper cache contains nonfinite hidden states")
    vector = torch.cat((hidden.mean(dim=0), hidden.std(dim=0, unbiased=False)))
    return np.asarray(vector.numpy(), dtype=np.float64)


def pair_center_features(
    features: NDArray[np.float64], inputs: Sequence[ToneProbeInput]
) -> NDArray[np.float64]:
    centered = features.copy()
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(inputs):
        groups.setdefault(row.source.base_id, []).append(index)
    for indices in groups.values():
        if len(indices) != 2:
            raise ValueError("Pair-centered diagnostic requires both delivery features")
        pair_mean = features[indices].mean(axis=0)
        centered[indices] -= pair_mean
    return centered


def acoustic_measurement(row: ToneProbeInput) -> AcousticMeasurement:
    audio, sample_rate = soundfile.read(row.example.audio_path, dtype="float64", always_2d=False)
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Probe audio must be finite, nonempty and mono")
    spectrum = np.abs(np.fft.rfft(values))
    frequencies = np.fft.rfftfreq(values.size, d=1 / sample_rate)
    power = float(spectrum.sum())
    centroid = float(np.dot(frequencies, spectrum) / power) if power > 0 else 0.0
    return AcousticMeasurement(
        example_id=row.example.example_id,
        base_id=row.source.base_id,
        intended_emotion=row.source.emotion,
        duration_seconds=values.size / sample_rate,
        rms=float(np.sqrt(np.mean(values**2))),
        absolute_peak=float(np.abs(values).max()),
        spectral_centroid_hz=centroid,
    )


def acoustic_pair_differences(
    measurements: Sequence[AcousticMeasurement],
) -> tuple[PairAcousticDifference, ...]:
    groups: dict[str, list[AcousticMeasurement]] = {}
    for row in measurements:
        groups.setdefault(row.base_id, []).append(row)
    differences: list[PairAcousticDifference] = []
    for base_id, pair in groups.items():
        if len(pair) != 2:
            raise ValueError("Acoustic pair differences require both same-text deliveries")
        first, second = pair
        differences.append(
            PairAcousticDifference(
                base_id=base_id,
                example_ids=(first.example_id, second.example_id),
                absolute_duration_difference_seconds=abs(
                    first.duration_seconds - second.duration_seconds
                ),
                absolute_rms_difference=abs(first.rms - second.rms),
                absolute_centroid_difference_hz=abs(
                    first.spectral_centroid_hz - second.spectral_centroid_hz
                ),
            )
        )
    return tuple(differences)


def classify_features(
    features: NDArray[np.float64],
    inputs: Sequence[ToneProbeInput],
    representation: FeatureRepresentation,
    config: ToneProbeConfig,
) -> tuple[tuple[ToneProbeMetric, ...], tuple[TonePrediction, ...]]:
    labels = np.asarray([LABELS.index(row.source.emotion) for row in inputs], dtype=np.int64)
    training = np.asarray([row.example.split == Split.TRAIN for row in inputs], dtype=np.bool_)
    if set(labels[training].tolist()) != set(range(len(LABELS))):
        raise ValueError("Selected training pairs must cover all four intended labels")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=config.maximum_iterations, class_weight="balanced", random_state=config.seed
        ),
    )
    model.fit(features[training], labels[training])
    predictions = np.asarray(model.predict(features), dtype=np.int64)
    majority = int(np.bincount(labels[training], minlength=len(LABELS)).argmax())
    metrics: list[ToneProbeMetric] = []
    for split in Split:
        selected = np.asarray([row.example.split == split for row in inputs], dtype=np.bool_)
        expected = labels[selected]
        predicted = predictions[selected]
        baseline = np.full(expected.shape, majority, dtype=np.int64)
        confusion = confusion_matrix(expected, predicted, labels=list(range(len(LABELS))))
        metrics.append(
            ToneProbeMetric(
                representation=representation,
                split=split,
                examples=int(selected.sum()),
                families=len(
                    {row.source.family_id for row in inputs if row.example.split == split}
                ),
                label_counts=tuple(
                    (emotion, int((expected == index).sum()))
                    for index, emotion in enumerate(LABELS)
                ),
                accuracy=float(accuracy_score(expected, predicted)),
                balanced_accuracy=float(balanced_accuracy_score(expected, predicted)),
                majority_label=LABELS[majority],
                majority_accuracy=float(accuracy_score(expected, baseline)),
                majority_balanced_accuracy=float(balanced_accuracy_score(expected, baseline)),
                confusion=tuple(tuple(int(value) for value in row) for row in confusion),
            )
        )
    rows = tuple(
        TonePrediction(
            example_id=row.example.example_id,
            base_id=row.source.base_id,
            family_id=row.source.family_id,
            split=row.example.split,
            intended_emotion=row.source.emotion,
            predicted_emotion=LABELS[int(prediction)],
            representation=representation,
        )
        for row, prediction in zip(inputs, predictions, strict=True)
    )
    return tuple(metrics), rows


def write_records(path: Path, rows: Sequence[Record]) -> None:
    path.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")


def run_tone_probe(
    config: ToneProbeConfig,
    examples: Sequence[Example],
    sources: Sequence[NeuEmotionalExampleSource],
) -> ToneProbeReport:
    started = time.monotonic()
    torch.set_num_threads(2)
    inputs = select_probe_inputs(examples, sources, config)
    missing = tuple(
        row.example.example_id for row in inputs if not row.example.feature_path.is_file()
    )
    if missing:
        raise ValueError(
            f"Probe requires all selected cached features; missing {len(missing)}: {missing[:4]}"
        )
    with threadpool_limits(limits=2):
        features = np.stack(
            [mean_std_feature(load_feature(row.example.feature_path)) for row in inputs]
        )
        measurements = tuple(acoustic_measurement(row) for row in inputs)
        differences = acoustic_pair_differences(measurements)
        classifier_started = time.monotonic()
        metrics, predictions = classify_features(
            features, inputs, FeatureRepresentation.RAW, config
        )
        if config.pair_centered:
            centered_metrics, centered_predictions = classify_features(
                pair_center_features(features, inputs),
                inputs,
                FeatureRepresentation.PAIR_CENTERED,
                config,
            )
            metrics += centered_metrics
            predictions += centered_predictions
        classifier_seconds = time.monotonic() - classifier_started
    config.output_directory.mkdir(parents=True, exist_ok=True)
    write_records(config.output_directory / "predictions.jsonl", predictions)
    write_records(config.output_directory / "acoustics.jsonl", measurements)
    write_records(config.output_directory / "pair_acoustic_differences.jsonl", differences)
    report = ToneProbeReport(
        configuration=config,
        manifest_sha256=digest(config.manifest),
        sidecar_sha256=digest(config.sidecar),
        selected_example_ids=tuple(row.example.example_id for row in inputs),
        feature_dimension=features.shape[1],
        classifier_seconds=classifier_seconds,
        total_wall_seconds=time.monotonic() - started,
        metrics=metrics,
        duration_seconds=distribution([row.duration_seconds for row in measurements]),
        rms=distribution([row.rms for row in measurements]),
        pair_rms_difference=distribution([row.absolute_rms_difference for row in differences]),
        pair_centroid_difference_hz=distribution(
            [row.absolute_centroid_difference_hz for row in differences]
        ),
        cpu_threads=2,
        limitations=(
            "Labels are intended NeuTTS delivery settings, not human-verified emotion gold.",
            "Family-held-out prediction can retain lexical/quota, duration or synthesis artifacts.",
            "Pair-centered features require both deliveries, "
            "so this diagnostic is not serving inference.",
            "RMS and centroid differences are acoustic diagnostics, "
            "not evidence of correct emotional perception.",
        ),
    )
    write_record(config.output_directory / "summary.json", report)
    lines = [
        "# Frozen Whisper intended-delivery probe",
        "",
        "CPU-only StandardScaler (fit on train only) + balanced LogisticRegression; "
        "frozen mean/std states, 1536 dimensions.",
        "",
        "| Representation | Split | Examples | Accuracy | Balanced accuracy | Majority accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {metric.representation.value} | {metric.split.value} | {metric.examples} | "
        f"{metric.accuracy:.4f} | {metric.balanced_accuracy:.4f} | {metric.majority_accuracy:.4f} |"
        for metric in metrics
    )
    lines.extend(
        (
            "",
            "Confusion rows/columns in summary.json: "
            + ", ".join(emotion.value for emotion in LABELS),
            "",
        )
    )
    lines.extend(report.limitations)
    (config.output_directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
