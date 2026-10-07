"""Train a deployable Neu intended-tone baseline using frozen cached speech features."""

import math
import pickle
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import sklearn
import torch
from numpy.typing import NDArray
from pydantic import Field, model_validator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.inventory_results import stable_digest, write_record
from speech_projector.acoustic_tone_probe import LABELS, ToneProbeInput, mean_std_feature
from speech_projector.cache import load_feature
from speech_projector.emotional_dataset import normalized_utterance
from speech_projector.journal import read_journal
from speech_projector.models import Example, FileArtifact, Record, Split
from speech_projector.overnight_data import NeuEmotionalExampleSource
from speech_projector.overnight_launcher import load_sources
from speech_projector.tts_pilot import PilotEmotion


class ToneClassifierConfig(Record):
    manifest: Path
    sidecar: Path
    output_directory: Path
    temporal_bins: int = Field(default=4, ge=1, le=16)
    regularization_c: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    maximum_iterations: int = Field(default=1000, ge=1)
    seed: int = Field(default=42, ge=0)
    cpu_threads: int = Field(default=2, ge=1, le=2)


class ToneProbability(Record):
    tone: PilotEmotion
    probability: float = Field(ge=0, le=1, allow_inf_nan=False)


class ToneClassifierPrediction(Record):
    example_id: str = Field(min_length=1)
    predicted_tone: PilotEmotion
    probabilities: tuple[ToneProbability, ...]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_probabilities(self) -> "ToneClassifierPrediction":
        if tuple(row.tone for row in self.probabilities) != LABELS:
            raise ValueError("Classifier probabilities must cover the four Neu labels in order")
        if not math.isclose(sum(row.probability for row in self.probabilities), 1, abs_tol=1e-6):
            raise ValueError("Classifier probabilities must sum to one")
        maximum = max(self.probabilities, key=lambda row: row.probability)
        if self.predicted_tone != maximum.tone or not math.isclose(
            self.confidence, maximum.probability, abs_tol=1e-9
        ):
            raise ValueError("Predicted tone/confidence must match the maximum probability")
        return self


class ToneClassifierMetric(Record):
    split: Split
    examples: int
    bases: int
    families: int
    labels: tuple[PilotEmotion, ...]
    label_counts: tuple[int, ...]
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    confusion: tuple[tuple[int, ...], ...]
    both_deliveries_correct_pairs: int
    both_deliveries_correct_fraction: float
    predicted_distinct_pairs: int
    predicted_distinct_fraction: float


class ToneClassifierReport(Record):
    configuration: ToneClassifierConfig
    manifest: FileArtifact
    sidecar: FileArtifact
    features: tuple[FileArtifact, ...]
    selected_example_ids: tuple[str, ...]
    model: FileArtifact
    prediction_files: tuple[FileArtifact, ...]
    metrics: tuple[ToneClassifierMetric, ...]
    feature_dimension: int
    feature_loading_seconds: float
    fitting_seconds: float
    total_wall_seconds: float
    sklearn_version: str
    source_git_commit: str
    limitations: tuple[str, ...]


def artifact(path: Path) -> FileArtifact:
    size, digest = stable_digest(path)
    return FileArtifact(path=path, source_path=path.resolve(), bytes=size, sha256=digest)


def select_classifier_inputs(
    examples: Sequence[Example], sources: Sequence[NeuEmotionalExampleSource]
) -> tuple[ToneProbeInput, ...]:
    examples_by_id = {row.example_id: row for row in examples}
    if len(examples_by_id) != len(examples):
        raise ValueError("Classifier manifest repeats example IDs")
    if len({row.example_id for row in sources}) != len(sources):
        raise ValueError("Classifier sidecar repeats example IDs")
    rows: list[ToneProbeInput] = []
    families: dict[str, Split] = {}
    words: dict[str, Split] = {}
    paths: dict[Path, Split] = {}
    pairs: dict[str, list[ToneProbeInput]] = {}
    for source in sources:
        if source.emotion not in LABELS or source.example_id not in examples_by_id:
            raise ValueError("Neu classifier source has an unsupported label or absent example")
        example = examples_by_id[source.example_id]
        literal_key = normalized_utterance(example.user_text)
        for grouping, key in ((families, source.family_id), (words, literal_key)):
            if grouping.setdefault(key, example.split) != example.split:
                raise ValueError("Classifier family or exact literal text crosses splits")
        if paths.setdefault(example.feature_path, example.split) != example.split:
            raise ValueError("Classifier cached feature path crosses splits")
        if paths.setdefault(example.audio_path, example.split) != example.split:
            raise ValueError("Classifier audio path crosses splits")
        row = ToneProbeInput(example, source)
        pairs.setdefault(source.base_id, []).append(row)
        rows.append(row)
    if not rows:
        raise ValueError("No Neu classifier examples found")
    for base, pair in pairs.items():
        if len(pair) != 2:
            raise ValueError(f"Classifier requires a complete two-delivery pair: {base}")
        first, second = pair
        if (
            first.example.split != second.example.split
            or first.example.user_text != second.example.user_text
            or first.source.family_id != second.source.family_id
            or first.source.emotion == second.source.emotion
        ):
            raise ValueError(f"Classifier pair literal words/family/split/labels differ: {base}")
    return tuple(rows)


def classifier_feature(hidden: torch.Tensor, temporal_bins: int) -> NDArray[np.float64]:
    pooled = mean_std_feature(hidden)
    if temporal_bins < 1 or hidden.shape[0] < temporal_bins:
        raise ValueError("Cached feature sequence must contain at least one frame per bin")
    pieces = torch.tensor_split(hidden.float(), temporal_bins, dim=0)
    temporal = np.concatenate([piece.mean(dim=0).numpy().astype(np.float64) for piece in pieces])
    return np.concatenate((pooled, temporal))


def fit_classifier(
    features: NDArray[np.float64], inputs: Sequence[ToneProbeInput], config: ToneClassifierConfig
) -> Pipeline:
    training = np.asarray([row.example.split == Split.TRAIN for row in inputs], dtype=np.bool_)
    labels = np.asarray([LABELS.index(row.source.emotion) for row in inputs], dtype=np.int64)
    if set(labels[training].tolist()) != set(range(len(LABELS))):
        raise ValueError("Training split must cover all four intended delivery labels")
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=config.regularization_c,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=config.maximum_iterations,
            random_state=config.seed,
        ),
    )
    classifier.fit(features[training], labels[training])
    return classifier


def predict_features(
    classifier: Pipeline, features: NDArray[np.float64], identities: Sequence[str]
) -> tuple[ToneClassifierPrediction, ...]:
    if len(features) != len(identities) or len(set(identities)) != len(identities):
        raise ValueError("Prediction feature rows and unique IDs must correspond")
    probabilities: NDArray[np.float64] = np.asarray(classifier.predict_proba(features))
    if probabilities.shape != (len(identities), len(LABELS)):
        raise ValueError("Classifier probability matrix does not cover the four-label vocabulary")
    return tuple(
        ToneClassifierPrediction(
            example_id=identity,
            predicted_tone=LABELS[int(values.argmax())],
            probabilities=tuple(
                ToneProbability(tone=tone, probability=float(value))
                for tone, value in zip(LABELS, values, strict=True)
            ),
            confidence=float(values.max()),
        )
        for identity, values in zip(identities, probabilities, strict=True)
    )


def classifier_metric(
    inputs: Sequence[ToneProbeInput], predictions: Sequence[ToneClassifierPrediction], split: Split
) -> ToneClassifierMetric:
    selected = tuple(row for row in inputs if row.example.split == split)
    by_id = {row.example_id: row for row in predictions}
    target = np.asarray([LABELS.index(row.source.emotion) for row in selected], dtype=np.int64)
    predicted = np.asarray(
        [LABELS.index(by_id[row.example.example_id].predicted_tone) for row in selected],
        dtype=np.int64,
    )
    pairs: dict[str, list[ToneProbeInput]] = {}
    for row in selected:
        pairs.setdefault(row.source.base_id, []).append(row)
    correct_pairs = sum(
        all(by_id[row.example.example_id].predicted_tone == row.source.emotion for row in pair)
        for pair in pairs.values()
    )
    distinct_pairs = sum(
        len({by_id[row.example.example_id].predicted_tone for row in pair}) == 2
        for pair in pairs.values()
    )
    matrix: NDArray[np.int64] = np.asarray(confusion_matrix(target, predicted, labels=range(4)))
    counts = Counter(target.tolist())
    return ToneClassifierMetric(
        split=split,
        examples=len(selected),
        bases=len(pairs),
        families=len({row.source.family_id for row in selected}),
        labels=LABELS,
        label_counts=tuple(counts[index] for index in range(4)),
        accuracy=float(accuracy_score(target, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(target, predicted)),
        macro_f1=float(
            f1_score(target, predicted, labels=range(4), average="macro", zero_division=0)
        ),
        confusion=tuple(tuple(int(value) for value in row) for row in matrix),
        both_deliveries_correct_pairs=correct_pairs,
        both_deliveries_correct_fraction=correct_pairs / len(pairs),
        predicted_distinct_pairs=distinct_pairs,
        predicted_distinct_fraction=distinct_pairs / len(pairs),
    )


def train_tone_classifier(
    config: ToneClassifierConfig, source_git_commit: str
) -> ToneClassifierReport:
    report_path = config.output_directory / "report.json"
    if report_path.exists():
        raise ValueError("Completed classifier output already exists; use its saved predictions")
    started = time.perf_counter()
    torch.set_num_threads(config.cpu_threads)
    examples = read_journal(config.manifest, Example)
    sources: list[NeuEmotionalExampleSource] = []
    for source in load_sources(config.sidecar):
        match source:
            case NeuEmotionalExampleSource():
                sources.append(source)
            case _:
                continue
    inputs = select_classifier_inputs(examples, sources)
    missing = tuple(
        row.example.feature_path for row in inputs if not row.example.feature_path.is_file()
    )
    if missing:
        raise ValueError(f"Classifier requires existing cached features: {missing[0]}")
    loading_started = time.perf_counter()
    features = np.stack(
        [
            classifier_feature(load_feature(row.example.feature_path), config.temporal_bins)
            for row in inputs
        ]
    )
    feature_artifacts = tuple(artifact(row.example.feature_path) for row in inputs)
    loading_seconds = time.perf_counter() - loading_started
    with threadpool_limits(limits=config.cpu_threads):
        fit_started = time.perf_counter()
        classifier = fit_classifier(features, inputs, config)
        fitting_seconds = time.perf_counter() - fit_started
    config.output_directory.mkdir(parents=True, exist_ok=True)
    write_record(config.output_directory / "config.json", config)
    model_path = config.output_directory / "classifier.pkl"
    with model_path.open("wb") as stream:
        pickle.dump(classifier, stream, protocol=pickle.HIGHEST_PROTOCOL)
    metrics: list[ToneClassifierMetric] = []
    prediction_files: list[FileArtifact] = []
    for split in (Split.TRAIN, Split.VALIDATION, Split.TEST):
        selected_indices = [index for index, row in enumerate(inputs) if row.example.split == split]
        with threadpool_limits(limits=config.cpu_threads):
            predictions = predict_features(
                classifier,
                features[selected_indices],
                [inputs[index].example.example_id for index in selected_indices],
            )
        metrics.append(classifier_metric(inputs, predictions, split))
        selected_ids = {row.example.example_id for row in inputs if row.example.split == split}
        path = config.output_directory / f"{split.value}_predictions.jsonl"
        path.write_text(
            "\n".join(
                row.model_dump_json() for row in predictions if row.example_id in selected_ids
            )
            + "\n",
            encoding="utf-8",
        )
        prediction_files.append(artifact(path))
    report = ToneClassifierReport(
        configuration=config,
        manifest=artifact(config.manifest),
        sidecar=artifact(config.sidecar),
        features=feature_artifacts,
        selected_example_ids=tuple(row.example.example_id for row in inputs),
        model=artifact(model_path),
        prediction_files=tuple(prediction_files),
        metrics=tuple(metrics),
        feature_dimension=features.shape[1],
        feature_loading_seconds=loading_seconds,
        fitting_seconds=fitting_seconds,
        total_wall_seconds=time.perf_counter() - started,
        sklearn_version=sklearn.__version__,
        source_git_commit=source_git_commit,
        limitations=(
            "Training and StandardScaler fit only TRAIN; fixed hyperparameters, no TEST selection.",
            "Raw mean/std and temporal-bin means only; no text, duration, "
            "signal statistics or label input.",
            "Intended synthetic delivery prediction is not human emotion recognition.",
            "Single Paul voice/Neu vocoder can encode synthetic signatures; "
            "natural-speech generalization is unknown.",
            "Old Qwen labels and audio are not used or silently aliased into Neu label classes.",
        ),
    )
    write_record(report_path, report)
    return report
