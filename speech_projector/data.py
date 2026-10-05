"""DeepDialogue metadata, conversation construction, and selective audio retrieval."""

import argparse
import hashlib
import math
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pyarrow.parquet as parquet
import soundfile as soundfile
from numpy.typing import NDArray
from pydantic import Field
from scipy.signal import resample_poly

from speech_projector.models import Example, Record, Role, Split, Turn

DATASET_URL = "https://huggingface.co/datasets/SALT-Research/DeepDialogue-xtts/resolve/main/"
SOURCE_COLUMNS = (
    "conversation_id",
    "model_dir",
    "turn_index",
    "speaker",
    "text",
    "domain",
    "emotion",
    "segment_audio_path",
    "audio_duration",
    "audio_cleaned_text",
    "audio_original_text",
    "audio_substituted_text",
)


class SourceSpeaker(str, Enum):
    FIRST = "LLM1"
    SECOND = "LLM2"


class SourceTurn(Record):
    conversation_id: str
    model_dir: str
    turn_index: int
    speaker: SourceSpeaker
    text: str
    domain: str
    emotion: str
    segment_audio_path: str | None
    audio_duration: float | None
    audio_cleaned_text: str | None
    audio_original_text: str | None = None
    audio_substituted_text: str | None = None

    @property
    def dialogue_id(self) -> str:
        return f"{self.model_dir}/{self.conversation_id}"


class DataConfig(Record):
    root: Path
    seed: int = 42
    history_turns: int = Field(default=2, ge=0)
    max_train: int = Field(default=30000, gt=0)
    validation_examples: int = Field(default=128, gt=0)
    test_examples: int = Field(default=128, gt=0)
    minimum_duration: float = 0.4
    maximum_duration: float = 30.0
    minimum_words: int = 2
    download_workers: int = 12


class Distribution(Record):
    minimum: float
    p10: float
    median: float
    p90: float
    p99: float
    maximum: float
    mean: float


class FilterCounts(Record):
    missing_audio: int
    invalid_duration: int
    empty_text: int
    nonalternating: int


class DatasetReport(Record):
    source_rows: int
    dialogues: int
    domains: int
    model_pairs: int
    usable_pairs: int
    filters: FilterCounts
    train_examples: int
    validation_examples: int
    test_examples: int
    duration: Distribution
    user_words: Distribution
    target_words: Distribution
    dialogue_turns: Distribution
    domains_selected: tuple[tuple[str, int], ...]
    random_dialogues: tuple[tuple[SourceTurn, ...], ...]
    source_columns: tuple[str, ...]
    split_method: str


@dataclass(frozen=True)
class DownloadResult:
    example_id: str
    success: bool
    bytes_downloaded: int
    error: str | None


def stable_digest(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def dialogue_split(dialogue_id: str, seed: int) -> Split:
    bucket = int(stable_digest(dialogue_id, seed)[:8], 16) % 10000
    if bucket < 9000:
        return Split.TRAIN
    if bucket < 9500:
        return Split.VALIDATION
    return Split.TEST


def distribution(values: list[float]) -> Distribution:
    quantiles = np.quantile(values, [0, 0.1, 0.5, 0.9, 0.99, 1])
    return Distribution(
        minimum=float(quantiles[0]),
        p10=float(quantiles[1]),
        median=float(quantiles[2]),
        p90=float(quantiles[3]),
        p99=float(quantiles[4]),
        maximum=float(quantiles[5]),
        mean=float(np.mean(values)),
    )


def turn_order(turn: SourceTurn) -> int:
    return turn.turn_index


def example_order(example: Example, seed: int) -> str:
    return stable_digest(example.example_id, seed)


def load_source(path: Path) -> tuple[list[SourceTurn], tuple[str, ...]]:
    table = parquet.read_table(path)
    columns = tuple(table.column_names)
    selected = table.select(SOURCE_COLUMNS)
    return [SourceTurn.model_validate(row) for row in selected.to_pylist()], columns


def build_examples(
    turns: list[SourceTurn], configuration: DataConfig
) -> tuple[list[Example], DatasetReport]:
    dialogues: defaultdict[str, list[SourceTurn]] = defaultdict(list)
    for turn in turns:
        dialogues[turn.dialogue_id].append(turn)
    examples: list[Example] = []
    failures: Counter[str] = Counter()
    for dialogue_id, conversation in dialogues.items():
        conversation.sort(key=turn_order)
        for position, (user, assistant) in enumerate(
            zip(conversation, conversation[1:], strict=False)
        ):
            if user.speaker != SourceSpeaker.FIRST:
                continue
            if (
                assistant.speaker != SourceSpeaker.SECOND
                or assistant.turn_index != user.turn_index + 1
            ):
                failures["nonalternating"] += 1
                continue
            if user.segment_audio_path is None or user.audio_duration is None:
                failures["missing_audio"] += 1
                continue
            if (
                not configuration.minimum_duration
                <= user.audio_duration
                <= configuration.maximum_duration
            ):
                failures["invalid_duration"] += 1
                continue
            if (
                min(len(user.text.split()), len(assistant.text.split()))
                < configuration.minimum_words
            ):
                failures["empty_text"] += 1
                continue
            history_start = max(0, position - configuration.history_turns)
            history = conversation[history_start:position]
            example_id = hashlib.sha256(f"{dialogue_id}:{user.turn_index}".encode()).hexdigest()[
                :20
            ]
            examples.append(
                Example(
                    example_id=example_id,
                    dialogue_id=dialogue_id,
                    split=dialogue_split(dialogue_id, configuration.seed),
                    history=tuple(
                        Turn(
                            role=Role.USER
                            if turn.speaker == SourceSpeaker.FIRST
                            else Role.ASSISTANT,
                            text=turn.text,
                        )
                        for turn in history
                    ),
                    user_text=user.text,
                    target_text=assistant.text,
                    audio_path=configuration.root / "audio" / user.segment_audio_path,
                    duration=user.audio_duration,
                    domain=user.domain,
                    emotion=user.emotion,
                    feature_path=configuration.root
                    / "features"
                    / example_id[:2]
                    / f"{example_id}.pt",
                )
            )
    examples.sort(key=partial(example_order, seed=configuration.seed))
    selected: list[Example] = []
    split_limits = (
        (Split.TRAIN, configuration.max_train),
        (Split.VALIDATION, configuration.validation_examples),
        (Split.TEST, configuration.test_examples),
    )
    for split, limit in split_limits:
        selected.extend([example for example in examples if example.split == split][:limit])
    generator = random.Random(configuration.seed)
    sample_ids = generator.sample(sorted(dialogues), min(5, len(dialogues)))
    report = DatasetReport(
        source_rows=len(turns),
        dialogues=len(dialogues),
        domains=len({turn.domain for turn in turns}),
        model_pairs=len({turn.model_dir for turn in turns}),
        usable_pairs=len(examples),
        filters=FilterCounts(
            missing_audio=failures["missing_audio"],
            invalid_duration=failures["invalid_duration"],
            empty_text=failures["empty_text"],
            nonalternating=failures["nonalternating"],
        ),
        train_examples=sum(example.split == Split.TRAIN for example in selected),
        validation_examples=sum(example.split == Split.VALIDATION for example in selected),
        test_examples=sum(example.split == Split.TEST for example in selected),
        duration=distribution([example.duration for example in examples]),
        user_words=distribution([float(len(example.user_text.split())) for example in examples]),
        target_words=distribution(
            [float(len(example.target_text.split())) for example in examples]
        ),
        dialogue_turns=distribution(
            [float(len(conversation)) for conversation in dialogues.values()]
        ),
        domains_selected=tuple(Counter(example.domain for example in selected).most_common()),
        random_dialogues=tuple(tuple(dialogues[dialogue_id]) for dialogue_id in sample_ids),
        source_columns=SOURCE_COLUMNS,
        split_method=(
            "SHA256(seed:model_dir/conversation_id) modulo10000: 90%train/5%validation/5%test; "
            "nested examples ordered by seeded SHA256 example ID"
        ),
    )
    return selected, report


def save_examples(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    with partial.open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(example.model_dump_json() + "\n")
    partial.replace(path)


def load_examples(path: Path, split: Split, limit: int | None = None) -> list[Example]:
    with path.open(encoding="utf-8") as stream:
        examples = [Example.model_validate_json(line) for line in stream]
    selected = [example for example in examples if example.split == split]
    return selected if limit is None else selected[:limit]


def load_audio(path: Path, sample_rate: int = 16000) -> NDArray[np.float32]:
    waveform, source_rate = soundfile.read(path, dtype="float32", always_2d=True)
    mono = waveform.mean(axis=1)
    if source_rate != sample_rate:
        divisor = math.gcd(source_rate, sample_rate)
        mono = resample_poly(mono, sample_rate // divisor, source_rate // divisor).astype(
            np.float32
        )
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError(f"Invalid audio samples: {path}")
    return mono


def download_audio(example: Example, root: Path) -> DownloadResult:
    if example.audio_path.exists() and example.audio_path.stat().st_size > 44:
        return DownloadResult(example.example_id, True, 0, None)
    relative = example.audio_path.relative_to(root / "audio").as_posix()
    destination = example.audio_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    error: str | None = None
    for attempt in range(5):
        try:
            with urlopen(DATASET_URL + "data/" + relative, timeout=90) as response:
                content = response.read()
            partial.write_bytes(content)
            soundfile.info(partial)
            partial.replace(destination)
            return DownloadResult(example.example_id, True, len(content), None)
        except (HTTPError, URLError, TimeoutError, RuntimeError, OSError) as exception:
            error = str(exception)
            time.sleep(min(16, 2**attempt))
    return DownloadResult(example.example_id, False, 0, error)


def prepare(configuration: DataConfig, download_train: int) -> None:
    configuration.root.mkdir(parents=True, exist_ok=True)
    metadata = configuration.root / "metadata.parquet"
    if not metadata.exists():
        with urlopen(DATASET_URL + "data/train-00000-of-00001.parquet", timeout=180) as response:
            metadata.write_bytes(response.read())
    turns, columns = load_source(metadata)
    examples, report = build_examples(turns, configuration)
    report = report.model_copy(update={"source_columns": columns})
    save_examples(configuration.root / "examples.jsonl", examples)
    (configuration.root / "dataset_report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    print(
        f"Prepared train={report.train_examples}, val={report.validation_examples}, "
        f"test={report.test_examples}, usable={report.usable_pairs}",
        flush=True,
    )
    selected = [example for example in examples if example.split != Split.TRAIN] + [
        example for example in examples if example.split == Split.TRAIN
    ][:download_train]
    started = time.monotonic()
    failures: list[DownloadResult] = []
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=configuration.download_workers) as executor:
        for index, result in enumerate(
            executor.map(partial(download_audio, root=configuration.root), selected), 1
        ):
            total_bytes += result.bytes_downloaded
            if not result.success:
                failures.append(result)
            if index % 50 == 0:
                print(
                    f"Audio {index}/{len(selected)} bytes={total_bytes} "
                    f"elapsed={time.monotonic() - started:.1f}s failures={len(failures)}",
                    flush=True,
                )
    print(
        f"Download complete seconds={time.monotonic() - started:.1f} "
        f"bytes={total_bytes} failures={len(failures)}",
        flush=True,
    )
    if failures:
        print(failures, flush=True)
        raise ValueError(f"Failed downloading {len(failures)} utterances")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--download-train", type=int, default=1000)
    parser.add_argument("--max-train", type=int, default=30000)
    parser.add_argument("--history-turns", type=int, default=2)
    parser.add_argument("--workers", type=int, default=12)
    arguments = parser.parse_args()
    prepare(
        DataConfig(
            root=arguments.root,
            max_train=arguments.max_train,
            history_turns=arguments.history_turns,
            download_workers=arguments.workers,
        ),
        arguments.download_train,
    )


if __name__ == "__main__":
    main()
