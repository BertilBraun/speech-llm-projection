"""Audit conversation, path, text-pair, and downloaded waveform overlap."""

import hashlib
import re
from collections import defaultdict
from pathlib import Path

from speech_projector.data import load_examples
from speech_projector.models import Example, Record, Split


class SplitOverlap(Record):
    left: Split
    right: Split
    shared_dialogue_ids: int
    shared_audio_paths: int
    identical_downloaded_waveforms: int
    exact_normalized_user_target_pairs: int


class LeakageAudit(Record):
    train_manifest_examples: int
    validation_examples: int
    test_examples: int
    hashed_downloaded_train_waveforms: int
    hashed_validation_waveforms: int
    hashed_test_waveforms: int
    overlaps: tuple[SplitOverlap, ...]
    waveform_hash_method: str
    interpretation: str


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while content := stream.read(1024 * 1024):
            digest.update(content)
    return digest.hexdigest()


def normalized_pair(example: Example) -> tuple[str, str]:
    return tuple(
        " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())
        for text in (example.user_text, example.target_text)
    )


def main() -> None:
    root = Path("/workspace/speech-projector/data")
    groups = {split: load_examples(root / "examples.jsonl", split) for split in Split}
    hashes: defaultdict[Split, set[str]] = defaultdict(set)
    hashed_counts: defaultdict[Split, int] = defaultdict(int)
    for split, examples in groups.items():
        for example in examples:
            if example.audio_path.exists():
                hashes[split].add(file_digest(example.audio_path))
                hashed_counts[split] += 1
    overlaps: list[SplitOverlap] = []
    for left, right in (
        (Split.TRAIN, Split.VALIDATION),
        (Split.TRAIN, Split.TEST),
        (Split.VALIDATION, Split.TEST),
    ):
        left_examples, right_examples = groups[left], groups[right]
        overlaps.append(
            SplitOverlap(
                left=left,
                right=right,
                shared_dialogue_ids=len(
                    {example.dialogue_id for example in left_examples}
                    & {example.dialogue_id for example in right_examples}
                ),
                shared_audio_paths=len(
                    {example.audio_path for example in left_examples}
                    & {example.audio_path for example in right_examples}
                ),
                identical_downloaded_waveforms=len(hashes[left] & hashes[right]),
                exact_normalized_user_target_pairs=len(
                    {normalized_pair(example) for example in left_examples}
                    & {normalized_pair(example) for example in right_examples}
                ),
            )
        )
    report = LeakageAudit(
        train_manifest_examples=len(groups[Split.TRAIN]),
        validation_examples=len(groups[Split.VALIDATION]),
        test_examples=len(groups[Split.TEST]),
        hashed_downloaded_train_waveforms=hashed_counts[Split.TRAIN],
        hashed_validation_waveforms=hashed_counts[Split.VALIDATION],
        hashed_test_waveforms=hashed_counts[Split.TEST],
        overlaps=tuple(overlaps),
        waveform_hash_method="SHA256 of complete WAV bytes for downloaded files only",
        interpretation=(
            "Conversation/path overlap covers the complete selected manifest. Byte-identical "
            "waveform overlap covers downloaded audio only. Coincidental repeated generic text "
            "pairs are reported separately from direct audio or whole-conversation leakage."
        ),
    )
    (root / "dataset_leakage.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json(indent=2))
    training_pairs = {normalized_pair(example) for example in groups[Split.TRAIN]}
    training_10k_pairs = {normalized_pair(example) for example in groups[Split.TRAIN][:10000]}
    for example in groups[Split.TEST]:
        pair = normalized_pair(example)
        if pair in training_pairs:
            print(
                "REPEATED_PAIR",
                example.example_id,
                example.dialogue_id,
                "in10k",
                pair in training_10k_pairs,
                "USER",
                example.user_text,
                "TARGET",
                example.target_text,
            )


if __name__ == "__main__":
    main()
