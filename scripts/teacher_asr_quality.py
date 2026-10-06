"""CPU-only lexical ASR evaluation against selected cleaned synthesis transcripts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from jiwer import process_words

from scripts.audit_synthesis_alignment import normalize
from scripts.package_results import FileArtifact, file_digest
from scripts.summarize_teacher_targets import HistoryGroup
from speech_projector.data import Distribution, distribution, load_examples
from speech_projector.models import AsrTranscript, Example, Record, Split


class AsrEdits(Record):
    hits: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def reference_words(self) -> int:
        return self.hits + self.substitutions + self.deletions


class AsrObservation(Record):
    example_id: str
    dialogue_id: str
    split: Split
    domain: str
    history: HistoryGroup
    reference: str
    recognized: str
    edits: AsrEdits
    word_error_rate: float


class AsrAggregate(Record):
    examples: int
    edits: AsrEdits
    reference_words: int
    word_error_rate: float
    example_word_error_rate: Distribution


class SplitAsrQuality(Record):
    split: Split
    metrics: AsrAggregate


class HistoryAsrQuality(Record):
    split: Split
    history: HistoryGroup
    metrics: AsrAggregate


class DomainAsrQuality(Record):
    split: Split
    domain: str
    metrics: AsrAggregate


class TeacherAsrQuality(Record):
    source_manifest: FileArtifact
    asr_transcripts: FileArtifact
    splits: tuple[SplitAsrQuality, ...]
    histories: tuple[HistoryAsrQuality, ...]
    domains: tuple[DomainAsrQuality, ...]
    normalization: str
    interpretation: str


@dataclass(frozen=True)
class AsrQualityConfiguration:
    manifest: Path
    transcripts: Path
    output: Path


def observations(
    examples: Sequence[Example], transcripts: Sequence[AsrTranscript]
) -> tuple[AsrObservation, ...]:
    selected = tuple(example for example in examples if example.split != Split.TRAIN)
    if not selected:
        raise ValueError("ASR quality requires heldout examples")
    if len({example.example_id for example in selected}) != len(selected):
        raise ValueError("Selected heldout examples contain duplicate identifiers")
    recognized = {transcript.example_id: transcript for transcript in transcripts}
    if len(recognized) != len(transcripts):
        raise ValueError("ASR journal contains duplicate identifiers")
    missing = {example.example_id for example in selected} - recognized.keys()
    if missing:
        raise ValueError(f"ASR extraction is pending for {len(missing)} selected heldout examples")
    records: list[AsrObservation] = []
    for example in selected:
        reference = normalize(example.user_text)
        hypothesis = normalize(recognized[example.example_id].text)
        if not reference:
            raise ValueError(f"Normalized synthesis reference is empty: {example.example_id}")
        alignment = process_words(reference, hypothesis)
        records.append(
            AsrObservation(
                example_id=example.example_id,
                dialogue_id=example.dialogue_id,
                split=example.split,
                domain=example.domain,
                history=HistoryGroup.PRESENT if example.history else HistoryGroup.EMPTY,
                reference=reference,
                recognized=hypothesis,
                edits=AsrEdits(
                    hits=alignment.hits,
                    substitutions=alignment.substitutions,
                    deletions=alignment.deletions,
                    insertions=alignment.insertions,
                ),
                word_error_rate=alignment.wer,
            )
        )
    return tuple(records)


def aggregate(records: Sequence[AsrObservation]) -> AsrAggregate:
    if not records:
        raise ValueError("ASR aggregation requires observations")
    edits = AsrEdits(
        hits=sum(record.edits.hits for record in records),
        substitutions=sum(record.edits.substitutions for record in records),
        deletions=sum(record.edits.deletions for record in records),
        insertions=sum(record.edits.insertions for record in records),
    )
    return AsrAggregate(
        examples=len(records),
        edits=edits,
        reference_words=edits.reference_words,
        word_error_rate=(edits.substitutions + edits.deletions + edits.insertions)
        / edits.reference_words,
        example_word_error_rate=distribution([record.word_error_rate for record in records]),
    )


def artifact(path: Path) -> FileArtifact:
    return FileArtifact(
        path=path, source_path=path, bytes=path.stat().st_size, sha256=file_digest(path)
    )


def load_asr_snapshot(path: Path) -> tuple[AsrTranscript, ...]:
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        raise ValueError("ASR journal has an incomplete final line; wait for extraction to finish")
    return tuple(AsrTranscript.model_validate_json(line) for line in content.splitlines())


def evaluate(configuration: AsrQualityConfiguration) -> TeacherAsrQuality:
    examples = [
        example
        for split in (Split.VALIDATION, Split.TEST)
        for example in load_examples(configuration.manifest, split)
    ]
    records = observations(examples, load_asr_snapshot(configuration.transcripts))
    splits: list[SplitAsrQuality] = []
    histories: list[HistoryAsrQuality] = []
    domains: list[DomainAsrQuality] = []
    for split in (Split.VALIDATION, Split.TEST):
        selected = tuple(record for record in records if record.split == split)
        splits.append(SplitAsrQuality(split=split, metrics=aggregate(selected)))
        for history in HistoryGroup:
            subset = tuple(record for record in selected if record.history == history)
            if subset:
                histories.append(
                    HistoryAsrQuality(split=split, history=history, metrics=aggregate(subset))
                )
        for domain in sorted({record.domain for record in selected}):
            domains.append(
                DomainAsrQuality(
                    split=split,
                    domain=domain,
                    metrics=aggregate(
                        tuple(record for record in selected if record.domain == domain)
                    ),
                )
            )
    report = TeacherAsrQuality(
        source_manifest=artifact(configuration.manifest),
        asr_transcripts=artifact(configuration.transcripts),
        splits=tuple(splits),
        histories=tuple(histories),
        domains=tuple(domains),
        normalization=(
            "Unicode lowercase; replace non-word/non-whitespace characters with spaces; "
            "collapse whitespace"
        ),
        interpretation=(
            "Pooled (substitutions+deletions+insertions)/reference words against "
            "audio_cleaned_text. Lexical ASR quality, not assistant response quality. "
            "History strata refer only to visible dataset turns."
        ),
    )
    configuration.output.mkdir(parents=True, exist_ok=True)
    (configuration.output / "asr_quality.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    (configuration.output / "asr_observations.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8"
    )
    lines = [
        "# Heldout ASR against cleaned synthesis text",
        "",
        report.normalization,
        "",
        report.interpretation,
        "",
        "| Split | Examples | Reference words | WER | Substitutions | Deletions | Insertions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in report.splits:
        metrics = entry.metrics
        lines.append(
            f"| {entry.split.value} | {metrics.examples} | {metrics.reference_words} | "
            f"{metrics.word_error_rate:.4f} | {metrics.edits.substitutions} | "
            f"{metrics.edits.deletions} | {metrics.edits.insertions} |"
        )
    (configuration.output / "asr_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = evaluate(
        AsrQualityConfiguration(arguments.manifest, arguments.transcripts, arguments.output)
    )
    print(
        tuple(
            (entry.split.value, entry.metrics.examples, entry.metrics.word_error_rate)
            for entry in report.splits
        )
    )


if __name__ == "__main__":
    main()
