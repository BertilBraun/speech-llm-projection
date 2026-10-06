"""Audit dialogue-to-synthesis alignment without changing the selected manifest."""

from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import jiwer
from pydantic import Field, model_validator

from speech_projector.cache import load_asr
from speech_projector.data import SourceSpeaker, SourceTurn, load_examples, load_source
from speech_projector.models import AsrTranscript, Example, Record, Split


class AlignmentKind(str, Enum):
    EXACT = "exact"
    FORMATTING = "only_case_punctuation_symbols_or_whitespace"
    LEXICAL = "normalized_word_sequence_differs"


class SynthesisAlignment(Record):
    example: Example
    source: SourceTurn
    split_index: int = Field(ge=0)
    turn_vs_audio_original: AlignmentKind
    original_vs_substituted: AlignmentKind
    turn_vs_audio_cleaned: AlignmentKind
    audio_original_vs_target: AlignmentKind

    @model_validator(mode="after")
    def validate_source(self) -> SynthesisAlignment:
        if (
            self.source.audio_original_text is None
            or self.source.audio_substituted_text is None
            or self.source.audio_cleaned_text is None
        ):
            raise ValueError("Selected audio must have original/substituted/cleaned synthesis text")
        if self.example.user_text != self.source.text:
            raise ValueError("Manifest user text differs from its canonical source turn")
        return self

    @property
    def original_text(self) -> str:
        assert self.source.audio_original_text is not None
        return self.source.audio_original_text

    @property
    def substituted_text(self) -> str:
        assert self.source.audio_substituted_text is not None
        return self.source.audio_substituted_text

    @property
    def cleaned_text(self) -> str:
        assert self.source.audio_cleaned_text is not None
        return self.source.audio_cleaned_text


class AlignmentSummary(Record):
    split: Split
    examples: int
    turn_original_exact: int
    turn_original_formatting_only: int
    turn_original_lexical_differences: int
    mismatched_audio_matches_assistant_target: int
    original_substituted_raw_changes: int
    original_substituted_lexical_differences: int
    turn_cleaned_raw_changes: int
    turn_cleaned_lexical_differences: int


class HeldoutTextSummary(Record):
    alignment: AlignmentSummary
    asr_word_error_rate_vs_turn_text: float
    asr_word_error_rate_vs_cleaned_text: float


class SynthesisAlignmentAudit(Record):
    manifest_sha256: str
    metadata_sha256: str
    dataset_card_url: str
    documented_synthesis_field: str
    normalization: str
    training: AlignmentSummary
    heldout: tuple[HeldoutTextSummary, ...]
    source_original_substituted_raw_changes: int
    source_original_substituted_normalized_word_changes: int
    source_substitution_examples: tuple[SourceTurn, ...]
    recommended_baseline_wording: str


@dataclass(frozen=True)
class AuditConfiguration:
    data_root: Path
    output_root: Path
    training_examples: int


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def difference(left: str, right: str) -> AlignmentKind:
    if left == right:
        return AlignmentKind.EXACT
    return (
        AlignmentKind.FORMATTING if normalize(left) == normalize(right) else AlignmentKind.LEXICAL
    )


def source_example_id(turn: SourceTurn) -> str:
    return hashlib.sha256(f"{turn.dialogue_id}:{turn.turn_index}".encode()).hexdigest()[:20]


def compare(example: Example, source: SourceTurn, split_index: int) -> SynthesisAlignment:
    if (
        source.audio_original_text is None
        or source.audio_substituted_text is None
        or source.audio_cleaned_text is None
    ):
        raise ValueError(f"Missing synthesis text: {example.example_id}")
    return SynthesisAlignment(
        example=example,
        source=source,
        split_index=split_index,
        turn_vs_audio_original=difference(example.user_text, source.audio_original_text),
        original_vs_substituted=difference(
            source.audio_original_text, source.audio_substituted_text
        ),
        turn_vs_audio_cleaned=difference(example.user_text, source.audio_cleaned_text),
        audio_original_vs_target=difference(source.audio_original_text, example.target_text),
    )


def summarize(split: Split, records: Sequence[SynthesisAlignment]) -> AlignmentSummary:
    return AlignmentSummary(
        split=split,
        examples=len(records),
        turn_original_exact=sum(
            record.turn_vs_audio_original == AlignmentKind.EXACT for record in records
        ),
        turn_original_formatting_only=sum(
            record.turn_vs_audio_original == AlignmentKind.FORMATTING for record in records
        ),
        turn_original_lexical_differences=sum(
            record.turn_vs_audio_original == AlignmentKind.LEXICAL for record in records
        ),
        mismatched_audio_matches_assistant_target=sum(
            record.turn_vs_audio_original == AlignmentKind.LEXICAL
            and record.audio_original_vs_target != AlignmentKind.LEXICAL
            for record in records
        ),
        original_substituted_raw_changes=sum(
            record.original_vs_substituted != AlignmentKind.EXACT for record in records
        ),
        original_substituted_lexical_differences=sum(
            record.original_vs_substituted == AlignmentKind.LEXICAL for record in records
        ),
        turn_cleaned_raw_changes=sum(
            record.turn_vs_audio_cleaned != AlignmentKind.EXACT for record in records
        ),
        turn_cleaned_lexical_differences=sum(
            record.turn_vs_audio_cleaned == AlignmentKind.LEXICAL for record in records
        ),
    )


def heldout_summary(
    split: Split, records: Sequence[SynthesisAlignment], transcripts: Mapping[str, AsrTranscript]
) -> HeldoutTextSummary:
    recognized = [normalize(transcripts[record.example.example_id].text) for record in records]
    return HeldoutTextSummary(
        alignment=summarize(split, records),
        asr_word_error_rate_vs_turn_text=jiwer.wer(
            [normalize(record.example.user_text) for record in records], recognized
        ),
        asr_word_error_rate_vs_cleaned_text=jiwer.wer(
            [normalize(record.cleaned_text) for record in records], recognized
        ),
    )


def write_alignments(path: Path, records: Sequence[SynthesisAlignment]) -> None:
    path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8"
    )


def load_alignments(path: Path) -> tuple[SynthesisAlignment, ...]:
    return tuple(
        SynthesisAlignment.model_validate_json(line) for line in path.read_bytes().splitlines()
    )


def readable_report(
    audit: SynthesisAlignmentAudit,
    heldout: Sequence[SynthesisAlignment],
    transcripts: Mapping[str, AsrTranscript],
) -> str:
    summaries = [audit.training] + [record.alignment for record in audit.heldout]
    lines = [
        "# Dialogue/synthesis alignment audit",
        "",
        f"Manifest SHA256: `{audit.manifest_sha256}`.",
        "",
        f"Metadata SHA256: `{audit.metadata_sha256}`.",
        "",
        audit.documented_synthesis_field + f" [Dataset card]({audit.dataset_card_url}).",
        "",
        audit.normalization,
        "",
        "| Split | N | Original exact | Formatting only | Material alignment mismatches | "
        "Mismatched audio equals target | Synthesis substitutions | Cleaned lexical differences |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in summaries:
        lines.append(
            f"| {record.split.value} | {record.examples:,} | {record.turn_original_exact:,} | "
            f"{record.turn_original_formatting_only} | "
            f"{record.turn_original_lexical_differences} | "
            f"{record.mismatched_audio_matches_assistant_target} | "
            f"{record.original_substituted_lexical_differences} | "
            f"{record.turn_cleaned_lexical_differences} |"
        )
    lines.extend(("", audit.recommended_baseline_wording, ""))
    for record in audit.heldout:
        lines.append(
            f"- {record.alignment.split.value} existing-ASR WER: "
            f"{record.asr_word_error_rate_vs_turn_text:.4%} vs dialogue text, "
            f"{record.asr_word_error_rate_vs_cleaned_text:.4%} vs cleaned synthesis text."
        )
    lines.extend(
        (
            "",
            "No new ASR or model evaluation was run; these use previously cached transcripts.",
            "",
            f"All source rows contain {audit.source_original_substituted_raw_changes} raw "
            "original/substituted changes, "
            f"{audit.source_original_substituted_normalized_word_changes} "
            "with normalized word changes (examples include LLM1/LLM2 replaced by actor labels).",
            "",
            "## Held-out material alignment mismatches",
            "",
        )
    )
    for record in heldout:
        if record.turn_vs_audio_original != AlignmentKind.LEXICAL:
            continue
        example = record.example
        lines.extend(
            (
                f"### {example.split.value}: {example.example_id}",
                "",
                f"Zero-based split index {record.split_index}; first48 generations: "
                f"{record.split_index < 48}; first32 paired controls: {record.split_index < 32}; "
                f"first16 qualitative cases: {record.split_index < 16}.",
                "",
                f"Dialogue user: {example.user_text}",
                "",
                f"Assistant target: {example.target_text}",
                "",
                f"Audio original: {record.original_text}",
                "",
                f"Audio cleaned: {record.cleaned_text}",
                "",
                f"Existing ASR: {transcripts[example.example_id].text}",
                "",
            )
        )
    return "\n".join(lines)


def run_audit(configuration: AuditConfiguration) -> SynthesisAlignmentAudit:
    root = configuration.data_root
    turns, _ = load_source(root / "metadata.parquet")
    by_id = {source_example_id(turn): turn for turn in turns if turn.speaker == SourceSpeaker.FIRST}
    training_examples = load_examples(
        root / "examples.jsonl", Split.TRAIN, configuration.training_examples
    )
    if len(training_examples) != configuration.training_examples:
        raise ValueError("Requested training prefix is larger than the selected manifest")
    training = tuple(
        compare(example, by_id[example.example_id], index)
        for index, example in enumerate(training_examples)
    )
    heldout_groups = tuple(
        tuple(
            compare(example, by_id[example.example_id], index)
            for index, example in enumerate(load_examples(root / "examples.jsonl", split))
        )
        for split in (Split.VALIDATION, Split.TEST)
    )
    transcripts = {record.example_id: record for record in load_asr(root / "asr_transcripts.jsonl")}
    substitutions = tuple(
        turn
        for turn in turns
        if turn.audio_original_text is not None
        and turn.audio_substituted_text is not None
        and normalize(turn.audio_original_text) != normalize(turn.audio_substituted_text)
    )
    audit = SynthesisAlignmentAudit(
        manifest_sha256=hashlib.sha256((root / "examples.jsonl").read_bytes()).hexdigest(),
        metadata_sha256=hashlib.sha256((root / "metadata.parquet").read_bytes()).hexdigest(),
        dataset_card_url="https://huggingface.co/datasets/SALT-Research/DeepDialogue-xtts",
        documented_synthesis_field=(
            "The dataset card identifies cleaned_text as text cleaned for TTS and uses it when "
            "displaying segment text. audio_cleaned_text is the documented synthesis reference, "
            "not an independently verified verbatim waveform transcript."
        ),
        normalization=(
            "Lowercase, replace non-word punctuation/symbols with spaces, collapse whitespace. "
            "Normalized word changes flag lexical differences; they do not alone prove a "
            "semantic change. Audio-original/turn mismatches are distinguished from TTS cleaning."
        ),
        training=summarize(Split.TRAIN, training),
        heldout=tuple(
            heldout_summary(split, records, transcripts)
            for split, records in zip((Split.VALIDATION, Split.TEST), heldout_groups, strict=True)
        ),
        source_original_substituted_raw_changes=sum(
            turn.audio_original_text is not None
            and turn.audio_substituted_text is not None
            and turn.audio_original_text != turn.audio_substituted_text
            for turn in turns
        ),
        source_original_substituted_normalized_word_changes=len(substitutions),
        source_substitution_examples=substitutions[:8],
        recommended_baseline_wording=(
            "The text reference baseline uses original dialogue turn text (Example.user_text), "
            "not guaranteed waveform transcript text. Material source audio/turn alignment errors "
            "occur, including audio carrying the assistant target. Preserve main metrics and "
            "report a sensitivity analysis excluding mismatches."
        ),
    )
    output = configuration.output_root
    output.mkdir(parents=True, exist_ok=True)
    heldout = tuple(record for group in heldout_groups for record in group)
    write_alignments(output / "synthesis_alignment_heldout.jsonl", heldout)
    write_alignments(
        output / "synthesis_alignment_training_mismatches.jsonl",
        tuple(
            record for record in training if record.turn_vs_audio_original == AlignmentKind.LEXICAL
        ),
    )
    (output / "synthesis_alignment_audit.json").write_text(
        audit.model_dump_json(indent=2), encoding="utf-8"
    )
    report = readable_report(audit, heldout, transcripts)
    (output / "synthesis_alignment_audit.md").write_text(report, encoding="utf-8")
    print(report)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-examples", type=int, default=20000)
    arguments = parser.parse_args()
    if arguments.train_examples <= 0:
        raise ValueError("Training prefix size must be positive")
    run_audit(AuditConfiguration(arguments.data, arguments.output, arguments.train_examples))


if __name__ == "__main__":
    main()
