"""Prepare clean, transcript-aligned inputs for the separate Qwen-teacher experiment."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from pydantic import Field

from scripts.audit_synthesis_alignment import (
    AlignmentKind,
    SynthesisAlignment,
    compare,
    source_example_id,
)
from scripts.leakage_audit import LeakageAudit, SplitOverlap, file_digest, normalized_pair
from speech_projector.cache import CacheConfig, load_asr
from speech_projector.data import (
    DataConfig,
    DatasetReport,
    DownloadResult,
    SourceTurn,
    build_examples,
    distribution,
    download_audio,
    load_examples,
    load_source,
    save_examples,
)
from speech_projector.models import Example, Record, Split, Turn


class TeacherDataConfig(Record):
    original_root: Path
    output_root: Path
    training_examples: int = Field(default=20000, gt=0)
    heldout_examples: int = Field(default=512, gt=0)
    seed: int = 42
    history_turns: int = Field(default=2, ge=0)
    workers: int = Field(default=4, gt=0)


class TeacherSplitPreparation(Record):
    split: Split
    selected_examples: int
    distinct_dialogues: int
    candidate_examples: int
    material_alignment_exclusions: int
    lexical_substitution_exclusions: int
    missing_synthesis_text_exclusions: int
    collision_dialogue_exclusions: tuple[str, ...]
    missing_audio: int
    missing_features: int
    selected_ids: tuple[str, ...]


class TeacherDataPreparation(Record):
    configuration: TeacherDataConfig
    original_manifest_sha256: str
    metadata_sha256: str
    source_manifest_sha256: str
    provenance_sha256: str
    splits: tuple[TeacherSplitPreparation, ...]
    subset_definition: str
    transcript_reference: str
    target_status: str


class TeacherDownloadReport(Record):
    examples: int
    downloaded_bytes: int
    elapsed_seconds: float
    failed_example_ids: tuple[str, ...]


class TeacherFeaturePlan(Record):
    configuration: CacheConfig
    source_manifest_sha256: str
    selected_examples: int
    existing_features: int
    pending_example_ids: tuple[str, ...]
    pending_audio_seconds: float
    estimated_new_bf16_feature_bytes: int


class SubsetAvailability(Record):
    split: Split
    examples: int
    missing_audio: int
    missing_features: int


class TranscriptOverlap(Record):
    left: Split
    right: Split
    shared_transcripts: int
    left_examples_with_overlap: int
    right_examples_with_overlap: int


class TeacherInputReadiness(Record):
    source_manifest_sha256: str
    bootstrap: tuple[SubsetAvailability, ...]
    cleaned_user_transcript_overlap: tuple[TranscriptOverlap, ...]


@dataclass(frozen=True)
class TeacherPrompt:
    user_text: str
    history: tuple[Turn, ...]


def teacher_prompt(example: Example) -> TeacherPrompt:
    return TeacherPrompt(example.user_text, example.history[-2:])


@dataclass(frozen=True)
class CleanCandidates:
    alignments: tuple[SynthesisAlignment, ...]
    material_exclusions: int
    substitution_exclusions: int
    missing_text_exclusions: int


def clean_candidates(
    examples: Sequence[Example], sources: Mapping[str, SourceTurn]
) -> CleanCandidates:
    selected: list[SynthesisAlignment] = []
    material = substitutions = missing = 0
    for index, example in enumerate(examples):
        source = sources[example.example_id]
        if (
            source.audio_original_text is None
            or source.audio_substituted_text is None
            or source.audio_cleaned_text is None
            or not source.audio_cleaned_text.strip()
        ):
            missing += 1
            continue
        alignment = compare(example, source, index)
        material += alignment.turn_vs_audio_original == AlignmentKind.LEXICAL
        substitutions += alignment.original_vs_substituted == AlignmentKind.LEXICAL
        if (
            alignment.turn_vs_audio_original == AlignmentKind.LEXICAL
            or alignment.original_vs_substituted == AlignmentKind.LEXICAL
        ):
            continue
        selected.append(alignment)
    return CleanCandidates(tuple(selected), material, substitutions, missing)


def prepared_example(alignment: SynthesisAlignment, output_root: Path) -> Example:
    original = alignment.example
    source = alignment.source
    assert source.segment_audio_path is not None
    audio = original.audio_path
    if not audio.exists():
        audio = output_root / "audio" / source.segment_audio_path
    feature = original.feature_path
    if not feature.exists():
        feature = output_root / "features" / original.example_id[:2] / f"{original.example_id}.pt"
    return original.model_copy(
        update={"user_text": alignment.cleaned_text, "audio_path": audio, "feature_path": feature}
    )


def select_heldout(
    candidates: Sequence[SynthesisAlignment],
    retained: Sequence[Example],
    count: int,
    output_root: Path,
) -> tuple[tuple[SynthesisAlignment, ...], tuple[str, ...]]:
    retained_dialogues = {example.dialogue_id for example in retained}
    retained_paths = {example.audio_path for example in retained}
    retained_pairs = {normalized_pair(example) for example in retained}
    retained_prompts = {teacher_prompt(example) for example in retained}
    excluded = {
        alignment.example.dialogue_id
        for alignment in candidates
        if (
            alignment.example.dialogue_id in retained_dialogues
            or prepared_example(alignment, output_root).audio_path in retained_paths
            or normalized_pair(prepared_example(alignment, output_root)) in retained_pairs
            or teacher_prompt(prepared_example(alignment, output_root)) in retained_prompts
        )
    }
    selected = tuple(
        alignment for alignment in candidates if alignment.example.dialogue_id not in excluded
    )[:count]
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} clean nonleaking heldout candidates; need {count}")
    return selected, tuple(sorted(excluded))


def write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to change prepared teacher inputs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_bytes(content)
    partial.replace(path)


def write_feature_plan(
    examples: Sequence[Example], configuration: TeacherDataConfig, manifest: Path
) -> TeacherFeaturePlan:
    pending = tuple(example for example in examples if not example.feature_path.exists())
    plan = TeacherFeaturePlan(
        configuration=CacheConfig(
            root=configuration.output_root,
            train_examples=configuration.training_examples,
            transcribe_heldout=True,
        ),
        source_manifest_sha256=file_digest(manifest),
        selected_examples=len(examples),
        existing_features=len(examples) - len(pending),
        pending_example_ids=tuple(example.example_id for example in pending),
        pending_audio_seconds=sum(example.duration for example in pending),
        estimated_new_bf16_feature_bytes=int(sum(example.duration for example in pending) * 76800),
    )
    (configuration.output_root / "feature_plan.json").write_text(
        plan.model_dump_json(indent=2), encoding="utf-8"
    )
    save_examples(configuration.output_root / "features_pending.jsonl", list(pending))
    return plan


def input_readiness(examples: Sequence[Example], manifest: Path) -> TeacherInputReadiness:
    groups = {split: [example for example in examples if example.split == split] for split in Split}
    bootstrap: list[SubsetAvailability] = []
    for split, selected in groups.items():
        subset = selected[:256] if split == Split.TRAIN else selected[:32]
        bootstrap.append(
            SubsetAvailability(
                split=split,
                examples=len(subset),
                missing_audio=sum(not example.audio_path.exists() for example in subset),
                missing_features=sum(not example.feature_path.exists() for example in subset),
            )
        )
    overlaps: list[TranscriptOverlap] = []
    for left, right in combinations(Split, 2):
        left_texts = [normalized_pair(example)[0] for example in groups[left]]
        right_texts = [normalized_pair(example)[0] for example in groups[right]]
        shared = set(left_texts) & set(right_texts)
        overlaps.append(
            TranscriptOverlap(
                left=left,
                right=right,
                shared_transcripts=len(shared),
                left_examples_with_overlap=sum(text in shared for text in left_texts),
                right_examples_with_overlap=sum(text in shared for text in right_texts),
            )
        )
    return TeacherInputReadiness(
        source_manifest_sha256=file_digest(manifest),
        bootstrap=tuple(bootstrap),
        cleaned_user_transcript_overlap=tuple(overlaps),
    )


def preserve_asr(examples: Sequence[Example], configuration: TeacherDataConfig) -> None:
    path = configuration.output_root / "asr_transcripts.jsonl"
    existing = load_asr(path)
    existing_ids = {record.example_id for record in existing}
    heldout_ids = {example.example_id for example in examples if example.split != Split.TRAIN}
    retained = [
        record
        for record in load_asr(configuration.original_root / "asr_transcripts.jsonl")
        if record.example_id in heldout_ids and record.example_id not in existing_ids
    ]
    if retained:
        with path.open("a", encoding="utf-8") as stream:
            stream.writelines(record.model_dump_json() + "\n" for record in retained)


def write_dataset_report(
    source_report: DatasetReport,
    examples: Sequence[Example],
    columns: tuple[str, ...],
    output_root: Path,
) -> None:
    report = source_report.model_copy(
        update={
            "train_examples": sum(example.split == Split.TRAIN for example in examples),
            "validation_examples": sum(example.split == Split.VALIDATION for example in examples),
            "test_examples": sum(example.split == Split.TEST for example in examples),
            "duration": distribution([example.duration for example in examples]),
            "user_words": distribution(
                [float(len(example.user_text.split())) for example in examples]
            ),
            "target_words": distribution(
                [float(len(example.target_text.split())) for example in examples]
            ),
            "domains_selected": tuple(
                Counter(example.domain for example in examples).most_common()
            ),
            "source_columns": columns,
            "split_method": (
                source_report.split_method
                + "; selected teacher inputs remove lexical synthesis mismatch/substitution "
                "and complete heldout dialogues with cleaned-user/source-target or exact "
                "cleaned-user plus role/text history[-2:] prompt collisions. "
                "Filters/usable_pairs describe source eligibility; "
                "additional cleaning is in preparation.json."
            ),
        }
    )
    (output_root / "dataset_report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    lines = [
        "# Teacher source-corpus summary",
        "",
        f"Selected: {report.train_examples} train, {report.validation_examples} validation, "
        f"{report.test_examples} test. Seed 42; split by complete model_dir/conversation_id.",
        "",
        "User words describe audio_cleaned_text, the documented synthesis input. Target words "
        "describe original dialogue responses preserved as provenance, "
        "not generated teacher targets.",
        "",
        "| Selected distribution | Mean | Median | P10 | P90 | P99 | Min | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, measured in (
        ("Audio seconds", report.duration),
        ("Cleaned user words", report.user_words),
        ("Original target words", report.target_words),
    ):
        lines.append(
            f"| {name} | {measured.mean:.3f} | {measured.median:.3f} | {measured.p10:.3f} "
            f"| {measured.p90:.3f} | {measured.p99:.3f} | {measured.minimum:.3f} "
            f"| {measured.maximum:.3f} |"
        )
    lines.extend(
        [
            "",
            "The original metadata source-stage filter counts and dialogue distributions remain "
            "in dataset_report.json. Additional alignment/substitution exclusions and selected "
            "dialogue counts are in preparation.json; byte-hash leakage coverage is in "
            "dataset_leakage.json. Source targets must be replaced before projector training.",
        ]
    )
    (output_root / "dataset_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(configuration: TeacherDataConfig) -> tuple[Example, ...]:
    if configuration.original_root.resolve() == configuration.output_root.resolve():
        raise ValueError(
            "Teacher output must use a separate root from the original experiment data"
        )
    turns, columns = load_source(configuration.original_root / "metadata.parquet")
    sources = {source_example_id(turn): turn for turn in turns}
    original = load_examples(configuration.original_root / "examples.jsonl", Split.TRAIN)
    candidates, source_report = build_examples(
        turns,
        DataConfig(
            root=configuration.original_root,
            seed=configuration.seed,
            history_turns=configuration.history_turns,
            max_train=len(original),
            validation_examples=len(turns),
            test_examples=len(turns),
        ),
    )
    regenerated_train = [example for example in candidates if example.split == Split.TRAIN]
    if regenerated_train != original:
        raise ValueError("Original training manifest differs from regenerated seeded candidates")
    retained: list[Example] = []
    provenance: list[SynthesisAlignment] = []
    summaries: list[TeacherSplitPreparation] = []
    for split in Split:
        split_candidates = (
            original
            if split == Split.TRAIN
            else [example for example in candidates if example.split == split]
        )
        clean = clean_candidates(split_candidates, sources)
        count = (
            configuration.training_examples
            if split == Split.TRAIN
            else configuration.heldout_examples
        )
        if split == Split.TRAIN:
            selected = clean.alignments[:count]
            excluded: tuple[str, ...] = ()
            if len(selected) != count:
                raise ValueError(f"Insufficient clean training candidates: {len(selected)}")
        else:
            selected, excluded = select_heldout(
                clean.alignments, retained, count, configuration.output_root
            )
        examples = tuple(
            prepared_example(alignment, configuration.output_root) for alignment in selected
        )
        retained.extend(examples)
        provenance.extend(selected)
        summaries.append(
            TeacherSplitPreparation(
                split=split,
                selected_examples=len(examples),
                distinct_dialogues=len({example.dialogue_id for example in examples}),
                candidate_examples=len(split_candidates),
                material_alignment_exclusions=clean.material_exclusions,
                lexical_substitution_exclusions=clean.substitution_exclusions,
                missing_synthesis_text_exclusions=clean.missing_text_exclusions,
                collision_dialogue_exclusions=excluded,
                missing_audio=sum(not example.audio_path.exists() for example in examples),
                missing_features=sum(not example.feature_path.exists() for example in examples),
                selected_ids=tuple(example.example_id for example in examples),
            )
        )
    root = configuration.output_root
    source_manifest = root / "examples_source.jsonl"
    provenance_path = root / "source_provenance.jsonl"
    write_immutable(
        source_manifest, "".join(example.model_dump_json() + "\n" for example in retained).encode()
    )
    write_immutable(
        provenance_path, "".join(record.model_dump_json() + "\n" for record in provenance).encode()
    )
    # The teacher owns replacement of this target manifest after initial preparation.
    if not (root / "examples.jsonl").exists():
        save_examples(root / "examples.jsonl", retained)
    summary = TeacherDataPreparation(
        configuration=configuration,
        original_manifest_sha256=file_digest(configuration.original_root / "examples.jsonl"),
        metadata_sha256=file_digest(configuration.original_root / "metadata.parquet"),
        source_manifest_sha256=file_digest(source_manifest),
        provenance_sha256=file_digest(provenance_path),
        splits=tuple(summaries),
        subset_definition=(
            "TRAIN: first requested clean examples in original 30k seeded order; nested prefixes. "
            "Heldout: original seed-42 dialogue/hash splits and example order, lexical alignment "
            "and TTS substitutions removed; whole dialogues with cross-split path/pair or exact "
            "teacher prompt (cleaned user plus role/text history[-2:]) collisions "
            "excluded. Original data.build_examples test-pair exclusion also applies to the pool."
        ),
        transcript_reference=(
            "audio_cleaned_text, documented synthesis input; not independently transcribed audio"
        ),
        target_status=(
            "Original dataset targets retained only as placeholders; "
            "Qwen teacher must replace examples.jsonl before training"
        ),
    )
    (root / "preparation.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    write_feature_plan(retained, configuration, source_manifest)
    preserve_asr(retained, configuration)
    write_dataset_report(source_report, retained, columns, root)
    (root / "input_readiness.json").write_text(
        input_readiness(retained, source_manifest).model_dump_json(indent=2), encoding="utf-8"
    )
    print(
        "\n".join(record.model_dump_json(exclude={"selected_ids"}) for record in summaries),
        flush=True,
    )
    return tuple(retained)


def download_selected(examples: Sequence[Example], configuration: TeacherDataConfig) -> None:
    pending = tuple(example for example in examples if not example.audio_path.exists())
    started = time.monotonic()
    results: list[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=configuration.workers) as executor:
        futures = [
            executor.submit(download_audio, example, configuration.output_root)
            for example in pending
        ]
        for index, future in enumerate(futures, 1):
            result = future.result()
            results.append(result)
            if index % 20 == 0 or not result.success:
                print(f"Teacher audio {index}/{len(pending)}: {result}", flush=True)
    report = TeacherDownloadReport(
        examples=len(pending),
        downloaded_bytes=sum(result.bytes_downloaded for result in results),
        elapsed_seconds=time.monotonic() - started,
        failed_example_ids=tuple(result.example_id for result in results if not result.success),
    )
    (configuration.output_root / "download_report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    if report.failed_example_ids:
        raise ValueError(f"Audio downloads failed for {len(report.failed_example_ids)} examples")


def audit(examples: Sequence[Example], output_root: Path) -> LeakageAudit:
    groups = {split: [example for example in examples if example.split == split] for split in Split}
    hashes: dict[Split, set[str]] = {}
    for split, selected in groups.items():
        if any(not example.audio_path.exists() for example in selected):
            raise ValueError(
                f"Cannot finish byte-hash leakage audit with missing {split.value} audio"
            )
        hashes[split] = {file_digest(example.audio_path) for example in selected}
        print(f"Hashed {len(selected)} {split.value} WAVs", flush=True)
    overlaps = tuple(
        SplitOverlap(
            left=left,
            right=right,
            shared_dialogue_ids=len(
                {example.dialogue_id for example in groups[left]}
                & {example.dialogue_id for example in groups[right]}
            ),
            shared_audio_paths=len(
                {example.audio_path for example in groups[left]}
                & {example.audio_path for example in groups[right]}
            ),
            identical_downloaded_waveforms=len(hashes[left] & hashes[right]),
            exact_normalized_user_target_pairs=len(
                {normalized_pair(example) for example in groups[left]}
                & {normalized_pair(example) for example in groups[right]}
            ),
        )
        for left, right in combinations(Split, 2)
    )
    report = LeakageAudit(
        train_manifest_examples=len(groups[Split.TRAIN]),
        validation_examples=len(groups[Split.VALIDATION]),
        test_examples=len(groups[Split.TEST]),
        hashed_downloaded_train_waveforms=len(groups[Split.TRAIN]),
        hashed_validation_waveforms=len(groups[Split.VALIDATION]),
        hashed_test_waveforms=len(groups[Split.TEST]),
        overlaps=overlaps,
        waveform_hash_method="SHA256 of every complete selected WAV file",
        interpretation=(
            "All selected source targets audited before Qwen teacher replacement; "
            "rerun pair audit on teacher manifest"
        ),
    )
    (output_root / "dataset_leakage.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    if any(
        overlap.shared_dialogue_ids
        or overlap.shared_audio_paths
        or overlap.identical_downloaded_waveforms
        or overlap.exact_normalized_user_target_pairs
        for overlap in overlaps
    ):
        raise ValueError(
            "Cross-split leakage detected; inspect dataset_leakage.json before training"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--audit", action="store_true")
    arguments = parser.parse_args()
    configuration = TeacherDataConfig(
        original_root=arguments.original_root, output_root=arguments.output_root
    )
    examples = prepare(configuration)
    if arguments.download:
        download_selected(examples, configuration)
    if arguments.audit:
        audit(examples, configuration.output_root)


if __name__ == "__main__":
    main()
