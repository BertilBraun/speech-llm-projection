"""Prepare a separate cached-feature V0 control excluding audited material mismatches."""

import argparse
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from scripts.audit_synthesis_alignment import (
    AlignmentKind,
    SynthesisAlignment,
    SynthesisAlignmentAudit,
    compare,
    load_alignments,
    source_example_id,
)
from scripts.package_results import FileArtifact, file_digest
from speech_projector.configuration import feasibility_run
from speech_projector.data import load_examples
from speech_projector.models import Example, Record, RunConfig, Split


@dataclass(frozen=True)
class PreparationInputs:
    manifest: Path
    metadata: Path
    audit_directory: Path
    original_configuration: Path
    results_root: Path


class CleanV0Preparation(Record):
    configuration: RunConfig
    preparation_git_commit: str
    input_artifacts: tuple[FileArtifact, ...]
    alternate_manifest: FileArtifact
    alternate_configuration: FileArtifact
    training_candidate_examples: int
    excluded_training_ids: tuple[str, ...]
    excluded_validation_ids: tuple[str, ...]
    excluded_test_ids: tuple[str, ...]
    selected_training_original_indices: tuple[int, ...]
    selection_definition: str
    interpretation: str


def artifact(path: Path) -> FileArtifact:
    return FileArtifact(
        path=path, source_path=path, bytes=path.stat().st_size, sha256=file_digest(path)
    )


def verify_alignments(
    examples: Sequence[Example], records: Sequence[SynthesisAlignment], *, complete: bool
) -> None:
    indices = tuple(record.split_index for record in records)
    if len(indices) != len(set(indices)) or indices != tuple(sorted(indices)):
        raise ValueError("Audit records must have unique indices in original manifest order")
    if complete and indices != tuple(range(len(examples))):
        raise ValueError("Held-out audit must cover the complete fixed split")
    for record in records:
        if record.split_index >= len(examples) or examples[record.split_index] != record.example:
            raise ValueError("Audited example differs from its original manifest position")
        if source_example_id(record.source) != record.example.example_id:
            raise ValueError("Audited source does not identify the selected example")
        if compare(record.example, record.source, record.split_index) != record:
            raise ValueError("Audit classification differs from its canonical synthesis text")


def select_clean_prefix(
    examples: Sequence[Example], mismatches: Sequence[SynthesisAlignment], count: int
) -> tuple[tuple[Example, ...], tuple[int, ...]]:
    if count <= 0:
        raise ValueError("Clean training selection count must be positive")
    verify_alignments(examples, mismatches, complete=False)
    if any(record.turn_vs_audio_original != AlignmentKind.LEXICAL for record in mismatches):
        raise ValueError("Training exclusion audit must contain only material mismatches")
    excluded = frozenset(record.example.example_id for record in mismatches)
    indices = tuple(
        index for index, example in enumerate(examples) if example.example_id not in excluded
    )[:count]
    if len(indices) != count:
        raise ValueError("Insufficient clean training examples in the audited candidate prefix")
    return tuple(examples[index] for index in indices), indices


def clean_heldout(
    examples: Sequence[Example], records: Sequence[SynthesisAlignment]
) -> tuple[tuple[Example, ...], tuple[str, ...]]:
    verify_alignments(examples, records, complete=True)
    excluded = tuple(
        record.example.example_id
        for record in records
        if record.turn_vs_audio_original == AlignmentKind.LEXICAL
    )
    return tuple(example for example in examples if example.example_id not in excluded), excluded


def write_immutable(path: Path, contents: bytes) -> None:
    if path.exists():
        if path.read_bytes() != contents:
            raise ValueError(f"Refusing to replace a different clean-control input: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)


def require_cached_features(examples: Sequence[Example]) -> None:
    for example in examples:
        if not example.feature_path.is_file():
            raise ValueError(f"Required cached Whisper feature is missing: {example.feature_path}")


def prepare_clean_v0(inputs: PreparationInputs) -> CleanV0Preparation:
    original = RunConfig.model_validate_json(inputs.original_configuration.read_bytes())
    if original != feasibility_run():
        raise ValueError("Clean control must exactly match the original V0 training configuration")
    audit_path = inputs.audit_directory / "synthesis_alignment_audit.json"
    heldout_path = inputs.audit_directory / "synthesis_alignment_heldout.jsonl"
    mismatches_path = inputs.audit_directory / "synthesis_alignment_training_mismatches.jsonl"
    audit = SynthesisAlignmentAudit.model_validate_json(audit_path.read_bytes())
    if file_digest(inputs.manifest) != audit.manifest_sha256:
        raise ValueError("Alignment audit SHA256 does not match the unchanged original manifest")
    if file_digest(inputs.metadata) != audit.metadata_sha256:
        raise ValueError("Alignment audit SHA256 does not match the original dataset metadata")
    if audit.training.split != Split.TRAIN or audit.training.examples != 20000:
        raise ValueError("Clean control requires the canonical first-20k training alignment audit")
    training = tuple(load_examples(inputs.manifest, Split.TRAIN, audit.training.examples))
    validation = tuple(load_examples(inputs.manifest, Split.VALIDATION, 128))
    test = tuple(load_examples(inputs.manifest, Split.TEST, 128))
    if len(training) != 20000 or len(validation) != 128 or len(test) != 128:
        raise ValueError("Original candidate/held-out split sizes do not match the audited program")
    mismatches = load_alignments(mismatches_path)
    if (
        len(mismatches) != audit.training.turn_original_lexical_differences
        or len(mismatches) != 389
    ):
        raise ValueError("Canonical training audit must contain all 389 material mismatch records")
    heldout = load_alignments(heldout_path)
    if len(heldout) != 256:
        raise ValueError("Canonical held-out audit must contain exactly 256 records")
    selected, indices = select_clean_prefix(training, mismatches, original.train_examples)
    clean_validation, excluded_validation = clean_heldout(
        validation, tuple(record for record in heldout if record.example.split == Split.VALIDATION)
    )
    clean_test, excluded_test = clean_heldout(
        test, tuple(record for record in heldout if record.example.split == Split.TEST)
    )
    if len(clean_validation) != 125 or len(clean_test) != 125:
        raise ValueError(
            "Clean held-out control must retain the canonical 125/125 aligned examples"
        )
    examples = selected + clean_validation + clean_test
    require_cached_features(examples)
    configuration = original.model_copy(
        update={
            "name": "v0_clean_256_mlp_10hz",
            "validation_examples": len(clean_validation),
            "test_examples": len(clean_test),
        }
    )
    directory = inputs.results_root / configuration.name / "inputs"
    manifest_path = directory / "examples.jsonl"
    configuration_path = directory / "config.json"
    write_immutable(
        manifest_path, "".join(example.model_dump_json() + "\n" for example in examples).encode()
    )
    write_immutable(configuration_path, configuration.model_dump_json(indent=2).encode())
    preparation = CleanV0Preparation(
        configuration=configuration,
        preparation_git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        input_artifacts=tuple(
            artifact(path)
            for path in (
                inputs.manifest,
                inputs.metadata,
                inputs.original_configuration,
                audit_path,
                heldout_path,
                mismatches_path,
            )
        ),
        alternate_manifest=artifact(manifest_path),
        alternate_configuration=artifact(configuration_path),
        training_candidate_examples=len(training),
        excluded_training_ids=tuple(record.example.example_id for record in mismatches),
        excluded_validation_ids=excluded_validation,
        excluded_test_ids=excluded_test,
        selected_training_original_indices=indices,
        selection_definition=(
            "First 256 retained examples in original TRAIN manifest order within its first20k "
            "prefix, excluding all389 canonical turn.text/audio_original_text normalized-word "
            "mismatches. Retain original fixed validation/test order after excluding3 mismatches "
            "per split. Preserve canonical examples, cache paths, history and response targets."
        ),
        interpretation=(
            "Late V0 feasibility QA control, separate from the original V0–V3 matrix. Excludes "
            "the audited material alignment defects and their accidental target-audio supervision. "
            "Does not claim that all remaining synthetic waveforms are error-free."
        ),
    )
    write_immutable(directory / "provenance.json", preparation.model_dump_json(indent=2).encode())
    return preparation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--audit-directory", type=Path, required=True)
    parser.add_argument("--original-configuration", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    arguments = parser.parse_args()
    prepared = prepare_clean_v0(
        PreparationInputs(
            manifest=arguments.manifest,
            metadata=arguments.metadata,
            audit_directory=arguments.audit_directory,
            original_configuration=arguments.original_configuration,
            results_root=arguments.results,
        )
    )
    print(prepared.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
