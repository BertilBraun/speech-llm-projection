"""Seal self-contained teacher-experiment assets after every writer has stopped."""

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from scripts.inventory_results import stable_digest
from scripts.package_results import (
    DatasetSource,
    FileArtifact,
    PackageConfiguration,
    package_environment,
    write_bytes_if_changed,
    write_record,
)
from scripts.prepare_teacher_data import TeacherDataPreparation
from speech_projector.models import Example, Record, Split
from speech_projector.teacher import TeacherExport, TeacherProvenance, TeacherTarget


@dataclass(frozen=True)
class TeacherPackageConfiguration:
    assets: PackageConfiguration
    original_results: Path
    writers_stopped: bool


class FeatureInventory(Record):
    source_manifest: FileArtifact
    examples: int
    features: tuple[FileArtifact, ...]
    feature_bytes: int
    storage_description: str


class AuditedTeacherInput(Record):
    example: Example
    audio: FileArtifact
    feature: FileArtifact


class TeacherPackageManifest(Record):
    dataset_source: DatasetSource
    copied_artifacts: tuple[FileArtifact, ...]
    audited_inputs: tuple[AuditedTeacherInput, ...]
    copied_bytes: int
    external_feature_files: int
    external_feature_bytes: int


def describe_file(source: Path, path: Path) -> FileArtifact:
    size, digest = stable_digest(source)
    return FileArtifact(path=path, source_path=source, bytes=size, sha256=digest)


def copy_snapshot(source: Path, destination: Path, results: Path) -> FileArtifact:
    before = describe_file(source, destination)
    target = results / destination
    if not target.exists() or stable_digest(target) != (before.bytes, before.sha256):
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        shutil.copyfile(source, partial)
        copied = stable_digest(partial)
        after = stable_digest(source)
        if copied != (before.bytes, before.sha256) or after != copied:
            partial.unlink()
            raise ValueError(f"Source changed while packaging; stop all writers first: {source}")
        partial.replace(target)
    return before


def pinned_dataset_source(original_results: Path) -> DatasetSource:
    provenance = DatasetSource.model_validate_json(
        (original_results / "dataset/source.json").read_bytes()
    )
    size, digest = stable_digest(original_results / "dataset/metadata.parquet")
    if size != provenance.metadata_bytes or digest != provenance.metadata_sha256:
        raise ValueError("Sealed metadata differs from its pinned dataset provenance")
    if digest != provenance.metadata_lfs_sha256:
        raise ValueError("Pinned dataset provenance does not match the immutable LFS digest")
    return provenance


def examples_from_snapshot(path: Path) -> tuple[Example, ...]:
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        raise ValueError(f"Manifest has an incomplete final line; stop writers first: {path}")
    examples = tuple(Example.model_validate_json(line) for line in content.splitlines())
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError(f"Manifest contains duplicate example identifiers: {path}")
    return examples


def verify_prepared_source(root: Path) -> tuple[Example, ...]:
    preparation = TeacherDataPreparation.model_validate_json(
        (root / "preparation.json").read_bytes()
    )
    if stable_digest(root / "examples_source.jsonl")[1] != preparation.source_manifest_sha256:
        raise ValueError("Teacher source differs from the clean preparation provenance")
    if stable_digest(root / "source_provenance.jsonl")[1] != preparation.provenance_sha256:
        raise ValueError("Teacher source alignment differs from the preparation provenance")
    source = examples_from_snapshot(root / "examples_source.jsonl")
    for subset in preparation.splits:
        identifiers = tuple(
            example.example_id for example in source if example.split == subset.split
        )
        if identifiers != subset.selected_ids or len(identifiers) != subset.selected_examples:
            raise ValueError("Teacher source order/count differs from the preparation provenance")
    for name in ("teacher_bootstrap", "teacher_examples"):
        exported = root / f"{name}.jsonl"
        provenance = TeacherExport.model_validate_json(
            (root / f"{name}.provenance.json").read_bytes()
        )
        if stable_digest(exported) != (provenance.manifest.bytes, provenance.manifest.sha256):
            raise ValueError(f"Completed teacher export differs from its provenance: {name}")
        if len(examples_from_snapshot(exported)) != provenance.examples:
            raise ValueError(f"Completed teacher export has an incorrect example count: {name}")
    return source


def audited_examples(examples: tuple[Example, ...], count_per_split: int) -> tuple[Example, ...]:
    if count_per_split <= 0:
        raise ValueError("Fixed audio selection requires a positive count per split")
    selected: list[Example] = []
    for split in (Split.VALIDATION, Split.TEST):
        subset = tuple(example for example in examples if example.split == split)[:count_per_split]
        if len(subset) != count_per_split:
            raise ValueError(f"Insufficient {split.value} examples for the fixed audio package")
        selected.extend(subset)
    return tuple(selected)


def verify_teacher_journal(source: tuple[Example, ...], manifest: Path, results: Path) -> None:
    directory = results / "teacher_targets"
    provenance = TeacherProvenance.model_validate_json((directory / "provenance.json").read_bytes())
    if stable_digest(manifest) != (
        provenance.input_manifest.bytes,
        provenance.input_manifest.sha256,
    ):
        raise ValueError("Teacher target journal refers to a different source manifest")
    content = (directory / "targets.jsonl").read_bytes()
    if content and not content.endswith(b"\n"):
        raise ValueError("Teacher target journal has an incomplete final line; stop writers first")
    targets = tuple(TeacherTarget.model_validate_json(line) for line in content.splitlines())
    expected = {example.example_id: example for example in source}
    completed = {target.example.example_id for target in targets}
    if len(completed) != len(targets) or completed != expected.keys():
        raise ValueError("Teacher target journal is incomplete or contains duplicate identifiers")
    for target in targets:
        if target.example != expected[target.example.example_id]:
            raise ValueError("Teacher target journal changed an immutable source example")


def feature_inventory(examples: tuple[Example, ...], manifest: Path) -> FeatureInventory:
    paths = tuple(sorted({example.feature_path for example in examples}))
    features = tuple(describe_file(path, path) for path in paths)
    return FeatureInventory(
        source_manifest=describe_file(manifest, manifest),
        examples=len(examples),
        features=features,
        feature_bytes=sum(feature.bytes for feature in features),
        storage_description=(
            "Full selected-cache hashes reference external original cache paths. Only the "
            "16 audited tensors are copied into this result bundle; frozen models and the "
            "complete feature/audio cache are not duplicated."
        ),
    )


def package_inputs(selected: tuple[Example, ...], results: Path) -> tuple[AuditedTeacherInput, ...]:
    records = tuple(
        AuditedTeacherInput(
            example=example,
            audio=copy_snapshot(
                example.audio_path, Path("audio") / f"{example.example_id}.wav", results
            ),
            feature=copy_snapshot(
                example.feature_path, Path("features") / f"{example.example_id}.pt", results
            ),
        )
        for example in selected
    )
    write_bytes_if_changed(
        results / "audio/manifest.jsonl",
        "".join(record.model_dump_json() + "\n" for record in records).encode(),
    )
    lines = [
        "# Fixed teacher-experiment audio inputs",
        "",
        "First eight fixed validation and first eight fixed test examples, in manifest order.",
        "Original dataset responses remain in dataset/source_provenance.jsonl.",
        "",
    ]
    for record in records:
        example = record.example
        lines.extend(
            (
                f"## {example.example_id} · {example.split.value} · {example.domain}",
                "",
                f"[Audio]({example.example_id}.wav) · {example.duration:.3f} seconds",
                "",
                f"Cleaned synthesis transcript: {example.user_text}",
                "",
                f"Qwen teacher response: {example.target_text}",
                "",
            )
        )
    write_bytes_if_changed(results / "audio/index.md", "\n".join(lines).encode())
    return records


def package_teacher_results(configuration: TeacherPackageConfiguration) -> TeacherPackageManifest:
    if not configuration.writers_stopped:
        raise ValueError(
            "Teacher packaging requires --writers-stopped after all jobs/reports finish"
        )
    assets = configuration.assets
    if assets.results_root.resolve().is_relative_to(configuration.original_results.resolve()):
        raise ValueError("Teacher package must not overwrite the sealed original results")
    provenance = pinned_dataset_source(configuration.original_results)
    source = verify_prepared_source(assets.dataset_root)
    verify_teacher_journal(
        source, assets.dataset_root / "examples_source.jsonl", assets.results_root
    )
    preparation = TeacherDataPreparation.model_validate_json(
        (assets.dataset_root / "preparation.json").read_bytes()
    )
    if preparation.metadata_sha256 != provenance.metadata_sha256:
        raise ValueError("Clean preparation metadata differs from the pinned original dataset")
    files = [
        copy_snapshot(
            configuration.original_results / "dataset" / name,
            Path("dataset") / name,
            assets.results_root,
        )
        for name in ("metadata.parquet", "source.json")
    ]
    for path in sorted(assets.dataset_root.rglob("*")):
        relative = path.relative_to(assets.dataset_root)
        if not path.is_file() or relative.parts[0] in ("audio", "features"):
            continue
        if path.suffix == ".part":
            raise ValueError(f"Incomplete data artifact remains before final packaging: {path}")
        files.append(copy_snapshot(path, Path("dataset") / relative, assets.results_root))
    inventory = feature_inventory(source, assets.dataset_root / "examples_source.jsonl")
    write_record(assets.results_root / "dataset/feature_inventory.json", inventory)
    exported = examples_from_snapshot(assets.dataset_root / "teacher_examples.jsonl")
    inputs = package_inputs(audited_examples(exported, 8), assets.results_root)
    files.extend(record.audio for record in inputs)
    files.extend(record.feature for record in inputs)
    package_environment(assets)
    manifest = TeacherPackageManifest(
        dataset_source=provenance,
        copied_artifacts=tuple(files),
        audited_inputs=inputs,
        copied_bytes=sum(file.bytes for file in files),
        external_feature_files=len(inventory.features),
        external_feature_bytes=inventory.feature_bytes,
    )
    write_record(assets.results_root / "package_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--original-results-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--hub", type=Path, required=True)
    parser.add_argument("--writers-stopped", action="store_true")
    arguments = parser.parse_args()
    manifest = package_teacher_results(
        TeacherPackageConfiguration(
            assets=PackageConfiguration(arguments.data_root, arguments.results_root, arguments.hub),
            original_results=arguments.original_results_root,
            writers_stopped=arguments.writers_stopped,
        )
    )
    print(manifest.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
