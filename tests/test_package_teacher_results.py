import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.package_results import DatasetSource, PackageConfiguration
from scripts.package_teacher_results import (
    TeacherPackageConfiguration,
    audited_examples,
    copy_snapshot,
    examples_from_snapshot,
    feature_inventory,
    package_inputs,
    package_teacher_results,
    pinned_dataset_source,
)
from speech_projector.models import Example, Split


def example(root: Path, identifier: str, split: Split) -> Example:
    audio = root / f"{identifier}.wav"
    feature = root / f"{identifier}.pt"
    audio.write_bytes(f"waveform {identifier}".encode())
    feature.write_bytes(f"feature {identifier}".encode())
    return Example(
        example_id=identifier,
        dialogue_id=f"dialogue-{identifier}",
        split=split,
        history=(),
        user_text="Cleaned speech transcript",
        target_text="Teacher response",
        audio_path=audio,
        duration=1.0,
        domain="test",
        emotion="neutral",
        feature_path=feature,
    )


def test_snapshot_is_idempotent_and_manifest_hash_matches_copied_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b"immutable source")
    results = tmp_path / "results"
    first = copy_snapshot(source, Path("dataset/source.json"), results)
    destination = results / first.path
    initial_modified = destination.stat().st_mtime_ns
    second = copy_snapshot(source, first.path, results)
    assert first == second
    assert destination.stat().st_mtime_ns == initial_modified
    assert first.sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert first.bytes == destination.stat().st_size


def test_source_mutation_during_copy_fails_without_publishing_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b"before")
    results = tmp_path / "results"

    def changing_copy(original: Path, destination: Path) -> Path:
        destination.write_bytes(original.read_bytes())
        original.write_bytes(b"changed while copied")
        return destination

    with patch("scripts.package_teacher_results.shutil.copyfile", side_effect=changing_copy):
        with pytest.raises(ValueError, match="Source changed while packaging"):
            copy_snapshot(source, Path("dataset/source.json"), results)
    assert not (results / "dataset/source.json").exists()
    assert not (results / "dataset/source.json.part").exists()


def test_pinned_metadata_verification_uses_existing_lfs_evidence(tmp_path: Path) -> None:
    directory = tmp_path / "dataset"
    directory.mkdir()
    content = b"original parquet bytes"
    metadata = directory / "metadata.parquet"
    metadata.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    provenance = DatasetSource(
        dataset_repository="SALT-Research/DeepDialogue-xtts",
        dataset_revision="immutable-commit",
        metadata_repository_path="data/train-00000-of-00001.parquet",
        original_download_url="https://example.invalid/main",
        metadata_url="https://example.invalid/immutable-commit",
        metadata_bytes=len(content),
        metadata_sha256=digest,
        metadata_lfs_sha256=digest,
    )
    (directory / "source.json").write_text(provenance.model_dump_json(), encoding="utf-8")
    assert pinned_dataset_source(tmp_path) == provenance
    metadata.write_bytes(b"tampered metadata")
    with pytest.raises(ValueError, match="Sealed metadata differs"):
        pinned_dataset_source(tmp_path)


def test_fixed_inputs_include_both_splits_in_original_manifest_order(tmp_path: Path) -> None:
    examples = tuple(
        example(tmp_path, f"{split.value}-{index}", split) for split in Split for index in range(2)
    )
    selected = audited_examples(examples, 1)
    assert tuple(item.example_id for item in selected) == ("validation-0", "test-0")
    with pytest.raises(ValueError, match="Insufficient"):
        audited_examples(examples, 3)


def test_audio_and_feature_bundle_is_independently_hash_verifiable(tmp_path: Path) -> None:
    selected = (example(tmp_path, "selected", Split.TEST),)
    results = tmp_path / "results"
    records = package_inputs(selected, results)
    for record in records:
        for artifact in (record.audio, record.feature):
            contents = (results / artifact.path).read_bytes()
            assert len(contents) == artifact.bytes
            assert hashlib.sha256(contents).hexdigest() == artifact.sha256
    assert "Cleaned speech transcript" in (results / "audio/index.md").read_text()
    assert "Teacher response" in (results / "audio/index.md").read_text()
    first = (results / "audio/manifest.jsonl").read_bytes()
    assert package_inputs(selected, results) == records
    assert (results / "audio/manifest.jsonl").read_bytes() == first


def test_full_feature_inventory_does_not_copy_external_cache(tmp_path: Path) -> None:
    selected = tuple(example(tmp_path, str(index), Split.TRAIN) for index in range(2))
    manifest = tmp_path / "examples.jsonl"
    manifest.write_text(
        "".join(item.model_dump_json() + "\n" for item in selected), encoding="utf-8"
    )
    inventory = feature_inventory(selected, manifest)
    assert inventory.examples == 2
    assert inventory.feature_bytes == sum(item.feature_path.stat().st_size for item in selected)
    assert tuple(item.path for item in inventory.features) == tuple(
        item.feature_path for item in selected
    )
    assert inventory.source_manifest.sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_incomplete_source_line_rejects_snapshot_without_journal_recovery(tmp_path: Path) -> None:
    selected = example(tmp_path, "selected", Split.TRAIN)
    manifest = tmp_path / "examples.jsonl"
    contents = selected.model_dump_json().encode()
    manifest.write_bytes(contents)
    with pytest.raises(ValueError, match="incomplete final line"):
        examples_from_snapshot(manifest)
    assert manifest.read_bytes() == contents


def test_package_refuses_to_run_while_writers_are_active(tmp_path: Path) -> None:
    configuration = TeacherPackageConfiguration(
        assets=PackageConfiguration(tmp_path / "data", tmp_path / "results", tmp_path / "hub"),
        original_results=tmp_path / "original_results",
        writers_stopped=False,
    )
    with pytest.raises(ValueError, match="writers-stopped"):
        package_teacher_results(configuration)
    assert not configuration.assets.results_root.exists()
