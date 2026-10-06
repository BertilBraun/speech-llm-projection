"""One-time migration of completed linear records with an unused hidden dimension."""

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from scripts.package_results import FileArtifact
from speech_projector.models import LinearProjectorConfig, Record, RunConfig, RunResult


class LinearProjectorRecordWithUnusedDimension(LinearProjectorConfig):
    hidden_dimension: int


class LinearRunConfigWithUnusedDimension(RunConfig):
    projector: LinearProjectorRecordWithUnusedDimension


class LinearRunResultWithUnusedDimension(RunResult):
    config: LinearRunConfigWithUnusedDimension


class LinearMigrationRecord(Record):
    migrated_at: datetime
    run_name: str
    original_records: tuple[FileArtifact, ...]
    migrated_records: tuple[FileArtifact, ...]


def file_artifact(root: Path, path: Path, content: bytes) -> FileArtifact:
    return FileArtifact(
        path=path.relative_to(root),
        source_path=path,
        bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def migrate_linear_run(directory: Path, writers_stopped: bool) -> LinearMigrationRecord:
    if not writers_stopped:
        raise ValueError("Linear record migration requires all experiment writers to be stopped")
    config_path = directory / "config.json"
    result_path = directory / "result.json"
    config_content = config_path.read_bytes()
    result_content = result_path.read_bytes()
    original_config = LinearRunConfigWithUnusedDimension.model_validate_json(config_content)
    original_result = LinearRunResultWithUnusedDimension.model_validate_json(result_content)
    if original_result.config != original_config:
        raise ValueError("The completed result and saved configuration disagree")
    migrated_config_content = original_config.model_dump_json(
        indent=2, exclude={"projector": {"hidden_dimension"}}
    ).encode("utf-8")
    migrated_result_content = original_result.model_dump_json(
        indent=2, exclude={"config": {"projector": {"hidden_dimension"}}}
    ).encode("utf-8")
    migrated_config = RunConfig.model_validate_json(migrated_config_content)
    migrated_result = RunResult.model_validate_json(migrated_result_content)
    assert migrated_result.config == migrated_config
    assert migrated_result.checkpoint_path == original_result.checkpoint_path
    archive_directory = directory / "original_records"
    if archive_directory.exists():
        raise ValueError(
            "Original records already exist; one-time migration will not overwrite them"
        )
    archive_directory.mkdir()
    original_artifacts: list[FileArtifact] = []
    migrated_artifacts: list[FileArtifact] = []
    records = (
        (config_path, config_content, migrated_config_content),
        (result_path, result_content, migrated_result_content),
    )
    for path, original_content, _ in records:
        archive_path = archive_directory / path.name
        archive_path.write_bytes(original_content)
        assert archive_path.read_bytes() == original_content
        original_artifacts.append(file_artifact(directory, archive_path, original_content))
    if any(path.read_bytes() != original_content for path, original_content, _ in records):
        raise ValueError(
            "An experiment record changed during migration; no conversion was performed"
        )
    for path, _, migrated_content in records:
        pending = path.with_suffix(".migration.pending.json")
        pending.write_bytes(migrated_content)
        pending.replace(path)
        migrated_artifacts.append(file_artifact(directory, path, migrated_content))
    record = LinearMigrationRecord(
        migrated_at=datetime.now(timezone.utc),
        run_name=original_config.name,
        original_records=tuple(original_artifacts),
        migrated_records=tuple(migrated_artifacts),
    )
    (archive_directory / "migration.json").write_text(
        record.model_dump_json(indent=2), encoding="utf-8"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--writers-stopped", action="store_true")
    arguments = parser.parse_args()
    print(
        migrate_linear_run(arguments.run_dir, arguments.writers_stopped).model_dump_json(indent=2)
    )


if __name__ == "__main__":
    main()
