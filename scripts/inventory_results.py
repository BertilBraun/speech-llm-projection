"""Hash a quiescent result tree, then verify the same bytes after relocation."""

import argparse
import hashlib
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from speech_projector.models import FileArtifact, Record

INVENTORY_PATH = Path("reproducibility/results_inventory.json")
VERIFICATION_PATH = Path("reproducibility/results_inventory_verification.json")
EXCLUDED_PATHS = frozenset(
    (
        INVENTORY_PATH,
        VERIFICATION_PATH,
        INVENTORY_PATH.with_suffix(".part"),
        VERIFICATION_PATH.with_suffix(".part"),
    )
)


class Mode(str, Enum):
    WRITE = "write-inventory"
    VERIFY = "verify"


class ResultInventory(Record):
    captured_at: datetime
    source_root: Path
    artifacts: tuple[FileArtifact, ...]


class InventoryVerification(Record):
    verified_at: datetime
    verified_root: Path
    files: int
    bytes_verified: int
    inventory_sha256: str


def result_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path.relative_to(root)
            for path in root.rglob("*")
            if path.is_file() and path.relative_to(root) not in EXCLUDED_PATHS
        )
    )


def stable_digest(path: Path) -> tuple[int, str]:
    initial = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while content := stream.read(1024 * 1024):
            digest.update(content)
    final = path.stat()
    if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
        raise ValueError(f"File changed during hashing; stop all writers first: {path}")
    return final.st_size, digest.hexdigest()


def write_record(destination: Path, record: Record) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    partial.write_text(record.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    partial.replace(destination)


def write_inventory(root: Path, writers_stopped: bool) -> ResultInventory:
    if not writers_stopped:
        raise ValueError(
            "Final inventory requires --writers-stopped after all jobs and reports finish"
        )
    paths = result_files(root)
    artifacts: list[FileArtifact] = []
    for relative_path in paths:
        source = root / relative_path
        size, digest = stable_digest(source)
        artifacts.append(
            FileArtifact(path=relative_path, source_path=source, bytes=size, sha256=digest)
        )
    if result_files(root) != paths:
        raise ValueError("Result file set changed during hashing; stop all writers first")
    inventory = ResultInventory(
        captured_at=datetime.now(timezone.utc), source_root=root, artifacts=tuple(artifacts)
    )
    write_record(root / INVENTORY_PATH, inventory)
    return inventory


def verify_inventory(root: Path) -> InventoryVerification:
    serialized = (root / INVENTORY_PATH).read_bytes()
    inventory = ResultInventory.model_validate_json(serialized)
    expected_paths = tuple(sorted(artifact.path for artifact in inventory.artifacts))
    if result_files(root) != expected_paths:
        raise ValueError("Result tree contains missing or additional files versus the inventory")
    for artifact in inventory.artifacts:
        size, digest = stable_digest(root / artifact.path)
        if size != artifact.bytes or digest != artifact.sha256:
            raise ValueError(f"Size or SHA256 differs from inventory: {artifact.path}")
    verification = InventoryVerification(
        verified_at=datetime.now(timezone.utc),
        verified_root=root,
        files=len(inventory.artifacts),
        bytes_verified=sum(artifact.bytes for artifact in inventory.artifacts),
        inventory_sha256=hashlib.sha256(serialized).hexdigest(),
    )
    write_record(root / VERIFICATION_PATH, verification)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", type=Mode, choices=tuple(mode.value for mode in Mode))
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--writers-stopped", action="store_true")
    arguments = parser.parse_args()
    root = arguments.results.resolve(strict=True)
    match Mode(arguments.mode):
        case Mode.WRITE:
            inventory = write_inventory(root, arguments.writers_stopped)
            print(f"Inventoried {len(inventory.artifacts)} files at {root / INVENTORY_PATH}")
        case Mode.VERIFY:
            verification = verify_inventory(root)
            print(verification.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
