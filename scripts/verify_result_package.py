"""Independently verify copied result snapshots against their typed manifests."""

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from scripts.package_results import AudioArtifact, FileArtifact, PackageManifest
from speech_projector.models import Record


class VerificationReport(Record):
    verified_at: datetime
    copied_files: int
    bytes_verified: int
    audio_examples: int
    feature_files: int
    package_manifest_sha256: str


def verify_file(results: Path, artifact: FileArtifact) -> None:
    destination = results / artifact.path
    if destination.stat().st_size != artifact.bytes:
        raise ValueError(f"Copied size differs from manifest: {artifact.path}")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if digest != artifact.sha256:
        raise ValueError(f"Copied SHA256 differs from manifest: {artifact.path}")


def verify_package(results: Path) -> VerificationReport:
    manifest_path = results / "package_manifest.json"
    serialized = manifest_path.read_bytes()
    package = PackageManifest.model_validate_json(serialized)
    for artifact in package.artifacts:
        verify_file(results, artifact)
    total_bytes = sum(artifact.bytes for artifact in package.artifacts)
    if total_bytes != package.copied_artifact_bytes:
        raise ValueError("Package total byte count differs from artifact records")
    with (results / "features" / "manifest.jsonl").open(encoding="utf-8") as stream:
        features = tuple(FileArtifact.model_validate_json(line) for line in stream)
    for artifact in features:
        if artifact not in package.artifacts:
            raise ValueError(f"Feature record absent from package manifest: {artifact.path}")
        verify_file(results, artifact)
    with (results / "audio" / "manifest.jsonl").open(encoding="utf-8") as stream:
        audio = tuple(AudioArtifact.model_validate_json(line) for line in stream)
    if audio != package.audio:
        raise ValueError("Audio manifest differs from package audio records")
    if {record.example_id for record in audio} != {record.path.stem for record in features}:
        raise ValueError("Audited audio and feature example IDs differ")
    for artifact in audio:
        if hashlib.sha256((results / artifact.path).read_bytes()).hexdigest() != artifact.sha256:
            raise ValueError(f"Audio SHA256 differs from sample manifest: {artifact.path}")
    return VerificationReport(
        verified_at=datetime.now(timezone.utc),
        copied_files=len(package.artifacts),
        bytes_verified=total_bytes,
        audio_examples=len(audio),
        feature_files=len(features),
        package_manifest_sha256=hashlib.sha256(serialized).hexdigest(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    arguments = parser.parse_args()
    report = verify_package(arguments.results)
    destination = arguments.results / "reproducibility" / "package_verification.json"
    partial = destination.with_suffix(".part")
    partial.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    partial.replace(destination)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
