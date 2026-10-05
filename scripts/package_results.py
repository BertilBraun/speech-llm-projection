"""Package lightweight provenance and auditable audio inputs beside experiment results."""

import argparse
import hashlib
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import Distribution, distributions
from pathlib import Path

from speech_projector.cache import CacheStatistics, load_asr
from speech_projector.data import DATASET_URL, load_examples
from speech_projector.models import Record, Split

DATASET_ARTIFACTS = (
    "dataset_report.json",
    "dataset_quality.json",
    "dataset_leakage.json",
    "asr_quality.json",
    "asr_transcripts.jsonl",
    "cache_history.jsonl",
    "examples.jsonl",
)


@dataclass(frozen=True)
class PackageConfiguration:
    dataset_root: Path
    results_root: Path
    hugging_face_hub: Path


class FileArtifact(Record):
    path: Path
    source_path: Path
    bytes: int
    sha256: str


class AudioArtifact(Record):
    example_id: str
    dialogue_id: str
    path: Path
    duration: float
    transcript: str
    sha256: str


class ModelRevision(Record):
    model_name: str
    snapshot_revisions: tuple[str, ...]
    main_revision: str | None


class InstalledPackage(Record):
    name: str
    version: str


class EnvironmentInventory(Record):
    python_version: str
    python_implementation: str
    python_executable: Path
    operating_system: str
    machine: str
    packages: tuple[InstalledPackage, ...]


class DatasetSource(Record):
    metadata_url: str
    metadata_bytes: int
    metadata_sha256: str


class CacheSummary(Record):
    largest_completed_cache: CacheStatistics
    measurements: tuple[CacheStatistics, ...]
    current_feature_file_count: int
    current_feature_file_bytes: int
    recorded_encoder_seconds: float
    recorded_asr_seconds: float
    recorded_gpu_synchronized_wall_hours: float
    measured_pipeline_wall_seconds: float
    unmeasured_pipeline_wall_records: int
    timing_interpretation: str


class PackageManifest(Record):
    artifacts: tuple[FileArtifact, ...]
    audio: tuple[AudioArtifact, ...]
    model_revisions: tuple[ModelRevision, ...]
    copied_artifact_bytes: int


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while content := stream.read(1024 * 1024):
            digest.update(content)
    return digest.hexdigest()


def write_bytes_if_changed(path: Path, contents: bytes) -> None:
    if path.exists() and path.read_bytes() == contents:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_bytes(contents)
    partial.replace(path)


def write_record(path: Path, record: Record) -> None:
    write_bytes_if_changed(path, (record.model_dump_json(indent=2) + "\n").encode())


def copy_artifact(
    source: Path, relative_destination: Path, configuration: PackageConfiguration
) -> FileArtifact:
    destination = configuration.results_root / relative_destination
    digest = file_digest(source)
    size = source.stat().st_size
    if (
        not destination.exists()
        or destination.stat().st_size != size
        or file_digest(destination) != digest
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        shutil.copyfile(source, partial)
        partial.replace(destination)
    return FileArtifact(path=relative_destination, source_path=source, bytes=size, sha256=digest)


def copy_dataset(configuration: PackageConfiguration) -> list[FileArtifact]:
    artifacts = [
        copy_artifact(
            configuration.dataset_root / name,
            Path("dataset") / name,
            configuration,
        )
        for name in DATASET_ARTIFACTS
    ]
    expansion = configuration.dataset_root / "download_expansion.json"
    if expansion.exists():
        artifacts.append(copy_artifact(expansion, Path("dataset") / expansion.name, configuration))
    source = configuration.dataset_root / "metadata.parquet"
    write_record(
        configuration.results_root / "dataset" / "source.json",
        DatasetSource(
            metadata_url=DATASET_URL + "data/train-00000-of-00001.parquet",
            metadata_bytes=source.stat().st_size,
            metadata_sha256=file_digest(source),
        ),
    )
    return artifacts


def package_audio(
    configuration: PackageConfiguration,
) -> tuple[list[FileArtifact], list[AudioArtifact]]:
    examples = load_examples(configuration.dataset_root / "examples.jsonl", Split.VALIDATION, 16)
    transcriptions = {
        record.example_id: record.text
        for record in load_asr(configuration.dataset_root / "asr_transcripts.jsonl")
    }
    files: list[FileArtifact] = []
    audio: list[AudioArtifact] = []
    lines = [
        "# Fixed validation audio inputs",
        "",
        "Original dataset WAV files for the first 16 fixed validation examples. "
        "These clips are also used in qualitative evaluation.",
        "",
    ]
    for example in examples:
        relative_path = Path("audio") / f"{example.example_id}.wav"
        artifact = copy_artifact(example.audio_path, relative_path, configuration)
        files.append(artifact)
        audio.append(
            AudioArtifact(
                example_id=example.example_id,
                dialogue_id=example.dialogue_id,
                path=relative_path,
                duration=example.duration,
                transcript=example.user_text,
                sha256=artifact.sha256,
            )
        )
        lines.extend(
            (
                f"## {example.example_id} · {example.domain} · {example.duration:.3f} seconds",
                "",
                f"[Audio clip]({example.example_id}.wav)",
                "",
                f"Dataset transcript: {example.user_text}",
                "",
                f"Whisper transcription: {transcriptions[example.example_id]}",
                "",
            )
        )
    manifest = "".join(record.model_dump_json() + "\n" for record in audio)
    write_bytes_if_changed(
        configuration.results_root / "audio" / "manifest.jsonl", manifest.encode()
    )
    write_bytes_if_changed(
        configuration.results_root / "audio" / "index.md", "\n".join(lines).encode()
    )
    return files, audio


def package_features(configuration: PackageConfiguration) -> list[FileArtifact]:
    examples = load_examples(configuration.dataset_root / "examples.jsonl", Split.VALIDATION, 16)
    artifacts = [
        copy_artifact(
            example.feature_path,
            Path("features") / f"{example.example_id}.pt",
            configuration,
        )
        for example in examples
    ]
    manifest = "".join(record.model_dump_json() + "\n" for record in artifacts)
    write_bytes_if_changed(
        configuration.results_root / "features" / "manifest.jsonl", manifest.encode()
    )
    return artifacts


def model_revisions(hub: Path) -> list[ModelRevision]:
    revisions: list[ModelRevision] = []
    for repository in sorted(hub.glob("models--*")):
        snapshots = repository / "snapshots"
        main_reference = repository / "refs" / "main"
        revisions.append(
            ModelRevision(
                model_name=repository.name.removeprefix("models--").replace("--", "/"),
                snapshot_revisions=tuple(
                    sorted(path.name for path in snapshots.iterdir() if path.is_dir())
                ),
                main_revision=main_reference.read_text().strip()
                if main_reference.exists()
                else None,
            )
        )
    return revisions


def installed_package(distribution: Distribution) -> InstalledPackage:
    return InstalledPackage(name=distribution.metadata["Name"], version=distribution.version)


def package_order(package: InstalledPackage) -> str:
    return package.name.lower()


def package_environment(configuration: PackageConfiguration) -> list[ModelRevision]:
    revisions = model_revisions(configuration.hugging_face_hub)
    serialized = "".join(record.model_dump_json() + "\n" for record in revisions)
    write_bytes_if_changed(
        configuration.results_root / "reproducibility" / "model_revisions.jsonl",
        serialized.encode(),
    )
    inventory = EnvironmentInventory(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        python_executable=Path(sys.executable),
        operating_system=platform.platform(),
        machine=platform.machine(),
        packages=tuple(
            sorted(
                (installed_package(distribution) for distribution in distributions()),
                key=package_order,
            )
        ),
    )
    write_record(configuration.results_root / "reproducibility" / "environment.json", inventory)
    if not (configuration.results_root / "environment.txt").exists():
        frozen = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True
        )
        write_bytes_if_changed(
            configuration.results_root / "environment.txt", frozen.stdout.encode()
        )
    return revisions


def cache_order(statistics: CacheStatistics) -> int:
    return statistics.feature_count


def package_cache(configuration: PackageConfiguration) -> list[FileArtifact]:
    measurements: list[CacheStatistics] = []
    history = configuration.dataset_root / "cache_history.jsonl"
    for line in history.read_text().splitlines():
        measurement = CacheStatistics.model_validate_json(line)
        if measurement not in measurements:
            measurements.append(measurement)
    artifacts: list[FileArtifact] = []
    current: list[CacheStatistics] = []
    for path in sorted(configuration.dataset_root.glob("cache_stats_*.json")):
        measurement = CacheStatistics.model_validate_json(path.read_text())
        if measurement not in measurements:
            measurements.append(measurement)
        if path.name.removeprefix("cache_stats_").split(".")[0].isdigit():
            current.append(measurement)
        artifacts.append(copy_artifact(path, Path("dataset") / path.name, configuration))
    largest = max(current, key=cache_order)
    features = tuple((configuration.dataset_root / "features").rglob("*.pt"))
    encoder_seconds = sum(record.extraction_seconds for record in measurements)
    asr_seconds = sum(record.asr_seconds for record in measurements)
    summary = CacheSummary(
        largest_completed_cache=largest,
        measurements=tuple(measurements),
        current_feature_file_count=len(features),
        current_feature_file_bytes=sum(path.stat().st_size for path in features),
        recorded_encoder_seconds=encoder_seconds,
        recorded_asr_seconds=asr_seconds,
        recorded_gpu_synchronized_wall_hours=(encoder_seconds + asr_seconds) / 3600,
        measured_pipeline_wall_seconds=sum(
            record.cache_wall_seconds
            for record in measurements
            if record.cache_wall_seconds is not None
        ),
        unmeasured_pipeline_wall_records=sum(
            record.cache_wall_seconds is None for record in measurements
        ),
        timing_interpretation=(
            "Encoder/ASR times are CUDA-synchronized wall measurements, not kernel profiler "
            "measurements. Pipeline wall time additionally includes audio loading, CPU feature "
            "processing and feature persistence, and excludes model loading/downloads. Initial "
            "pipeline wall time was unmeasured (explicit null). Measurements deduplicated across "
            "history and per-subset files. Actual feature bytes include unused replaced clips "
            "and any newly extracted files from currently running jobs."
        ),
    )
    write_record(configuration.results_root / "dataset" / "cache_summary.json", summary)
    return artifacts


def package_results(configuration: PackageConfiguration) -> PackageManifest:
    files = copy_dataset(configuration)
    audio_files, audio = package_audio(configuration)
    files.extend(audio_files)
    files.extend(package_features(configuration))
    files.extend(package_cache(configuration))
    revisions = package_environment(configuration)
    manifest = PackageManifest(
        artifacts=tuple(files),
        audio=tuple(audio),
        model_revisions=tuple(revisions),
        copied_artifact_bytes=sum(record.bytes for record in files),
    )
    write_record(configuration.results_root / "package_manifest.json", manifest)
    print(manifest.model_dump_json(indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--hub", type=Path, required=True)
    arguments = parser.parse_args()
    package_results(
        PackageConfiguration(
            dataset_root=arguments.data,
            results_root=arguments.results,
            hugging_face_hub=arguments.hub,
        )
    )


if __name__ == "__main__":
    main()
