"""Download audio from a fixed experiment manifest without regenerating examples."""

import argparse
import hashlib
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from speech_projector.data import download_audio, load_examples
from speech_projector.models import Record, Split

MINIMUM_FREE_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class DownloadConfiguration:
    root: Path
    train_examples: int
    workers: int


class DownloadFailure(Record):
    example_id: str
    error: str


class DownloadProgress(Record):
    manifest_sha256: str
    started_at: datetime
    updated_at: datetime
    training_examples: int
    heldout_examples: int
    existing_examples: int
    processed_examples: int
    completed_examples: int
    newly_downloaded_examples: int
    newly_downloaded_bytes: int
    elapsed_seconds: float
    free_disk_bytes: int
    failures: tuple[DownloadFailure, ...]


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while contents := stream.read(1024 * 1024):
            digest.update(contents)
    return digest.hexdigest()


def save_progress(root: Path, progress: DownloadProgress) -> None:
    destination = root / "download_expansion.json"
    partial = destination.with_suffix(".part")
    partial.write_text(progress.model_dump_json(indent=2), encoding="utf-8")
    partial.replace(destination)


def download_manifest(configuration: DownloadConfiguration) -> None:
    if configuration.train_examples <= 0 or configuration.workers <= 0:
        raise ValueError("Training-example and download-worker counts must be positive")
    if shutil.disk_usage(configuration.root).free < MINIMUM_FREE_BYTES:
        raise ValueError("Less than 2 GiB of free disk remains for training checkpoints")
    manifest = configuration.root / "examples.jsonl"
    manifest_digest = file_digest(manifest)
    training = load_examples(manifest, Split.TRAIN, configuration.train_examples)
    if len(training) != configuration.train_examples:
        raise ValueError("Requested training subset exceeds the fixed manifest")
    heldout = load_examples(manifest, Split.VALIDATION) + load_examples(manifest, Split.TEST)
    examples = heldout + training
    existing = sum(example.audio_path.exists() for example in examples)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc)
    failures: list[DownloadFailure] = []
    completed = 0
    downloaded = 0
    total_bytes = 0
    print(
        f"Fixed manifest {manifest_digest}: selected={len(examples)}, existing={existing}, "
        f"missing={len(examples) - existing}, workers={configuration.workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=configuration.workers) as executor:
        results = executor.map(partial(download_audio, root=configuration.root), examples)
        for processed, result in enumerate(results, 1):
            if result.success:
                completed += 1
                if result.bytes_downloaded:
                    downloaded += 1
                    total_bytes += result.bytes_downloaded
            else:
                assert result.error is not None
                failures.append(DownloadFailure(example_id=result.example_id, error=result.error))
                print(f"Failed {result.example_id}: {result.error}", flush=True)
            if processed % 100 == 0 or processed == len(examples):
                progress = DownloadProgress(
                    manifest_sha256=manifest_digest,
                    started_at=started_at,
                    updated_at=datetime.now(timezone.utc),
                    training_examples=len(training),
                    heldout_examples=len(heldout),
                    existing_examples=existing,
                    processed_examples=processed,
                    completed_examples=completed,
                    newly_downloaded_examples=downloaded,
                    newly_downloaded_bytes=total_bytes,
                    elapsed_seconds=time.monotonic() - started,
                    free_disk_bytes=shutil.disk_usage(configuration.root).free,
                    failures=tuple(failures),
                )
                save_progress(configuration.root, progress)
                print(progress.model_dump_json(), flush=True)
                if progress.free_disk_bytes < MINIMUM_FREE_BYTES:
                    raise ValueError("Download stopped to preserve 2 GiB for training checkpoints")
    if file_digest(manifest) != manifest_digest:
        raise ValueError("The fixed experiment manifest changed during audio download")
    if failures:
        raise ValueError(
            f"{len(failures)} clips failed; supervisor restart will retry missing files"
        )
    print("All selected audio downloaded; fixed manifest preserved", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train-examples", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    arguments = parser.parse_args()
    download_manifest(
        DownloadConfiguration(
            root=arguments.root,
            train_examples=arguments.train_examples,
            workers=arguments.workers,
        )
    )


if __name__ == "__main__":
    main()
