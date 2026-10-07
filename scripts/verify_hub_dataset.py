"""Verify a public immutable Hub revision against every local release file."""

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile, RepoFolder

from scripts.inventory_results import stable_digest, write_record
from speech_projector.models import Record


class HubDatasetVerification(Record):
    verified_at: datetime
    repository: str
    revision: str
    files: int
    bytes_verified: int
    parquet_shards: int
    lfs_files: int
    git_files: int


def verify_file(path: Path, remote: RepoFile) -> int:
    size, digest = stable_digest(path)
    if size != remote.size:
        raise ValueError(f"Remote file size mismatch: {remote.path}")
    if remote.lfs is not None:
        if digest != remote.lfs.sha256 or size != remote.lfs.size:
            raise ValueError(f"Remote LFS SHA256 mismatch: {remote.path}")
    else:
        git_digest = hashlib.sha1(f"blob {size}\0".encode() + path.read_bytes()).hexdigest()
        if git_digest != remote.blob_id:
            raise ValueError(f"Remote Git blob mismatch: {remote.path}")
    return size


def verify_hub_dataset(directory: Path, repository: str) -> HubDatasetVerification:
    api = HfApi(token=False)
    information = api.dataset_info(repository)
    if information.private or information.sha is None:
        raise ValueError("Dataset must have a public committed revision")
    remote: dict[str, RepoFile] = {}
    for item in api.list_repo_tree(
        repository, repo_type="dataset", revision=information.sha, recursive=True
    ):
        match item:
            case RepoFile():
                remote[item.path] = item
            case RepoFolder():
                pass
    local = tuple(
        path
        for path in directory.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(directory).parts
    )
    expected = {path.relative_to(directory).as_posix() for path in local}
    if set(remote) - {".gitattributes"} != expected:
        raise ValueError("Remote file coverage differs from the local public release")
    total_bytes = 0
    lfs_files = 0
    for path in local:
        item = remote[path.relative_to(directory).as_posix()]
        total_bytes += verify_file(path, item)
        if item.lfs is not None:
            lfs_files += 1
    return HubDatasetVerification(
        verified_at=datetime.now(timezone.utc),
        repository=repository,
        revision=information.sha,
        files=len(local),
        bytes_verified=total_bytes,
        parquet_shards=sum(path.suffix == ".parquet" for path in local),
        lfs_files=lfs_files,
        git_files=len(local) - lfs_files,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    verification = verify_hub_dataset(arguments.directory, arguments.repository)
    write_record(arguments.output, verification)
    print(
        f"Verified public {verification.repository}: {verification.files} files, "
        f"{verification.bytes_verified / 1e9:.3g} GB at {verification.revision}"
    )


if __name__ == "__main__":
    main()
