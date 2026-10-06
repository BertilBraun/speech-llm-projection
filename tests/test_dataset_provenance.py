import hashlib
from pathlib import Path

import pytest
from huggingface_hub.hf_api import BlobLfsInfo, RepoFile

from scripts.package_results import METADATA_REPOSITORY_PATH, verify_dataset_source


def metadata_for(content: bytes) -> RepoFile:
    metadata = RepoFile(path=METADATA_REPOSITORY_PATH, size=len(content), oid="git-blob")
    metadata.lfs = BlobLfsInfo(
        size=len(content), sha256=hashlib.sha256(content).hexdigest(), pointer_size=130
    )
    return metadata


def test_dataset_source_pins_matching_parquet(tmp_path: Path) -> None:
    content = b"parquet metadata"
    path = tmp_path / "metadata.parquet"
    path.write_bytes(content)
    revision = "a" * 40
    source = verify_dataset_source(path, revision, metadata_for(content))
    assert source.dataset_revision == revision
    assert source.metadata_bytes == len(content)
    assert source.metadata_sha256 == source.metadata_lfs_sha256
    assert f"/resolve/{revision}/" in source.metadata_url
    assert "/resolve/main/" in source.original_download_url


@pytest.mark.parametrize("mismatch", ["hash", "size", "missing_lfs", "path"])
def test_dataset_source_rejects_mismatched_metadata(tmp_path: Path, mismatch: str) -> None:
    content = b"parquet metadata"
    path = tmp_path / "metadata.parquet"
    path.write_bytes(content)
    metadata = metadata_for(content)
    match mismatch:
        case "hash":
            path.write_bytes(b"Parquet metadata")
        case "size":
            path.write_bytes(content + b"extra")
        case "missing_lfs":
            metadata.lfs = None
        case "path":
            metadata.path = "wrong.parquet"
    with pytest.raises(ValueError):
        verify_dataset_source(path, "a" * 40, metadata)
