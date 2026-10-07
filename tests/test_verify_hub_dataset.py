"""Remote publication verification must reject changed bytes and incorrect sizes."""

import hashlib
from pathlib import Path

import pytest
from huggingface_hub.hf_api import BlobLfsInfo, RepoFile

from scripts.verify_hub_dataset import verify_file


def git_file() -> RepoFile:
    return RepoFile(path="README.md", size=4, oid=hashlib.sha1(b"blob 4\0data").hexdigest())


@pytest.mark.parametrize("lfs", (False, True))
def test_verifies_exact_git_and_lfs_payload(tmp_path: Path, lfs: bool) -> None:
    path = tmp_path / "payload"
    path.write_bytes(b"data")
    remote = git_file()
    if lfs:
        remote.lfs = BlobLfsInfo(
            size=4, sha256=hashlib.sha256(b"data").hexdigest(), pointer_size=100
        )
    assert verify_file(path, remote) == 4


@pytest.mark.parametrize("mismatch", ("file_size", "git_hash", "lfs_hash", "lfs_size"))
def test_rejects_remote_payload_mismatch(tmp_path: Path, mismatch: str) -> None:
    path = tmp_path / "payload"
    path.write_bytes(b"data")
    remote = git_file()
    match mismatch:
        case "file_size":
            remote.size = 5
        case "git_hash":
            remote.blob_id = "0" * 40
        case "lfs_hash":
            remote.lfs = BlobLfsInfo(size=4, sha256="0" * 64, pointer_size=100)
        case "lfs_size":
            remote.lfs = BlobLfsInfo(
                size=5, sha256=hashlib.sha256(b"data").hexdigest(), pointer_size=100
            )
    with pytest.raises(ValueError, match="mismatch"):
        verify_file(path, remote)
