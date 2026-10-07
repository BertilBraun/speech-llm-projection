import shutil
from pathlib import Path

import pytest

from scripts.inventory_results import verify_inventory, write_inventory, write_record
from speech_projector.models import AsrTranscript


def test_serialized_record_has_portable_exact_utf8_lf_bytes(tmp_path: Path) -> None:
    record = AsrTranscript(example_id="speech", text="First line\nSecond line — café")
    path = tmp_path / "record.json"
    write_record(path, record)
    serialized = path.read_bytes()
    assert serialized == record.model_dump_json(indent=2).encode("utf-8")
    assert serialized.count(b"\n") >= 3
    assert b"\r\n" not in serialized
    assert not path.with_suffix(".part").exists()


def test_inventory_survives_relocation(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "checkpoint.pt").write_bytes(b"checkpoint")
    (remote / "metrics.json").write_bytes(b'{"loss":1.2}')
    inventory = write_inventory(remote, writers_stopped=True)
    local = tmp_path / "local"
    shutil.copytree(remote, local)
    first = verify_inventory(local)
    second = verify_inventory(local)
    assert first.files == second.files == len(inventory.artifacts) == 2
    assert first.bytes_verified == len(b"checkpoint") + len(b'{"loss":1.2}')


@pytest.mark.parametrize("mutation", ["changed", "added", "removed"])
def test_verifier_rejects_changed_tree(tmp_path: Path, mutation: str) -> None:
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"checkpoint")
    write_inventory(tmp_path, writers_stopped=True)
    match mutation:
        case "changed":
            artifact.write_bytes(b"Checkpoint")
        case "added":
            (tmp_path / "unexpected.json").write_text("{}", encoding="utf-8")
        case "removed":
            artifact.unlink()
    with pytest.raises(ValueError):
        verify_inventory(tmp_path)


def test_inventory_requires_stopped_writers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="writers-stopped"):
        write_inventory(tmp_path, writers_stopped=False)
