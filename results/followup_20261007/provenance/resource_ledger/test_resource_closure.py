"""Resource scopes reject unfinished or overlapping evidence instead of guessing."""

from pathlib import Path

import pytest

from artifacts.followup10hz.resource_closure import ProcessScope, process_intervals


def test_cpu_overlap_does_not_duplicate_gpu_process_wall(tmp_path: Path) -> None:
    path = tmp_path / "lifecycle.log"
    path.write_text(
        "2026-10-07 08:00:00,000 INFO spawned: 'followup10hz-fullpass' with pid 1\n"
        "2026-10-07 08:00:01,000 INFO spawned: 'followup10hz-tone-classifier' with pid 2\n"
        "2026-10-07 08:00:03,000 INFO exited: followup10hz-tone-classifier "
        "(exit status 0; expected)\n"
        "2026-10-07 08:00:10,000 INFO exited: followup10hz-fullpass (exit status 0; expected)\n",
        encoding="utf-8",
    )
    rows = process_intervals(path)
    assert sum(row.wall_seconds for row in rows if row.scope != ProcessScope.CPU) == 10
    assert sum(row.wall_seconds for row in rows if row.scope == ProcessScope.CPU) == 2


@pytest.mark.parametrize("failure", ["unfinished", "missing_start", "negative_duration", "overlap"])
def test_invalid_process_evidence_is_not_summed(tmp_path: Path, failure: str) -> None:
    started = "2026-10-07 08:00:01,000 INFO spawned: 'followup10hz-fullpass' with pid 1\n"
    exited = (
        "2026-10-07 08:00:10,000 INFO exited: followup10hz-fullpass (exit status 0; expected)\n"
    )
    match failure:
        case "unfinished":
            lines = started
        case "missing_start":
            lines = exited
        case "negative_duration":
            lines = started + exited.replace("08:00:10", "08:00:00")
        case "overlap":
            lines = (
                started
                + "2026-10-07 08:00:02,000 INFO spawned: 'followup10hz-tone-baseline' with pid 2\n"
                + "2026-10-07 08:00:09,000 INFO exited: followup10hz-tone-baseline "
                "(exit status 0; expected)\n" + exited
            )
    path = tmp_path / "lifecycle.log"
    path.write_text(lines, encoding="utf-8")
    with pytest.raises(ValueError):
        process_intervals(path)


def test_failed_attempt_duration_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "lifecycle.log"
    path.write_text(
        "2026-10-07 08:00:00,000 INFO spawned: 'followup10hz-ordinarykl-smoke' with pid 1\n"
        "2026-10-07 08:00:03,000 INFO exited: followup10hz-ordinarykl-smoke "
        "(exit status 1; not expected)\n",
        encoding="utf-8",
    )
    row = process_intervals(path)[0]
    assert row.exit_status == 1 and row.wall_seconds == 3
    assert row.scope == ProcessScope.SMOKE
