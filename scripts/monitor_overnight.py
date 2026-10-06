"""Record sampled GPU/disk use for this study without accessing any model runtime."""

import argparse
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import Field

from speech_projector.journal import append_record
from speech_projector.models import Record


class OvernightResourceObservation(Record):
    captured_at: datetime
    gpu_used_mib: int = Field(ge=0)
    gpu_total_mib: int = Field(gt=0)
    gpu_utilization_percent: int = Field(ge=0, le=100)
    temperature_celsius: int
    power_watts: float = Field(ge=0)
    disk_free_bytes: int = Field(ge=0)
    supervised_jobs: tuple[str, ...]
    timing_scope: str = "Periodic point observations; not a continuous physical-memory peak."


def observe(workspace: Path, jobs: tuple[str, ...]) -> OvernightResourceObservation:
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = query.splitlines()
    if len(rows) != 1:
        raise ValueError("This study requires exactly one visible GPU")
    used, total, utilization, temperature, power = rows[0].split(",")
    statuses = subprocess.run(
        [
            "supervisorctl",
            "status",
            *jobs,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return OvernightResourceObservation(
        captured_at=datetime.now(timezone.utc),
        gpu_used_mib=int(used),
        gpu_total_mib=int(total),
        gpu_utilization_percent=int(utilization),
        temperature_celsius=int(temperature),
        power_watts=float(power),
        disk_free_bytes=shutil.disk_usage(workspace).free,
        supervised_jobs=tuple(statuses.stdout.splitlines()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", nargs="+", required=True)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    if arguments.interval <= 0:
        raise ValueError("Observation interval must be positive")
    while True:
        append_record(arguments.output, observe(arguments.workspace, tuple(arguments.jobs)))
        if arguments.once:
            break
        time.sleep(arguments.interval)


if __name__ == "__main__":
    main()
