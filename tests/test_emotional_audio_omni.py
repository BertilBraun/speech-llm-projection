"""CPU validation in the explicitly configured isolated Omni runtime."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.skipif(
    "EMOTIONAL_OMNI_PYTHON" not in os.environ, reason="Isolated Omni runtime not configured"
)
def test_isolated_omni_synthesis_and_resume_contracts() -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    process = subprocess.run(
        [
            os.environ["EMOTIONAL_OMNI_PYTHON"],
            "-B",
            "-m",
            "pytest",
            "-q",
            "tests/omni_runtime_checks.py",
        ],
        cwd=Path(__file__).parents[1],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "8 passed" in process.stdout
