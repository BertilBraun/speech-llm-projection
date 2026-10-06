"""CPU import/CLI validation in the explicitly configured isolated vLLM runtime."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.skipif(
    "EMOTIONAL_VLLM_PYTHON" not in os.environ, reason="Isolated vLLM runtime not configured"
)
def test_isolated_vllm_entrypoint_imports_and_exposes_shared_pipeline() -> None:
    executable = Path(os.environ["EMOTIONAL_VLLM_PYTHON"])
    process = subprocess.run(
        [str(executable), "-m", "scripts.generate_emotional_vllm", "--help"],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "--config" in process.stdout
    assert "drafts" in process.stdout and "targets" in process.stdout
