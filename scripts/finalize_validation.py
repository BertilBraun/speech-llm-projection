"""Preserve measured resources and record repaired validation failures."""

import argparse
import time
from pathlib import Path

from speech_projector.models import ExperimentFailure, RunResult
from speech_projector.training import TrainingResourceRecord


def finalize(result_root: Path) -> None:
    directory = result_root / "v0_256_mlp_10hz"
    original = RunResult.model_validate_json((directory / "result_wrong_eos.json").read_bytes())
    corrected = RunResult.model_validate_json((directory / "result.json").read_bytes())
    corrected = corrected.model_copy(
        update={"peak_vram_gb": max(original.peak_vram_gb, corrected.peak_vram_gb)}
    )
    (directory / "result.json").write_text(corrected.model_dump_json(indent=2), encoding="utf-8")
    resources = TrainingResourceRecord(peak_vram_gb=corrected.peak_vram_gb)
    (directory / "training_resources.json").write_text(
        resources.model_dump_json(indent=2), encoding="utf-8"
    )
    recorded_at = time.time()
    failures = (
        ExperimentFailure(
            run_name="environment_causal_conv_wheel",
            attempt=1,
            exception_type="ImportError",
            message=(
                "Downloaded causal-conv1d wheel had an incompatible PyTorch C++ ABI. "
                "Resolved by compiling version 1.5.2 for installed PyTorch 2.6/cu124 and sm86."
            ),
            traceback="The initial extension import reported an undefined C++ symbol.",
            timestamp=recorded_at,
        ),
        ExperimentFailure(
            run_name="audio_download_initial_concurrency",
            attempt=1,
            exception_type="DownloadFailure",
            message=(
                "Twelve audio requests exhausted initial retries with 12 download workers. "
                "Reduced concurrency to 4 and resumed existing files; all 10k training clips "
                "and held-out clips subsequently downloaded and passed leakage audits."
            ),
            traceback="Progress and retries retained in data/download.log.",
            timestamp=recorded_at,
        ),
        ExperimentFailure(
            run_name="v0_generation_stop_token",
            attempt=1,
            exception_type="EvaluationBug",
            message=(
                "Qwen text config EOS 248044 differed from tokenizer chat EOS 248046. "
                "Generated replies could continue into invented roles. Fixed tokenizer EOS, "
                "archived invalid generations, and re-evaluated the saved projector. "
                "Training and teacher-forced loss were unchanged."
            ),
            traceback="Invalid results are preserved in *_wrong_eos archives.",
            timestamp=recorded_at,
        ),
        ExperimentFailure(
            run_name="generation_left_padding_optimization",
            attempt=1,
            exception_type="AssertionError",
            message=(
                "An initial left-padding parity test diverged before the chat EOS repair. "
                "Optimization deferred; production generation uses actual unpadded prompts."
            ),
            traceback="Reproduction script: scripts/compare_generation_padding.py.",
            timestamp=recorded_at,
        ),
        ExperimentFailure(
            run_name="training_probe_first_launch",
            attempt=1,
            exception_type="ModuleNotFoundError",
            message=(
                "Direct helper script could not import the uninstalled project package. "
                "Installed project editable without replacing dependencies; probe completed."
            ),
            traceback="Original import traces retained in probe_error.log.",
            timestamp=recorded_at,
        ),
    )
    destination = result_root / "validation_failures.jsonl"
    destination.write_text(
        "\n".join(failure.model_dump_json() for failure in failures) + "\n", encoding="utf-8"
    )
    print(f"Recorded {len(failures)} repaired/deferred validation failures", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    arguments = parser.parse_args()
    finalize(arguments.results)


if __name__ == "__main__":
    main()
