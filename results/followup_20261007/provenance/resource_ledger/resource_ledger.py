"""Measured resource scopes from completed copies, never from active training files."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from artifacts.followup10hz.backup_completed_run import CompletedRunCapture
from scripts.inventory_results import stable_digest, write_record
from speech_projector.models import FileArtifact, Record
from speech_projector.overnight_continuation import ContinuationProvenance
from speech_projector.tone_classifier import ToneClassifierReport
from speech_projector.training import TrainingState


class TrainingIncrement(Record):
    name: str
    source_commit: str
    cumulative_updates: int
    added_updates: int
    cumulative_examples: int
    added_examples: int
    cumulative_target_tokens: int
    added_target_tokens: int
    cumulative_training_seconds: float
    added_training_seconds: float
    peak_pytorch_allocated_gb: float
    capture: FileArtifact
    parent_state: FileArtifact


class ResourceLedger(Record):
    captured_at: datetime
    completed_training: tuple[TrainingIncrement, ...]
    classifier_report: FileArtifact
    classifier_cpu_wall_seconds: float
    classifier_feature_loading_seconds: float
    classifier_fitting_seconds: float
    scope_notes: tuple[str, ...]


def evidence(path: Path) -> FileArtifact:
    size, digest = stable_digest(path)
    return FileArtifact(path=path, source_path=path.resolve(), bytes=size, sha256=digest)


def training_increment(capture_path: Path, followup: Path, previous: Path) -> TrainingIncrement:
    capture = CompletedRunCapture.model_validate_json(capture_path.read_bytes())
    run = followup / "runs" / capture.result.config.name
    continuation = run / "continuation.json"
    branch = run / "objective_branch.json"
    if continuation.exists() == branch.exists():
        raise ValueError("Captured run must have exactly one canonical continuation provenance")
    provenance = ContinuationProvenance.model_validate_json(
        (continuation if continuation.exists() else branch).read_bytes()
    )
    states = tuple(
        entry for entry in provenance.source_artifacts if entry.path.name == "state.json"
    )
    if len(states) != 1:
        raise ValueError("Continuation provenance must bind one exact parent training state")
    entry = states[0]
    relative = entry.source_path.relative_to(Path("/workspace"))
    if relative.parts[:2] == ("speech-projector", "results_followup_20261007"):
        parent_path = followup / "runs" / Path(*relative.parts[2:])
    else:
        parent_path = previous / relative
    if stable_digest(parent_path) != (entry.bytes, entry.sha256):
        raise ValueError("Restored parent state differs from continuation provenance")
    parent = TrainingState.model_validate_json(parent_path.read_bytes())
    state = capture.state
    if state.step <= parent.step or state.elapsed_seconds < parent.elapsed_seconds:
        raise ValueError("Captured continuation does not advance its preserved parent")
    if abs(capture.result.runtime_seconds - state.elapsed_seconds) > 1e-6:
        raise ValueError("Result timing disagrees with preserved final training state")
    return TrainingIncrement(
        name=capture.result.config.name,
        source_commit=capture.result.git_commit,
        cumulative_updates=state.step,
        added_updates=state.step - parent.step,
        cumulative_examples=state.examples_seen,
        added_examples=state.examples_seen - parent.examples_seen,
        cumulative_target_tokens=state.target_tokens_seen,
        added_target_tokens=state.target_tokens_seen - parent.target_tokens_seen,
        cumulative_training_seconds=state.elapsed_seconds,
        added_training_seconds=state.elapsed_seconds - parent.elapsed_seconds,
        peak_pytorch_allocated_gb=capture.result.peak_vram_gb,
        capture=evidence(capture_path),
        parent_state=evidence(parent_path),
    )


def build_ledger(followup: Path, previous: Path) -> ResourceLedger:
    classifier_path = followup / "tone_classifier/report.json"
    classifier = ToneClassifierReport.model_validate_json(classifier_path.read_bytes())
    captures = tuple(sorted((followup / "runs/transfer").glob("*_capture.json")))
    return ResourceLedger(
        captured_at=datetime.now(timezone.utc),
        completed_training=tuple(training_increment(path, followup, previous) for path in captures),
        classifier_report=evidence(classifier_path),
        classifier_cpu_wall_seconds=classifier.total_wall_seconds,
        classifier_feature_loading_seconds=classifier.feature_loading_seconds,
        classifier_fitting_seconds=classifier.fitting_seconds,
        scope_notes=(
            "Training increments subtract the exact SHA-bound parent checkpoint elapsed time; "
            "cumulative counters are references and must not be summed again.",
            "Training-loop wall includes recorded training overhead; it is not measured GPU "
            "utilization, CUDA kernel time, or the whole process including startup/evaluation.",
            "Classifier total CPU wall contains feature loading and fitting; nested timers "
            "must not be added to its total.",
            "Classifier memory observation was RSS 1,876,056 KiB at one instant, not a peak; "
            "its 10000×4608 float64 feature matrix occupies 368,640,000 bytes before copies.",
            "PyTorch allocated decimal GB is distinct from reserved memory and NVIDIA/NVML usage.",
            "Saved training peak is a lineage maximum: continuation copies training_resources.json "
            "and preserves max(previous peak, newly observed peak), so it can include the parent "
            "rather than measuring only the new branch.",
            "No new training-feature cache extraction, audio synthesis or teacher generation; "
            "three existing neutral follow-up clips (followup_short, next_step, stop) were "
            "transcribed once for the conversation comparator and reused thereafter. "
            "Their Whisper encoder/decoder runtime is contained in the conversation-job "
            "scope, not a separate extraction benchmark.",
            "Final inference/evaluation, smoke/startup/debug and rental-time measurements "
            "remain separate or unmeasured here. No rental rate or cost is inferred.",
            "Only completed immutable run captures are read; active branches are pending.",
        ),
    )


def render(ledger: ResourceLedger) -> str:
    lines = [
        "# Follow-up resource ledger (completed artifacts only)",
        "",
        f"Captured {ledger.captured_at.isoformat()}.",
        "",
        "| Run | Added updates/examples/target tokens | Added training-loop s | "
        "Cumulative training s (do not add) | Saved lineage peak allocated decimal GB |",
        "|---|---|---:|---:|---:|",
    ]
    for row in ledger.completed_training:
        lines.append(
            f"| {row.name} | {row.added_updates}/{row.added_examples}/{row.added_target_tokens} | "
            f"{row.added_training_seconds:.6f} | {row.cumulative_training_seconds:.6f} | "
            f"{row.peak_pytorch_allocated_gb:.6f} |"
        )
    lines.extend(
        (
            "",
            f"Total distinct completed training increments: "
            f"{sum(row.added_training_seconds for row in ledger.completed_training):.6f} s.",
            "",
            f"Classifier CPU function wall {ledger.classifier_cpu_wall_seconds:.6f} s, "
            f"nested feature loading {ledger.classifier_feature_loading_seconds:.6f} s "
            f"and fitting {ledger.classifier_fitting_seconds:.6f} s.",
            "",
            *(f"- {note}" for note in ledger.scope_notes),
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--followup", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    ledger = build_ledger(arguments.followup, arguments.previous)
    write_record(arguments.output, ledger)
    arguments.output.with_suffix(".md").write_text(render(ledger), encoding="utf-8")


if __name__ == "__main__":
    main()
