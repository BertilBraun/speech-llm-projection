"""Hash-bound final resource scopes from completed follow-up artifacts."""

import argparse
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from artifacts.followup10hz.resource_ledger import ResourceLedger, evidence
from scripts.inventory_results import stable_digest, write_record
from speech_projector.followup_conversation import PipelineConversationSummary
from speech_projector.followup_tone_baselines import OracleReuseReceipt
from speech_projector.models import EvaluationMetrics, FileArtifact, Record
from speech_projector.overnight_conversation_execution import ConversationEvaluationSummary


class ProcessScope(str, Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"
    SMOKE = "smoke"
    CPU = "cpu"


class ProcessInterval(Record):
    name: str
    scope: ProcessScope
    started_at: datetime
    exited_at: datetime
    wall_seconds: float
    exit_status: int


class EvaluatorEvidence(Record):
    source: FileArtifact
    metrics: EvaluationMetrics


class ConversationGenerationEvidence(Record):
    source: FileArtifact
    generation_seconds: float


class ResourceClosure(Record):
    created_at: datetime
    training_ledger: FileArtifact
    distinct_training_loop_seconds: float
    lifecycle_log: FileArtifact
    processes: tuple[ProcessInterval, ...]
    evaluators: tuple[EvaluatorEvidence, ...]
    reused_oracle_receipt: FileArtifact
    conversation_generations: tuple[ConversationGenerationEvidence, ...]
    scope_notes: tuple[str, ...]


def process_scope(name: str) -> ProcessScope:
    match name:
        case "followup10hz-code-check" | "followup10hz-tone-classifier":
            return ProcessScope.CPU
        case "followup10hz-ordinarykl-smoke":
            return ProcessScope.SMOKE
        case (
            "followup10hz-fullpass"
            | "followup10hz-ce-control"
            | "followup10hz-transcript30"
            | "followup10hz-ordinarykl"
            | "followup10hz-selected9550-training"
        ):
            return ProcessScope.TRAINING
        case (
            "followup10hz-tone-baseline"
            | "followup10hz-test-epoch1"
            | "followup10hz-conversation-epoch1"
            | "followup10hz-conversation-tone"
            | "followup10hz-test-ordinarykl"
            | "followup10hz-test-ce_control"
            | "followup10hz-test-transcript30"
            | "followup10hz-selected9550-test"
            | "followup10hz-selected9550-conversation"
            | "followup10hz-conversation-ce_control"
        ):
            return ProcessScope.EVALUATION
        case _:
            raise ValueError(f"Unknown scoped job classification: {name}")


def process_intervals(path: Path) -> tuple[ProcessInterval, ...]:
    starts: dict[str, datetime] = {}
    intervals: list[ProcessInterval] = []
    pattern = re.compile(
        r"^(?P<time>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) INFO "
        r"(?P<event>spawned|exited): '?(?P<name>followup10hz-[\w-]+)'?"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        matched = pattern.match(line)
        if matched is None:
            continue
        name = matched.group("name")
        timestamp = datetime.strptime(matched.group("time"), "%Y-%m-%d %H:%M:%S,%f").replace(
            tzinfo=timezone.utc
        )
        if matched.group("event") == "spawned":
            if name in starts:
                raise ValueError(f"Scoped job restarted before its earlier exit: {name}")
            starts[name] = timestamp
            continue
        status = re.search(r"\(exit status (\d+);", line)
        if name not in starts or status is None:
            raise ValueError(f"Scoped exit lacks a start/status: {name}")
        started = starts.pop(name)
        if timestamp < started:
            raise ValueError(f"Scoped process exits before its recorded start: {name}")
        intervals.append(
            ProcessInterval(
                name=name,
                scope=process_scope(name),
                started_at=started,
                exited_at=timestamp,
                wall_seconds=(timestamp - started).total_seconds(),
                exit_status=int(status.group(1)),
            )
        )
    if starts:
        raise ValueError("Final lifecycle evidence still contains an unfinished scoped job")
    gpu = sorted(
        (row for row in intervals if row.scope != ProcessScope.CPU), key=lambda row: row.started_at
    )
    if any(
        first.exited_at > second.started_at for first, second in zip(gpu, gpu[1:], strict=False)
    ):
        raise ValueError("GPU-using process intervals overlap; do not sum without review")
    return tuple(intervals)


def build(root: Path) -> ResourceClosure:
    ledger_path = root / "provenance/resource_ledger/resource_ledger.json"
    ledger = ResourceLedger.model_validate_json(ledger_path.read_bytes())
    lifecycle = root / "provenance/operational_final/payload/supervisor/lifecycle_events.log"
    evaluations = tuple(sorted(root.glob("evaluation_*/speech/evaluation.json")))
    validations = tuple(sorted(root.glob("runs/*/validation/evaluation.json")))
    baseline = root / "baseline/asr_predicted_tone/evaluation.json"
    oracle_receipt = root / "baseline/asr_oracle_tone/reuse.json"
    reuse = OracleReuseReceipt.model_validate_json(oracle_receipt.read_bytes())
    remote_predicted = Path(
        "/workspace/speech-projector/results_followup_20261007/baseline/asr_predicted_tone"
    )
    for source in reuse.reused_files:
        local = (
            root / "baseline/asr_predicted_tone" / source.source_path.relative_to(remote_predicted)
        )
        if stable_digest(local) != (source.bytes, source.sha256):
            raise ValueError("Oracle reuse receipt differs from its preserved predicted source")
    if EvaluationMetrics.model_validate_json(
        baseline.read_bytes()
    ) != EvaluationMetrics.model_validate_json(
        (root / "baseline/asr_oracle_tone/evaluation.json").read_bytes()
    ):
        raise ValueError("Oracle copied evaluator metrics differ from the predicted evaluation")
    evaluators = tuple(
        EvaluatorEvidence(
            source=evidence(path), metrics=EvaluationMetrics.model_validate_json(path.read_bytes())
        )
        for path in (*evaluations, *validations, baseline)
    )
    speech_paths = tuple(
        root / name / "speech/summary.json"
        for name in ("conversation_epoch1", "conversation_ce_control")
    )
    pipeline_paths = tuple(
        root / name / "summary.json"
        for name in ("conversation_epoch1/text", "conversation_epoch1/asr", "conversation_asr_tone")
    )
    conversations = tuple(
        ConversationGenerationEvidence(
            source=evidence(path),
            generation_seconds=ConversationEvaluationSummary.model_validate_json(
                path.read_bytes()
            ).generation_seconds,
        )
        for path in speech_paths
    ) + tuple(
        ConversationGenerationEvidence(
            source=evidence(path),
            generation_seconds=PipelineConversationSummary.model_validate_json(
                path.read_bytes()
            ).generation_seconds,
        )
        for path in pipeline_paths
    )
    return ResourceClosure(
        created_at=datetime.now(timezone.utc),
        training_ledger=evidence(ledger_path),
        distinct_training_loop_seconds=sum(
            row.added_training_seconds for row in ledger.completed_training
        ),
        lifecycle_log=evidence(lifecycle),
        processes=process_intervals(lifecycle),
        evaluators=evaluators,
        reused_oracle_receipt=evidence(oracle_receipt),
        conversation_generations=conversations,
        scope_notes=(
            "Whole-process wall includes startup, training/evaluation and shutdown; "
            "it is not CUDA compute or utilization.",
            "Training-loop increments and component evaluation timers overlap whole-process wall "
            "and are not added to it.",
            "Evaluator generation timers are nested; periodic training validation lies within "
            "recorded training-loop time.",
            "Oracle-tone evaluation was reused from predicted-tone output; "
            "its copied timer is excluded.",
            "Conversation summaries and their per-reply timers are nested; matched subsets/reused "
            "comparators are not counted again.",
            "Three neutral follow-up ASRs occurred once within the original conversation process; "
            "no new training feature extraction.",
            "Saved PyTorch allocated peak is a lineage maximum; reserved memory and point NVIDIA "
            "usage are separate.",
            "Provider billing duration/rate and GPU utilization-hours were not measured; "
            "no rental cost is inferred.",
        ),
    )


def render(record: ResourceClosure) -> str:
    gpu = tuple(row for row in record.processes if row.scope != ProcessScope.CPU)
    lines = [
        "# Final follow-up resource scopes",
        "",
        f"Distinct new training-loop time: {record.distinct_training_loop_seconds:.6f}s.",
        f"Recorded nonoverlapping GPU-using process wall: "
        f"{sum(row.wall_seconds for row in gpu):.6f}s ({len(gpu)} process executions).",
        "",
        "| Process | Scope | Wall seconds | Exit status |",
        "|---|---|---:|---:|",
    ]
    lines.extend(
        f"| {row.name} | {row.scope.value} | {row.wall_seconds:.3f} | {row.exit_status} |"
        for row in record.processes
    )
    lines.extend(
        (
            "",
            "| Evaluator artifact | Total evaluation seconds | Nested generation seconds |",
            "|---|---:|---:|",
        )
    )
    lines.extend(
        f"| {row.source.path} | {row.metrics.evaluation_seconds:.6f} | "
        f"{row.metrics.generation_seconds:.6f} |"
        for row in record.evaluators
    )
    lines.extend(
        (
            "",
            f"Standalone evaluator-call total: "
            f"{sum(row.metrics.evaluation_seconds for row in record.evaluators):.6f}s, "
            "a partial component scope within process wall.",
            "",
            "| Conversation summary | Nested generation seconds |",
            "|---|---:|",
        )
    )
    lines.extend(
        f"| {row.source.path} | {row.generation_seconds:.6f} |"
        for row in record.conversation_generations
    )
    lines.extend(("", *(f"- {note}" for note in record.scope_notes), ""))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    record = build(arguments.root)
    write_record(arguments.output, record)
    arguments.output.with_suffix(".md").write_text(render(record), encoding="utf-8")


if __name__ == "__main__":
    main()
