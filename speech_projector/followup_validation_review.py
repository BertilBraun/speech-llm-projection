"""Bind complete fixed validation content reviews to the frozen branch-selection rule."""

from pathlib import Path

from pydantic import Field

from scripts.inventory_results import write_record
from speech_projector.followup_evaluation import file_artifact
from speech_projector.followup_selection import (
    AlignmentSelectionDecision,
    AlignmentSelectionPolicy,
    ValidationContentChange,
    select_alignment_branch,
)
from speech_projector.models import (
    EvaluationCondition,
    FileArtifact,
    Record,
    RunResult,
    SampleGeneration,
)
from speech_projector.overnight_evaluation import SweepCandidate
from speech_projector.overnight_preparation import load_records


class ValidationCaseReview(Record):
    example_id: str
    finding: str = Field(min_length=1)
    changes: tuple[ValidationContentChange, ...] = ()


class AlignmentBranchReview(Record):
    run_name: str
    control_generations: FileArtifact
    branch_generations: FileArtifact
    cases: tuple[ValidationCaseReview, ...]


class ReviewedAlignmentBranch(Record):
    run_directory: Path
    review: Path


class ValidationSelectionConfig(Record):
    policy: Path
    control_run_directory: Path
    branches: tuple[ReviewedAlignmentBranch, ...] = Field(min_length=1)
    output_directory: Path


class ValidationSelectionReceipt(Record):
    configuration: ValidationSelectionConfig
    inputs: tuple[FileArtifact, ...]
    validation_example_ids: tuple[str, ...]


def primary_generations(path: Path) -> tuple[SampleGeneration, ...]:
    rows = tuple(
        row
        for row in load_records(path, SampleGeneration)
        if row.condition == EvaluationCondition.SPEECH
    )
    if len(rows) != 24 or len({row.example_id for row in rows}) != 24:
        raise ValueError(
            "Branch content review requires exactly 24 unique primary validation replies"
        )
    return rows


def verify_generation_artifact(path: Path, artifact: FileArtifact) -> None:
    actual = file_artifact(path)
    if (actual.sha256, actual.bytes) != (artifact.sha256, artifact.bytes):
        raise ValueError("Validation content review is not bound to the actual generation file")


def validated_content_changes(
    review: AlignmentBranchReview,
    control_path: Path,
    branch_path: Path,
    branch_name: str,
) -> tuple[ValidationContentChange, ...]:
    verify_generation_artifact(control_path, review.control_generations)
    verify_generation_artifact(branch_path, review.branch_generations)
    if review.run_name != branch_name:
        raise ValueError("Content review names a different completed branch")
    control = primary_generations(control_path)
    branch = primary_generations(branch_path)
    identifiers = tuple(row.example_id for row in control)
    if tuple(row.example_id for row in branch) != identifiers:
        raise ValueError("Branch and control use different ordered validation panels")
    if tuple(row.example_id for row in review.cases) != identifiers:
        raise ValueError("Content review must explicitly cover all 24 ordered validation cases")
    for first, second in zip(control, branch, strict=True):
        if (first.user_transcript, first.history, first.gold_response) != (
            second.user_transcript,
            second.history,
            second.gold_response,
        ):
            raise ValueError(
                "Branch validation words, history or target differ from the CE control"
            )
    changes: list[ValidationContentChange] = []
    for case in review.cases:
        if len({row.kind for row in case.changes}) != len(case.changes):
            raise ValueError("A validation case repeats the same change kind")
        for change in case.changes:
            if change.example_id != case.example_id or change.run_name != branch_name:
                raise ValueError(
                    "Content change identity differs from its reviewed validation case"
                )
        changes.extend(case.changes)
    return tuple(changes)


def select_reviewed_branches(
    configuration: ValidationSelectionConfig,
) -> AlignmentSelectionDecision:
    policy = AlignmentSelectionPolicy.model_validate_json(configuration.policy.read_bytes())
    control_directory = configuration.control_run_directory
    control_result = RunResult.model_validate_json((control_directory / "result.json").read_bytes())
    control_candidate = SweepCandidate.model_validate_json(
        (control_directory / "candidate.json").read_bytes()
    )
    control_generations = control_directory / "validation/evaluation_generations.jsonl"
    primary = primary_generations(control_generations)
    results = [control_result]
    candidates = [control_candidate]
    changes: list[ValidationContentChange] = []
    paths = [
        configuration.policy,
        control_directory / "result.json",
        control_directory / "candidate.json",
        control_generations,
    ]
    for branch in configuration.branches:
        result_path = branch.run_directory / "result.json"
        candidate_path = branch.run_directory / "candidate.json"
        generation_path = branch.run_directory / "validation/evaluation_generations.jsonl"
        result = RunResult.model_validate_json(result_path.read_bytes())
        candidate = SweepCandidate.model_validate_json(candidate_path.read_bytes())
        review = AlignmentBranchReview.model_validate_json(branch.review.read_bytes())
        changes.extend(
            validated_content_changes(
                review, control_generations, generation_path, result.config.name
            )
        )
        results.append(result)
        candidates.append(candidate)
        paths.extend((result_path, candidate_path, generation_path, branch.review))
    decision = select_alignment_branch(
        results, candidates, control_result.config.name, policy, changes
    )
    receipt = ValidationSelectionReceipt(
        configuration=configuration,
        inputs=tuple(file_artifact(path) for path in paths),
        validation_example_ids=tuple(row.example_id for row in primary),
    )
    write_record(configuration.output_directory / "decision.json", decision)
    write_record(configuration.output_directory / "provenance.json", receipt)
    lines = [
        "# Frozen validation-only branch selection",
        "",
        f"Selected: {decision.selected_run}; CE control: {decision.control_run}.",
        "",
        "All 24 primary validation cases were explicitly reviewed for each auxiliary branch. "
        "The policy and complete reviews are bound to their exact input bytes. "
        "TEST results do not enter this decision. Thresholds are pragmatic safeguards, "
        "not calibrated significance or meaningful-effect bounds.",
        "",
        "| Branch | Ordinary CE | Macro CE | Robust Neu margin | Eligible | In macro band |",
        "|---|---:|---:|---:|---|---|",
    ]
    for candidate in decision.validation_candidates:
        name = candidate.configuration.name
        lines.append(
            f"| {name} | {candidate.validation.old_ordinary.cross_entropy:.6f} | "
            f"{candidate.validation.macro_cross_entropy:.6f} | "
            f"{candidate.robust_neu_margin:.6f} | {name in decision.eligible_runs} | "
            f"{name in decision.macro_band_runs} |"
        )
    lines += ["", "## Material corrections and regressions", ""]
    if not changes:
        lines.append("No new material failure or clearly corrected material failure was recorded.")
    for change in changes:
        lines.append(
            f"- {change.run_name} / {change.example_id} / {change.kind.value}: {change.reason}"
        )
    (configuration.output_directory / "decision.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return decision
