"""Accept an optional second full pass using its predeclared validation-only safeguards."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, TypeAdapter, model_validator

from scripts.inventory_results import write_record
from speech_projector.followup_evaluation import file_artifact
from speech_projector.followup_selection import AlignmentSelectionDecision, ContentChangeKind
from speech_projector.followup_validation_review import (
    AlignmentBranchReview,
    ValidationSelectionReceipt,
    validated_content_changes,
)
from speech_projector.models import FileArtifact, Record, RunResult
from speech_projector.objective_continuation import ObjectiveContinuationProvenance
from speech_projector.overnight_evaluation import SweepCandidate


class FullPassExtensionPolicy(Record):
    endpoint_updates: int = Field(default=9550, gt=0)
    ordinary_ce_tolerance: float = Field(default=0.01, ge=0)
    neu_margin_tolerance: float = Field(default=0.02, ge=0)
    minimum_material_regressions: int = Field(default=3, ge=1)


class FullPassExtensionConfig(Record):
    branch_decision: Path
    policy: Path
    parent_run_directory: Path
    extension_run_directory: Path
    review: Path
    output_directory: Path


class FullPassExtensionEvidence(Record):
    configuration: FullPassExtensionConfig
    policy: FullPassExtensionPolicy
    branch_decision: AlignmentSelectionDecision
    branch_decision_file: FileArtifact
    parent_result: RunResult
    parent_result_file: FileArtifact
    parent_candidate: SweepCandidate
    parent_candidate_file: FileArtifact
    extension_result: RunResult
    extension_result_file: FileArtifact
    extension_candidate: SweepCandidate
    continuation: ObjectiveContinuationProvenance
    review: AlignmentBranchReview
    input_files: tuple[FileArtifact, ...]

    @model_validator(mode="after")
    def validate_selected_trajectory(self) -> FullPassExtensionEvidence:
        parent, extension = self.parent_result, self.extension_result
        decision = self.branch_decision
        if (
            parent.config.name != decision.selected_run
            or parent.steps != decision.policy.endpoint_updates
            or parent.config != self.parent_candidate.configuration
            or extension.config != self.extension_candidate.configuration
        ):
            raise ValueError("Extension must follow the actual selected matched branch")
        selected = tuple(
            row
            for row in decision.validation_candidates
            if row.configuration.name == decision.selected_run
        )
        if len(selected) != 1 or selected[0] != self.parent_candidate:
            raise ValueError(
                "Extension parent candidate differs from the immutable branch decision"
            )
        expected = parent.config.model_copy(
            update={
                "name": extension.config.name,
                "max_optimizer_updates": self.policy.endpoint_updates,
            }
        )
        effective_batch = parent.config.microbatch_size * parent.config.gradient_accumulation
        updates_per_pass = (parent.train_examples + effective_batch - 1) // effective_batch
        if (
            extension.config != expected
            or extension.config.name == parent.config.name
            or extension.steps != self.policy.endpoint_updates
            or extension.steps != 2 * updates_per_pass
            or parent.config.epochs < 2
            or parent.train_examples != extension.train_examples
        ):
            raise ValueError(
                "Extension must preserve objective/configuration and finish the second full pass"
            )
        continuation = self.continuation
        if (
            continuation.continuation.source_configuration != parent.config
            or continuation.continuation.continuation_configuration != extension.config
            or continuation.job.configuration != extension.config
        ):
            raise ValueError("Saved continuation provenance differs from the selected trajectory")
        for required in (self.parent_result_file, self.parent_candidate_file):
            if not artifact_is_bound(required, continuation.data_artifacts):
                raise ValueError(
                    "Continuation is not hash-bound to the actual parent result/candidate"
                )
        if self.review.run_name != extension.config.name:
            raise ValueError("Content review names a different extension")
        identifiers = tuple(row.example_id for row in self.review.cases)
        if len(identifiers) != 24 or len(set(identifiers)) != 24:
            raise ValueError("Extension requires an explicit full 24-case validation review")
        for case in self.review.cases:
            for change in case.changes:
                if change.example_id != case.example_id or change.run_name != extension.config.name:
                    raise ValueError("Extension content change differs from its reviewed case")
        return self


class ExtensionRejectionReason(str, Enum):
    MACRO_NOT_IMPROVED = "macro_ce_not_strictly_improved"
    ORDINARY_DEGRADED = "ordinary_ce_above_parent_tolerance"
    NEU_MARGIN_DEGRADED = "robust_neu_margin_below_parent_tolerance"
    MATERIAL_CONTENT_VETO = "systematic_new_material_content_failures"


def rejection_reasons(evidence: FullPassExtensionEvidence) -> tuple[ExtensionRejectionReason, ...]:
    parent, extension = evidence.parent_candidate, evidence.extension_candidate
    reasons: list[ExtensionRejectionReason] = []
    if extension.validation.macro_cross_entropy >= parent.validation.macro_cross_entropy:
        reasons.append(ExtensionRejectionReason.MACRO_NOT_IMPROVED)
    if (
        extension.validation.old_ordinary.cross_entropy
        > parent.validation.old_ordinary.cross_entropy + evidence.policy.ordinary_ce_tolerance
    ):
        reasons.append(ExtensionRejectionReason.ORDINARY_DEGRADED)
    if (
        extension.robust_neu_margin
        < parent.robust_neu_margin - evidence.policy.neu_margin_tolerance
    ):
        reasons.append(ExtensionRejectionReason.NEU_MARGIN_DEGRADED)
    regressions = sum(
        any(change.kind == ContentChangeKind.REGRESSION for change in case.changes)
        for case in evidence.review.cases
    )
    corrections = sum(
        any(change.kind == ContentChangeKind.CORRECTION for change in case.changes)
        for case in evidence.review.cases
    )
    if regressions >= evidence.policy.minimum_material_regressions and regressions > corrections:
        reasons.append(ExtensionRejectionReason.MATERIAL_CONTENT_VETO)
    return tuple(reasons)


class AcceptedFullPassExtension(Record):
    kind: Literal["accepted"] = "accepted"
    evidence: FullPassExtensionEvidence

    @model_validator(mode="after")
    def validate_acceptance(self) -> AcceptedFullPassExtension:
        if rejection_reasons(self.evidence):
            raise ValueError("An accepted extension violates its recorded validation policy")
        return self


class RejectedFullPassExtension(Record):
    kind: Literal["rejected"] = "rejected"
    evidence: FullPassExtensionEvidence
    reasons: tuple[ExtensionRejectionReason, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_rejection(self) -> RejectedFullPassExtension:
        if self.reasons != rejection_reasons(self.evidence):
            raise ValueError("Extension rejection reasons differ from its validation evidence")
        return self


FullPassExtensionDecision: TypeAlias = Annotated[
    AcceptedFullPassExtension | RejectedFullPassExtension, Field(discriminator="kind")
]
EXTENSION_DECISION = TypeAdapter(FullPassExtensionDecision)


class BranchCheckpointSelection(Record):
    kind: Literal["branch"] = "branch"
    branch_decision: Path


class ExtensionCheckpointSelection(Record):
    kind: Literal["extension"] = "extension"
    branch_decision: Path
    extension_decision: Path


FinalCheckpointSelection: TypeAlias = Annotated[
    BranchCheckpointSelection | ExtensionCheckpointSelection, Field(discriminator="kind")
]


def artifact_is_bound(required: FileArtifact, artifacts: tuple[FileArtifact, ...]) -> bool:
    return any((required.sha256, required.bytes) == (row.sha256, row.bytes) for row in artifacts)


def validated_final_checkpoint(selection: FinalCheckpointSelection, run_result: Path) -> RunResult:
    branch = AlignmentSelectionDecision.model_validate_json(selection.branch_decision.read_bytes())
    actual = file_artifact(run_result)
    result = RunResult.model_validate_json(run_result.read_bytes())
    match selection:
        case BranchCheckpointSelection():
            receipt = ValidationSelectionReceipt.model_validate_json(
                selection.branch_decision.with_name("provenance.json").read_bytes()
            )
            if not artifact_is_bound(actual, receipt.inputs):
                raise ValueError("Final branch result is not bound to its validation selection")
            selected = tuple(
                row
                for row in branch.validation_candidates
                if row.configuration.name == branch.selected_run
            )
            if (
                len(selected) != 1
                or result.config != selected[0].configuration
                or result.steps != branch.policy.endpoint_updates
            ):
                raise ValueError("Final checkpoint differs from the selected matched branch")
        case ExtensionCheckpointSelection():
            decision = EXTENSION_DECISION.validate_json(selection.extension_decision.read_bytes())
            match decision:
                case RejectedFullPassExtension():
                    raise ValueError(
                        "A rejected extension cannot become the final reviewed checkpoint"
                    )
                case AcceptedFullPassExtension():
                    evidence = decision.evidence
                    if (
                        branch != evidence.branch_decision
                        or not artifact_is_bound(
                            file_artifact(selection.branch_decision),
                            (evidence.branch_decision_file,),
                        )
                        or result != evidence.extension_result
                        or not artifact_is_bound(actual, (evidence.extension_result_file,))
                    ):
                        raise ValueError(
                            "Final extension differs from its accepted validation evidence"
                        )
    return result


def select_full_pass_extension(configuration: FullPassExtensionConfig) -> FullPassExtensionDecision:
    parent = configuration.parent_run_directory
    extension = configuration.extension_run_directory
    parent_result_path = parent / "result.json"
    parent_candidate_path = parent / "candidate.json"
    extension_result_path = extension / "result.json"
    extension_candidate_path = extension / "candidate.json"
    branch = AlignmentSelectionDecision.model_validate_json(
        configuration.branch_decision.read_bytes()
    )
    policy = FullPassExtensionPolicy.model_validate_json(configuration.policy.read_bytes())
    parent_result = validated_final_checkpoint(
        BranchCheckpointSelection(branch_decision=configuration.branch_decision),
        parent_result_path,
    )
    extension_result = RunResult.model_validate_json(extension_result_path.read_bytes())
    review = AlignmentBranchReview.model_validate_json(configuration.review.read_bytes())
    validated_content_changes(
        review,
        parent / "validation/evaluation_generations.jsonl",
        extension / "validation/evaluation_generations.jsonl",
        extension_result.config.name,
    )
    evidence = FullPassExtensionEvidence(
        configuration=configuration,
        policy=policy,
        branch_decision=branch,
        branch_decision_file=file_artifact(configuration.branch_decision),
        parent_result=parent_result,
        parent_result_file=file_artifact(parent_result_path),
        parent_candidate=SweepCandidate.model_validate_json(parent_candidate_path.read_bytes()),
        parent_candidate_file=file_artifact(parent_candidate_path),
        extension_result=extension_result,
        extension_result_file=file_artifact(extension_result_path),
        extension_candidate=SweepCandidate.model_validate_json(
            extension_candidate_path.read_bytes()
        ),
        continuation=ObjectiveContinuationProvenance.model_validate_json(
            (extension / "continuation_data.json").read_bytes()
        ),
        review=review,
        input_files=tuple(
            file_artifact(path)
            for path in (
                configuration.policy,
                extension_candidate_path,
                configuration.review,
                extension / "continuation_data.json",
            )
        ),
    )
    reasons = rejection_reasons(evidence)
    decision: FullPassExtensionDecision = (
        RejectedFullPassExtension(evidence=evidence, reasons=reasons)
        if reasons
        else AcceptedFullPassExtension(evidence=evidence)
    )
    write_record(configuration.output_directory / "decision.json", decision)
    lines = [
        "# Optional second-pass validation decision",
        "",
        f"Outcome: {decision.kind}; parent: {parent_result.config.name}; "
        f"extension: {extension_result.config.name}.",
        "",
        "The matched-branch decision remains immutable. This extension uses only validation: "
        f"strict macro CE improvement, ordinary CE ≤ parent +{policy.ordinary_ce_tolerance:g}, "
        f"robust Neu margin ≥ parent −{policy.neu_margin_tolerance:g}, and a full 24-case "
        "material-content review. Thresholds are pragmatic "
        "safeguards, not significance bounds. TEST is not used to accept the extension.",
        "",
        "| Endpoint | Updates | Ordinary CE | Macro CE | Robust Neu margin |",
        "|---|---:|---:|---:|---:|",
    ]
    for result, candidate in (
        (parent_result, evidence.parent_candidate),
        (extension_result, evidence.extension_candidate),
    ):
        lines.append(
            f"| {result.config.name} | {result.steps} | "
            f"{candidate.validation.old_ordinary.cross_entropy:.6f} | "
            f"{candidate.validation.macro_cross_entropy:.6f} | {candidate.robust_neu_margin:.6f} |"
        )
    lines += ["", "Rejection reasons: " + (", ".join(row.value for row in reasons) or "none"), ""]
    for case in review.cases:
        lines.append(f"- {case.example_id}: {case.finding}")
        lines.extend(f"  - {change.kind.value}: {change.reason}" for change in case.changes)
    (configuration.output_directory / "decision.md").write_text("\n".join(lines), encoding="utf-8")
    return decision
