"""Optional second-pass acceptance never loosens the matched endpoint or consumes TEST."""

from pathlib import Path

import pytest

from scripts.inventory_results import write_record
from speech_projector.followup_evaluation import file_artifact
from speech_projector.followup_extension_selection import (
    AcceptedFullPassExtension,
    ExtensionCheckpointSelection,
    ExtensionRejectionReason,
    FullPassExtensionConfig,
    FullPassExtensionEvidence,
    FullPassExtensionPolicy,
    RejectedFullPassExtension,
    rejection_reasons,
    validated_final_checkpoint,
)
from speech_projector.followup_selection import (
    AlignmentSelectionPolicy,
    ContentChangeKind,
    ValidationContentChange,
    select_alignment_branch,
)
from speech_projector.followup_validation_review import AlignmentBranchReview, ValidationCaseReview
from speech_projector.models import FileArtifact, TranscriptMixtureObjective
from speech_projector.objective_branch import ObjectiveBranchJob
from speech_projector.objective_continuation import ObjectiveContinuationProvenance
from speech_projector.overnight_continuation import ContinuationProvenance
from tests.test_overnight_evaluation import candidate
from tests.test_overnight_report import result


def prepared_evidence(
    directory: Path,
    *,
    ordinary_ce: float = 1.49,
    macro_ce: float = 0.99,
    neu_margin: float = 0.19,
    regressions: int = 0,
    corrections: int = 0,
) -> FullPassExtensionEvidence:
    parent = candidate("chosen", 1.5, 1, 0.2, 10)
    parent = parent.model_copy(
        update={
            "configuration": parent.configuration.model_copy(
                update={"train_examples": 38193, "epochs": 3, "max_optimizer_updates": 6775}
            )
        }
    )
    parent_result = result().model_copy(
        update={"config": parent.configuration, "steps": 6775, "train_examples": 38193}
    )
    branch_decision = select_alignment_branch(
        (parent_result,), (parent,), "chosen", AlignmentSelectionPolicy()
    )
    extension = candidate("extension", ordinary_ce, macro_ce, neu_margin, 10)
    extension = extension.model_copy(
        update={
            "configuration": parent.configuration.model_copy(
                update={"name": "extension", "max_optimizer_updates": 9550}
            )
        }
    )
    extension_result = parent_result.model_copy(
        update={"config": extension.configuration, "steps": 9550}
    )
    branch_path = directory / "branch_decision.json"
    parent_path = directory / "parent_result.json"
    candidate_path = directory / "parent_candidate.json"
    extension_path = directory / "extension_result.json"
    for path, record in (
        (branch_path, branch_decision),
        (parent_path, parent_result),
        (candidate_path, parent),
        (extension_path, extension_result),
    ):
        write_record(path, record)
    parent_file, candidate_file = file_artifact(parent_path), file_artifact(candidate_path)
    configuration = FullPassExtensionConfig(
        branch_decision=branch_path,
        policy=directory / "policy.json",
        parent_run_directory=directory / "parent",
        extension_run_directory=directory / "extension",
        review=directory / "review.json",
        output_directory=directory / "output",
    )
    generation_file = FileArtifact(
        path=Path("generations"), source_path=Path("generations"), bytes=1, sha256="g"
    )
    cases: list[ValidationCaseReview] = []
    for index in range(24):
        changes: list[ValidationContentChange] = []
        if index < regressions:
            changes.append(
                ValidationContentChange(
                    kind=ContentChangeKind.REGRESSION,
                    run_name="extension",
                    example_id=f"case_{index}",
                    reason="New literal contradiction.",
                )
            )
        if regressions <= index < regressions + corrections:
            changes.append(
                ValidationContentChange(
                    kind=ContentChangeKind.CORRECTION,
                    run_name="extension",
                    example_id=f"case_{index}",
                    reason="Corrected parent's material factual reversal.",
                )
            )
        cases.append(
            ValidationCaseReview(
                example_id=f"case_{index}",
                finding="Compared actual parent and extension replies.",
                changes=tuple(changes),
            )
        )
    return FullPassExtensionEvidence(
        configuration=configuration,
        policy=FullPassExtensionPolicy(),
        branch_decision=branch_decision,
        branch_decision_file=file_artifact(branch_path),
        parent_result=parent_result,
        parent_result_file=parent_file,
        parent_candidate=parent,
        parent_candidate_file=candidate_file,
        extension_result=extension_result,
        extension_result_file=file_artifact(extension_path),
        extension_candidate=extension,
        continuation=ObjectiveContinuationProvenance(
            job=ObjectiveBranchJob(
                source_run=configuration.parent_run_directory,
                data_root=directory / "data",
                output_root=directory,
                configuration=extension.configuration,
            ),
            continuation=ContinuationProvenance(
                source_configuration=parent.configuration,
                continuation_configuration=extension.configuration,
                source_artifacts=(),
            ),
            data_artifacts=(parent_file, candidate_file),
        ),
        review=AlignmentBranchReview(
            run_name="extension",
            control_generations=generation_file,
            branch_generations=generation_file,
            cases=tuple(cases),
        ),
        input_files=(),
    )


@pytest.mark.parametrize(
    ("ordinary_ce", "macro_ce", "margin", "expected"),
    (
        (1.49, 0.99, 0.19, ()),
        (1.49, 1.0, 0.19, (ExtensionRejectionReason.MACRO_NOT_IMPROVED,)),
        (1.511, 0.99, 0.19, (ExtensionRejectionReason.ORDINARY_DEGRADED,)),
        (1.49, 0.99, 0.15, (ExtensionRejectionReason.NEU_MARGIN_DEGRADED,)),
    ),
)
def test_strict_validation_improvement_and_safeguards(
    tmp_path: Path,
    ordinary_ce: float,
    macro_ce: float,
    margin: float,
    expected: tuple[ExtensionRejectionReason, ...],
) -> None:
    evidence = prepared_evidence(
        tmp_path, ordinary_ce=ordinary_ce, macro_ce=macro_ce, neu_margin=margin
    )
    assert rejection_reasons(evidence) == expected
    if expected:
        with pytest.raises(ValueError, match="violates"):
            AcceptedFullPassExtension(evidence=evidence)
    else:
        assert AcceptedFullPassExtension(evidence=evidence).kind == "accepted"


@pytest.mark.parametrize(
    ("regressions", "corrections", "blocked"), ((2, 0, False), (3, 0, True), (3, 3, False))
)
def test_material_veto_requires_three_regressions_exceeding_corrections(
    tmp_path: Path, regressions: int, corrections: int, blocked: bool
) -> None:
    evidence = prepared_evidence(tmp_path, regressions=regressions, corrections=corrections)
    assert (
        ExtensionRejectionReason.MATERIAL_CONTENT_VETO in rejection_reasons(evidence)
    ) == blocked


def test_final_review_accepts_only_the_hash_bound_accepted_extension(tmp_path: Path) -> None:
    evidence = prepared_evidence(tmp_path)
    path = tmp_path / "extension_decision.json"
    write_record(path, AcceptedFullPassExtension(evidence=evidence))
    selection = ExtensionCheckpointSelection(
        branch_decision=evidence.configuration.branch_decision, extension_decision=path
    )
    result_path = tmp_path / "extension_result.json"
    assert validated_final_checkpoint(selection, result_path) == evidence.extension_result
    write_record(path, AcceptedFullPassExtension(evidence=evidence))
    changed = evidence.extension_result.model_copy(update={"runtime_seconds": 9999})
    write_record(result_path, changed)
    with pytest.raises(ValueError, match="accepted validation evidence"):
        validated_final_checkpoint(selection, result_path)


def test_rejected_extension_cannot_become_final_checkpoint(tmp_path: Path) -> None:
    evidence = prepared_evidence(tmp_path, macro_ce=1)
    path = tmp_path / "extension_decision.json"
    write_record(
        path, RejectedFullPassExtension(evidence=evidence, reasons=rejection_reasons(evidence))
    )
    selection = ExtensionCheckpointSelection(
        branch_decision=evidence.configuration.branch_decision, extension_decision=path
    )
    with pytest.raises(ValueError, match="rejected extension"):
        validated_final_checkpoint(selection, tmp_path / "extension_result.json")


def test_extension_cannot_silently_switch_training_objective(tmp_path: Path) -> None:
    evidence = prepared_evidence(tmp_path)
    changed = evidence.extension_result.config.model_copy(
        update={"objective": TranscriptMixtureObjective()}
    )
    with pytest.raises(ValueError, match="selected matched branch|preserve objective"):
        FullPassExtensionEvidence.model_validate(
            evidence.model_copy(
                update={
                    "extension_result": evidence.extension_result.model_copy(
                        update={"config": changed}
                    )
                }
            ).model_dump()
        )
