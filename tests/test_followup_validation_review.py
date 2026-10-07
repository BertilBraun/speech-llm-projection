"""Unreviewed or differently generated validation cases cannot enter the qualitative veto."""

from pathlib import Path

import pytest

from speech_projector.followup_evaluation import file_artifact
from speech_projector.followup_selection import ContentChangeKind, ValidationContentChange
from speech_projector.followup_validation_review import (
    AlignmentBranchReview,
    ValidationCaseReview,
    validated_content_changes,
)
from speech_projector.models import EvaluationCondition, SampleGeneration


def prepared_review(directory: Path) -> tuple[AlignmentBranchReview, Path, Path]:
    control = directory / "control.jsonl"
    branch = directory / "branch.jsonl"
    samples = tuple(
        SampleGeneration(
            example_id=f"case_{index:02d}",
            dialogue_id=f"dialogue_{index:02d}",
            condition=EvaluationCondition.SPEECH,
            history=(),
            user_transcript="The appointment moved to five, not three.",
            gold_response="Understood, five it is.",
            generated_response="Understood, five it is.",
            duration=2,
        )
        for index in range(24)
    )
    content = "".join(row.model_dump_json() + "\n" for row in samples)
    control.write_text(content, encoding="utf-8")
    branch.write_text(content, encoding="utf-8")
    review = AlignmentBranchReview(
        run_name="auxiliary",
        control_generations=file_artifact(control),
        branch_generations=file_artifact(branch),
        cases=tuple(
            ValidationCaseReview(example_id=row.example_id, finding="Same faithful timing reply.")
            for row in samples
        ),
    )
    return review, control, branch


def test_complete_review_reuses_exact_canonical_changes(tmp_path: Path) -> None:
    review, control, branch = prepared_review(tmp_path)
    change = ValidationContentChange(
        kind=ContentChangeKind.CORRECTION,
        run_name="auxiliary",
        example_id=review.cases[0].example_id,
        reason="The branch now preserves five rather than claiming three.",
    )
    cases = (review.cases[0].model_copy(update={"changes": (change,)}),) + review.cases[1:]
    assert validated_content_changes(
        review.model_copy(update={"cases": cases}), control, branch, "auxiliary"
    ) == (change,)


def test_missing_case_cannot_silently_mean_no_material_regression(tmp_path: Path) -> None:
    review, control, branch = prepared_review(tmp_path)
    with pytest.raises(ValueError, match="all 24"):
        validated_content_changes(
            review.model_copy(update={"cases": review.cases[:-1]}), control, branch, "auxiliary"
        )


def test_same_ids_but_changed_reply_requires_new_review_provenance(tmp_path: Path) -> None:
    review, control, branch = prepared_review(tmp_path)
    branch.write_text(branch.read_text().replace("Understood, five it is.", "See you at three."))
    with pytest.raises(ValueError, match="actual generation file"):
        validated_content_changes(review, control, branch, "auxiliary")


def test_hash_bound_file_with_different_current_words_is_not_a_matched_comparison(
    tmp_path: Path,
) -> None:
    review, control, branch = prepared_review(tmp_path)
    branch.write_text(branch.read_text().replace("moved to five", "moved to six"))
    rebound = review.model_copy(update={"branch_generations": file_artifact(branch)})
    with pytest.raises(ValueError, match="words, history or target"):
        validated_content_changes(rebound, control, branch, "auxiliary")


@pytest.mark.parametrize("wrong_run", (False, True))
def test_change_must_belong_to_its_exact_reviewed_case(tmp_path: Path, wrong_run: bool) -> None:
    review, control, branch = prepared_review(tmp_path)
    change = ValidationContentChange(
        kind=ContentChangeKind.REGRESSION,
        run_name="different_run" if wrong_run else "auxiliary",
        example_id=review.cases[0].example_id if wrong_run else "not_in_validation",
        reason="Material timing reversal.",
    )
    cases = (review.cases[0].model_copy(update={"changes": (change,)}),) + review.cases[1:]
    with pytest.raises(ValueError, match="identity differs"):
        validated_content_changes(
            review.model_copy(update={"cases": cases}), control, branch, "auxiliary"
        )
