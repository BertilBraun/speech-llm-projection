"""Validation export retains the frozen eligibility/selection and measured robust margin."""

import csv
from pathlib import Path

from scripts.plot_followup_validation import write_validation_plot
from speech_projector.followup_selection import AlignmentSelectionPolicy, select_alignment_branch
from speech_projector.models import OrdinaryResponseKLObjective, TranscriptMixtureObjective
from tests.test_overnight_evaluation import candidate
from tests.test_overnight_report import result


def test_validation_plot_preserves_tone_guard_and_does_not_reselect_from_test(
    tmp_path: Path,
) -> None:
    control = candidate("full_control_id", 1.6, 1.1, 0.2, 10)
    transcript = candidate("full_transcript_id", 1.4, 1.12, 0.2, 10)
    transcript = transcript.model_copy(
        update={
            "configuration": transcript.configuration.model_copy(
                update={"objective": TranscriptMixtureObjective()}
            )
        }
    )
    teacher_kl = candidate("full_kl_id", 1.2, 0.9, 0.1, 10)
    teacher_kl = teacher_kl.model_copy(
        update={
            "configuration": teacher_kl.configuration.model_copy(
                update={"objective": OrdinaryResponseKLObjective()}
            )
        }
    )
    candidates = (control, transcript, teacher_kl)
    completed = tuple(
        result().model_copy(update={"config": row.configuration, "steps": 6775})
        for row in candidates
    )
    decision = select_alignment_branch(
        completed, candidates, control.configuration.name, AlignmentSelectionPolicy()
    )
    assert decision.selected_run == transcript.configuration.name
    write_validation_plot(decision, tmp_path)
    with (tmp_path / "validation_branches.csv").open(encoding="utf-8", newline="") as stream:
        rows = tuple(csv.reader(stream))
    assert rows[2][0] == "full_transcript_id"
    assert rows[2][-3:] == ["True", "True", "True"]
    assert rows[3][-3:] == ["False", "False", "False"]
    assert float(rows[3][12]) == teacher_kl.robust_neu_margin
    assert (tmp_path / "validation_branches.png").is_file()
