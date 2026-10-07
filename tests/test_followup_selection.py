"""Selection uses matched validation endpoints and preserves tone/content safeguards."""

import pytest

from speech_projector.followup_selection import AlignmentSelectionPolicy, select_alignment_branch
from tests.test_overnight_evaluation import candidate
from tests.test_overnight_report import result


def test_alignment_selects_content_within_macro_band_and_rejects_tone_regression() -> None:
    candidates = (
        candidate("control", 1.6, 1.1, 0.2, 10),
        candidate("content", 1.4, 1.12, 0.2, 10),
        candidate("tone_loss", 1.2, 0.9, 0.1, 10),
        candidate("ordinary_loss", 1.7, 1, 0.2, 10),
    )
    results = tuple(
        result().model_copy(update={"config": row.configuration, "steps": 6775})
        for row in candidates
    )
    decision = select_alignment_branch(results, candidates, "control", AlignmentSelectionPolicy())
    assert decision.selected_run == "content"
    assert set(decision.eligible_runs) == {"control", "content"}
    with pytest.raises(ValueError, match="matched endpoint"):
        select_alignment_branch(
            (results[0].model_copy(update={"steps": 4775}),) + results[1:],
            candidates,
            "control",
            AlignmentSelectionPolicy(),
        )
