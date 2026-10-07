"""Precommitted validation-only lexical-alignment branch safeguards."""

from collections.abc import Sequence
from enum import Enum

from pydantic import Field

from speech_projector.models import Record, RunResult
from speech_projector.overnight_evaluation import SweepCandidate


class AlignmentSelectionPolicy(Record):
    endpoint_updates: int = Field(default=6775, gt=0)
    ordinary_ce_tolerance: float = Field(default=0.05, ge=0)
    neu_margin_tolerance: float = Field(default=0.02, ge=0)
    macro_ce_band: float = Field(default=0.03, ge=0)
    minimum_material_regressions: int = Field(default=3, ge=1)


class ContentChangeKind(str, Enum):
    REGRESSION = "new_material_failure"
    CORRECTION = "corrected_material_failure"


class ValidationContentChange(Record):
    kind: ContentChangeKind
    run_name: str
    example_id: str
    reason: str = Field(min_length=1)


class AlignmentSelectionDecision(Record):
    policy: AlignmentSelectionPolicy
    control_run: str
    selected_run: str
    eligible_runs: tuple[str, ...]
    macro_band_runs: tuple[str, ...]
    content_changes: tuple[ValidationContentChange, ...]
    validation_candidates: tuple[SweepCandidate, ...]


def select_alignment_branch(
    results: Sequence[RunResult],
    candidates: Sequence[SweepCandidate],
    control_run: str,
    policy: AlignmentSelectionPolicy,
    content_changes: Sequence[ValidationContentChange] = (),
) -> AlignmentSelectionDecision:
    by_name = {row.configuration.name: row for row in candidates}
    if len(by_name) != len(candidates) or control_run not in by_name:
        raise ValueError("Branch selection requires unique candidates and the named CE control")
    if len(results) != len(candidates) or {row.config.name for row in results} != by_name.keys():
        raise ValueError("Branch completion records differ from validation candidates")
    for result in results:
        if result.steps != policy.endpoint_updates:
            raise ValueError("Select only matched endpoint branches, not the earlier parent")
        if result.config != by_name[result.config.name].configuration:
            raise ValueError("Completed branch config differs from its validation candidate")
    if any(row.run_name not in by_name for row in content_changes):
        raise ValueError("Content regression references a branch outside the comparison")
    if any(row.run_name == control_run for row in content_changes):
        raise ValueError("Blocking regressions must describe deterioration versus the CE control")
    blocked: set[str] = set()
    for name in by_name:
        regressions = {
            row.example_id
            for row in content_changes
            if row.run_name == name and row.kind == ContentChangeKind.REGRESSION
        }
        corrections = {
            row.example_id
            for row in content_changes
            if row.run_name == name and row.kind == ContentChangeKind.CORRECTION
        }
        if len(regressions) >= policy.minimum_material_regressions and len(regressions) > len(
            corrections
        ):
            blocked.add(name)
    control = by_name[control_run]
    ordinary_limit = control.validation.old_ordinary.cross_entropy + policy.ordinary_ce_tolerance
    margin_floor = control.robust_neu_margin - policy.neu_margin_tolerance
    eligible = tuple(
        row
        for row in candidates
        if row.configuration.name not in blocked
        and row.validation.old_ordinary.cross_entropy <= ordinary_limit
        and row.robust_neu_margin >= margin_floor
    )
    assert eligible
    best_macro = min(row.validation.macro_cross_entropy for row in eligible)
    band = tuple(
        row
        for row in eligible
        if row.validation.macro_cross_entropy <= best_macro + policy.macro_ce_band
    )
    selected = min(
        band,
        key=lambda row: (
            row.validation.old_ordinary.cross_entropy,
            row.validation.macro_cross_entropy,
            row.configuration.name,
        ),
    )
    return AlignmentSelectionDecision(
        policy=policy,
        control_run=control_run,
        selected_run=selected.configuration.name,
        eligible_runs=tuple(row.configuration.name for row in eligible),
        macro_band_runs=tuple(row.configuration.name for row in band),
        content_changes=tuple(content_changes),
        validation_candidates=tuple(candidates),
    )
