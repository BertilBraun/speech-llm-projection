"""Quality exports preserve paired directions and uncertainty separately for both axes."""

import csv
from pathlib import Path

from scripts.plot_followup_quality import write_quality_plot
from speech_projector.followup_quality import (
    CohortQualityComparison,
    QualityComparisonConfig,
    QualityComparisonProvenance,
    QualityResponseSource,
    QualityScoredReport,
    QualitySlotMapping,
    summarize_ratings,
)
from speech_projector.models import EvaluationCondition, FileArtifact
from speech_projector.overnight_data import Cohort
from speech_projector.response_quality import CaseRating, RatingConfidence, ResponseRating
from tests.test_followup_quality import card


def test_quality_csv_keeps_tone_gain_grounding_loss_and_bootstrap_intervals(tmp_path: Path) -> None:
    cards = (card("a"), card("b"))
    ratings = tuple(
        CaseRating(
            example_id=row.example_id,
            response_a=ResponseRating(
                tone_helpfulness=2, literal_grounding=0, reason="warm but wrong"
            ),
            response_b=ResponseRating(
                tone_helpfulness=1, literal_grounding=2, reason="generic but faithful"
            ),
            confidence=RatingConfidence.HIGH,
            ambiguity_note="",
        )
        for row in cards
    )
    mapping = tuple(
        QualitySlotMapping(
            example_id=row.example_id, source_a="full_run_name", source_b="plain_asr"
        )
        for row in cards
    )
    summary = summarize_ratings(cards, ratings, mapping, "full_run_name")
    configuration = QualityComparisonConfig(
        selection=Path("selection"),
        sources=Path("sources"),
        first=QualityResponseSource(
            name="full_run_name", generations=Path("first"), condition=EvaluationCondition.SPEECH
        ),
        second=QualityResponseSource(
            name="plain_asr", generations=Path("second"), condition=EvaluationCondition.ASR
        ),
        cohorts=(Cohort.NEU_EMOTIONAL,),
        output_directory=tmp_path,
    )
    report = QualityScoredReport(
        provenance=QualityComparisonProvenance(configuration=configuration, artifacts=()),
        ratings=FileArtifact(path=Path("locked"), source_path=Path("locked"), bytes=1, sha256="a"),
        overall=summary,
        cohorts=(CohortQualityComparison(cohort=Cohort.NEU_EMOTIONAL, comparison=summary),),
    )
    write_quality_plot(report, tmp_path, "Speech 10Hz CE", "ASR")
    with (tmp_path / "quality_comparison.csv").open(encoding="utf-8", newline="") as stream:
        rows = tuple(csv.reader(stream))
    assert len(rows) == 3
    assert rows[1][:6] == ["full_run_name", "plain_asr", "neu_emotional", "2", "2", "tone"]
    assert tuple(float(value) for value in rows[1][8:11]) == (1, 1, 1)
    assert tuple(float(value) for value in rows[2][8:11]) == (-2, -2, -2)
    assert rows[1][11:14] == ["2", "0", "0"]
    assert rows[2][11:14] == ["0", "0", "2"]
    assert (tmp_path / "quality_comparison.png").is_file()
    assert (tmp_path / "quality_comparison.pdf").is_file()
