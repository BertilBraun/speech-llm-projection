from pathlib import Path

from speech_projector.report import aggregate_report


def test_empty_report_explicitly_reports_pending_stages(tmp_path: Path) -> None:
    path = aggregate_report(tmp_path)
    report = path.read_text(encoding="utf-8")
    assert report.count("No completed result is available yet.") == 4
    assert "frozen Qwen" in report
    assert (tmp_path / "summary.csv").exists()
