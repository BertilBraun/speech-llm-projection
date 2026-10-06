from pathlib import Path

from speech_projector.models import EvaluationCondition, SampleGeneration
from speech_projector.report import aggregate_report


def test_empty_report_explicitly_reports_pending_stages(tmp_path: Path) -> None:
    path = aggregate_report(tmp_path)
    report = path.read_text(encoding="utf-8")
    assert report.count("No completed result is available yet.") == 4
    assert "frozen Qwen" in report
    assert (tmp_path / "summary.csv").exists()


def test_qualitative_report_excludes_archived_wrong_eos_outputs(tmp_path: Path) -> None:
    valid_directory = tmp_path / "v0_run" / "validation"
    archived_directory = tmp_path / "v0_run" / "validation_wrong_eos"
    training_probe_directory = tmp_path / "v0_run" / "overfit_probe"
    valid_directory.mkdir(parents=True)
    archived_directory.mkdir(parents=True)
    training_probe_directory.mkdir(parents=True)
    sample = SampleGeneration(
        example_id="example",
        dialogue_id="dialogue",
        condition=EvaluationCondition.SPEECH,
        history=(),
        user_transcript="hello",
        gold_response="hello there",
        generated_response="valid reply",
        duration=1.0,
    )
    (valid_directory / "evaluation_generations.jsonl").write_text(
        sample.model_dump_json() + "\n", encoding="utf-8"
    )
    (archived_directory / "evaluation_generations.jsonl").write_text(
        sample.model_copy(update={"generated_response": "invalid continuation"}).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    (training_probe_directory / "speech_generations.jsonl").write_text(
        sample.model_copy(
            update={"generated_response": "memorized training reply"}
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    aggregate_report(tmp_path)
    qualitative = (tmp_path / "qualitative_comparison.md").read_text(encoding="utf-8")
    assert "valid reply" in qualitative
    assert "invalid continuation" not in qualitative
    assert "memorized training reply" not in qualitative
    assert "v0_run/validation / speech" in qualitative
