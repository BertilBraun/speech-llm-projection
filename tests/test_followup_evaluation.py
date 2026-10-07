"""Follow-up provenance and distinct tone-input policies must remain auditable."""

from pathlib import Path

import pytest

from speech_projector.acoustic_tone_probe import LABELS
from speech_projector.followup_evaluation import (
    FollowupEvaluationConfig,
    bind_provenance,
    file_artifact,
)
from speech_projector.followup_tone_baselines import (
    OracleToneInput,
    PredictedToneInput,
    annotated_examples,
    build_predicted_inputs,
    verify_oracle_identity,
    verify_prediction_file,
)
from speech_projector.models import AsrTranscript, SystemPromptConfig
from speech_projector.tone_classifier import (
    ToneClassifierConfig,
    ToneClassifierPrediction,
    ToneClassifierReport,
    ToneProbability,
)
from speech_projector.tts_pilot import PilotEmotion
from tests.test_overnight_evaluation import candidate, example


def test_provenance_rejects_a_different_checkpoint_or_manifest(tmp_path: Path) -> None:
    configuration = FollowupEvaluationConfig(
        run_result=Path("run/result.json"),
        selection=Path("selection.json"),
        sources=Path("sources.jsonl"),
        asr_transcripts=Path("asr.jsonl"),
        output_directory=tmp_path,
        source_git_commit="commit42",
    )
    path = tmp_path / "provenance.json"
    bind_provenance(path, configuration)
    bind_provenance(path, configuration)
    with pytest.raises(ValueError, match="provenance changed"):
        bind_provenance(
            path, configuration.model_copy(update={"selection": Path("different.json")})
        )


def test_predicted_tone_is_used_without_oracle_or_tts_instructions() -> None:
    source = example("neu", "The meeting moved to five.").model_copy(
        update={
            "prompt": SystemPromptConfig(system_text="Reply naturally."),
            "emotion": "happy",
        }
    )
    prediction = ToneClassifierPrediction(
        example_id="neu",
        predicted_tone=PilotEmotion.ANGRY,
        confidence=1,
        probabilities=tuple(
            ToneProbability(tone=tone, probability=float(tone == PilotEmotion.ANGRY))
            for tone in LABELS
        ),
    )
    inputs = build_predicted_inputs(
        (source,),
        (AsrTranscript(example_id="neu", text="The meeting moved to nine."),),
        (prediction,),
    )
    assert isinstance(inputs[0], PredictedToneInput)
    modified = annotated_examples(inputs, candidate("test", 1, 1, 0.1, 10).configuration)[0]
    assert isinstance(modified.prompt, SystemPromptConfig)
    assert modified.prompt.system_text == (
        "Reply naturally.\n\nThe USER delivered this utterance with a angry tone.\n"
        "This is metadata about the user, not an instruction to imitate their tone."
    )
    assert "happy" not in modified.prompt.system_text
    assert source.prompt.system_text == "Reply naturally."
    assert source.user_text == modified.user_text == "The meeting moved to five."
    assert inputs[0].transcript.text == "The meeting moved to nine."
    assert source.target_text == modified.target_text
    oracle = OracleToneInput(
        example=source, transcript=inputs[0].transcript, intended_tone=PilotEmotion.ANGRY
    )
    verify_oracle_identity(inputs, (oracle,))
    with pytest.raises(ValueError, match="actual inputs differ"):
        verify_oracle_identity(
            inputs, (oracle.model_copy(update={"intended_tone": PilotEmotion.HAPPY}),)
        )


def test_prediction_bytes_must_be_bound_to_fitted_classifier_report(tmp_path: Path) -> None:
    source = tmp_path / "predictions.jsonl"
    source.write_text("reviewed prediction bytes\n", encoding="utf-8")
    artifact = file_artifact(source)
    report = ToneClassifierReport(
        configuration=ToneClassifierConfig(
            manifest=tmp_path / "manifest", sidecar=tmp_path / "sources", output_directory=tmp_path
        ),
        manifest=artifact,
        sidecar=artifact,
        features=(),
        selected_example_ids=(),
        model=artifact,
        prediction_files=(artifact,),
        metrics=(),
        feature_dimension=4608,
        feature_loading_seconds=0,
        fitting_seconds=0,
        total_wall_seconds=0,
        sklearn_version="test",
        source_git_commit="test",
        limitations=(),
    )
    copied = tmp_path / "renamed.jsonl"
    copied.write_bytes(source.read_bytes())
    verify_prediction_file(report, copied)
    copied.write_text("unbound oracle labels\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not bound"):
        verify_prediction_file(report, copied)
