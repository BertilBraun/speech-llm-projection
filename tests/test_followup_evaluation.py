"""Follow-up provenance and distinct tone-input policies must remain auditable."""

from pathlib import Path

import pytest

from speech_projector.acoustic_tone_probe import LABELS
from speech_projector.followup_evaluation import FollowupEvaluationConfig, bind_provenance
from speech_projector.followup_tone_baselines import (
    PredictedToneInput,
    annotated_examples,
    build_predicted_inputs,
)
from speech_projector.models import AsrTranscript, SystemPromptConfig
from speech_projector.tone_classifier import ToneClassifierPrediction, ToneProbability
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
