"""A classifier cue belongs to the first recognized USER message, never later turns."""

import hashlib
from pathlib import Path
from typing import cast

import pytest

from speech_projector.acoustic_tone_probe import LABELS
from speech_projector.followup_conversation_tone import (
    PredictedInitialToneConversationConfig,
    PredictedInitialToneConversationProvenance,
    build_initial_inputs,
    evaluate_predicted_initial_tone,
    prepare_conversation_tone,
)
from speech_projector.followup_evaluation import file_artifact
from speech_projector.followup_tone_baselines import PredictedToneInput
from speech_projector.inputs import TranscriptInput
from speech_projector.llm import FrozenQwen
from speech_projector.models import (
    AsrTranscript,
    GreedyDecodingConfig,
    Role,
    Split,
    SystemPromptConfig,
    Turn,
)
from speech_projector.overnight_conversation import (
    DelayedCueConfiguration,
    DelayedCueFixtures,
    build_delayed_cue_fixtures,
)
from speech_projector.overnight_data import NeuEmotionalExampleSource
from speech_projector.tone_classifier import (
    ToneClassifierConfig,
    ToneClassifierPrediction,
    ToneClassifierReport,
    ToneProbability,
)
from speech_projector.tts_pilot import PilotEmotion
from tests.test_overnight_conversation import followups
from tests.test_overnight_conversation_execution import ConversationWrapper
from tests.test_overnight_evaluation import candidate, example
from tests.test_overnight_report import result


def fixtures() -> DelayedCueFixtures:
    examples = tuple(
        example(f"base_{base}_{tone.value}", f"The meeting {base} moved to five.").model_copy(
            update={
                "split": Split.TEST,
                "prompt": SystemPromptConfig(system_text="Respond naturally in two sentences."),
            }
        )
        for base in range(3)
        for tone in (PilotEmotion.HAPPY, PilotEmotion.ANGRY)
    )
    sources = tuple(
        NeuEmotionalExampleSource(
            example_id=row.example_id,
            source_manifest=Path("source"),
            source_example_id=row.example_id,
            base_id=f"base_{index // 2}",
            family_id=f"family_{index // 2}",
            emotion=PilotEmotion.HAPPY if index % 2 == 0 else PilotEmotion.ANGRY,
        )
        for index, row in enumerate(examples)
    )
    return build_delayed_cue_fixtures(examples, sources, followups(), DelayedCueConfiguration())


def recognized_words(panel: DelayedCueFixtures) -> tuple[AsrTranscript, ...]:
    return tuple(
        AsrTranscript(example_id=row.example_id, text="The meeting moved to nine.")
        for scenario in panel.scenarios
        for row in scenario.initial
    ) + tuple(
        AsrTranscript(example_id=row.clip.case.case_id, text=f"Recognized follow-up {index}.")
        for index, row in enumerate(panel.scenarios[0].followups)
    )


def predictions(panel: DelayedCueFixtures) -> tuple[ToneClassifierPrediction, ...]:
    return tuple(
        ToneClassifierPrediction(
            example_id=example.example_id,
            predicted_tone=PilotEmotion.SAD,
            confidence=1,
            probabilities=tuple(
                ToneProbability(tone=tone, probability=float(tone == PilotEmotion.SAD))
                for tone in LABELS
            ),
        )
        for scenario in panel.scenarios
        for example in scenario.initial
    )


def classifier_report(
    path: Path, prediction_rows: tuple[ToneClassifierPrediction, ...]
) -> ToneClassifierReport:
    path.write_text(
        "".join(row.model_dump_json() + "\n" for row in prediction_rows), encoding="utf-8"
    )
    artifact = file_artifact(path)
    return ToneClassifierReport(
        configuration=ToneClassifierConfig(
            manifest=path.parent / "manifest",
            sidecar=path.parent / "sources",
            output_directory=path.parent,
        ),
        manifest=artifact,
        sidecar=artifact,
        features=(),
        selected_example_ids=tuple(row.example_id for row in prediction_rows),
        model=artifact,
        prediction_files=(artifact,),
        metrics=(),
        feature_dimension=4608,
        feature_loading_seconds=0,
        fitting_seconds=0,
        total_wall_seconds=0,
        sklearn_version="test",
        source_git_commit="commit42",
        limitations=(),
    )


def provenance(
    directory: Path, panel: DelayedCueFixtures, initial: tuple[PredictedToneInput, ...]
) -> PredictedInitialToneConversationProvenance:
    return PredictedInitialToneConversationProvenance(
        configuration=PredictedInitialToneConversationConfig(
            run_result=Path("run/result.json"),
            fixtures=Path("fixtures.json"),
            asr_transcripts=Path("asr.jsonl"),
            followup_transcripts=Path("followup.jsonl"),
            predictions=Path("predictions.jsonl"),
            classifier_report=Path("classifier.json"),
            output_directory=directory,
            source_git_commit="commit42",
        ),
        inference_configuration=candidate("test", 1, 1, 0.1, 10).configuration.model_copy(
            update={
                "history_turns": 6,
                "max_history_tokens": 8192,
                "max_new_tokens": 256,
                "decoding": GreedyDecodingConfig(),
            }
        ),
        artifacts=(),
        initial_inputs=initial,
        transcripts=recognized_words(panel),
        fixtures_sha256=hashlib.sha256(panel.model_dump_json().encode()).hexdigest(),
    )


def test_initial_prediction_stays_in_its_user_message_and_resume_generates_nothing(
    tmp_path: Path,
) -> None:
    panel = fixtures()
    recognized = recognized_words(panel)
    predicted = predictions(panel)
    report = classifier_report(tmp_path / "predictions.jsonl", predicted)
    initial = build_initial_inputs(panel, recognized, predicted, report)
    proof = provenance(tmp_path / "evaluation", panel, initial)
    wrapper = ConversationWrapper(proof.inference_configuration)
    replies = evaluate_predicted_initial_tone(cast(FrozenQwen, wrapper), panel, proof)
    assert len(replies) == len(wrapper.calls) == 36
    assert sum(row.reply.kind == "controlled" for row in replies) == 12
    assert sum(row.reply.kind == "rollout" for row in replies) == 24
    annotation = "[User delivery metadata: sad]\nThe meeting moved to nine."
    for request in wrapper.calls:
        assert request.prompt == SystemPromptConfig(
            system_text="Respond naturally in two sentences."
        )
        assert "sad" not in request.prompt.system_text
        if not request.history:
            assert request.current == TranscriptInput(annotation)
        else:
            assert request.history[0] == Turn(role=Role.USER, text=annotation)
            assert all("delivery metadata" not in turn.text for turn in request.history[1:])
            assert isinstance(request.current, TranscriptInput)
            assert request.current.text.startswith("Recognized follow-up")
            assert "metadata" not in request.current.text
    controlled = next(row for row in wrapper.calls if row.identifier.endswith("text_history:2"))
    assert controlled.history[1] == panel.scenarios[0].fixed_assistants[0]
    rollout = next(
        row
        for row in wrapper.calls
        if row.identifier.startswith("rollout:") and row.identifier.endswith(":2")
    )
    assert rollout.history[1].text == "Actual generated assistant text."
    assert replies[0].reply.history_for_judge[0].text.endswith("moved to five.")
    assert all(row.user_words[0] == annotation for row in replies)
    assert evaluate_predicted_initial_tone(cast(FrozenQwen, wrapper), panel, proof) == replies
    assert len(wrapper.calls) == 36
    changed = initial[0].model_copy(
        update={"transcript": initial[0].transcript.model_copy(update={"text": "Different words."})}
    )
    with pytest.raises(ValueError, match="provenance changed"):
        evaluate_predicted_initial_tone(
            cast(FrozenQwen, wrapper),
            panel,
            proof.model_copy(update={"initial_inputs": (changed,) + initial[1:]}),
        )
    assert len(wrapper.calls) == 36


@pytest.mark.parametrize("missing", ("prediction", "initial_asr", "followup_asr", "report"))
def test_partial_classifier_or_asr_coverage_fails_instead_of_falling_back(
    tmp_path: Path, missing: str
) -> None:
    panel = fixtures()
    recognized = recognized_words(panel)
    predicted = predictions(panel)
    report = classifier_report(tmp_path / "predictions.jsonl", predicted)
    if missing == "prediction":
        predicted = predicted[1:]
    elif missing == "initial_asr":
        recognized = recognized[1:]
    elif missing == "followup_asr":
        recognized = recognized[:-1]
    else:
        report = report.model_copy(update={"selected_example_ids": ()})
    with pytest.raises(ValueError, match="lacks|does not cover"):
        build_initial_inputs(panel, recognized, predicted, report)


def test_duplicate_predictions_and_training_inputs_are_rejected(tmp_path: Path) -> None:
    panel = fixtures()
    recognized = recognized_words(panel)
    predicted = predictions(panel)
    report = classifier_report(tmp_path / "predictions.jsonl", predicted)
    with pytest.raises(ValueError, match="unique"):
        build_initial_inputs(panel, recognized, predicted + predicted[:1], report)
    first = panel.scenarios[0]
    changed = first.model_copy(
        update={
            "initial": (
                first.initial[0].model_copy(update={"split": Split.TRAIN}),
                first.initial[1],
            )
        }
    )
    contaminated = panel.model_copy(update={"scenarios": (changed,) + panel.scenarios[1:]})
    with pytest.raises(ValueError, match="held-out"):
        build_initial_inputs(contaminated, recognized, predicted, report)


def test_prepare_binds_exact_prediction_bytes_and_shared_greedy_policy(tmp_path: Path) -> None:
    panel = fixtures()
    predicted = predictions(panel)
    report = classifier_report(tmp_path / "predictions.jsonl", predicted)
    (tmp_path / "classifier.json").write_text(report.model_dump_json(), encoding="utf-8")
    (tmp_path / "fixtures.json").write_text(panel.model_dump_json(), encoding="utf-8")
    (tmp_path / "result.json").write_text(result().model_dump_json(), encoding="utf-8")
    recognized = recognized_words(panel)
    (tmp_path / "initial_asr.jsonl").write_text(
        "".join(row.model_dump_json() + "\n" for row in recognized[:6]), encoding="utf-8"
    )
    (tmp_path / "followup_asr.jsonl").write_text(
        "".join(row.model_dump_json() + "\n" for row in recognized[6:]), encoding="utf-8"
    )
    configuration = PredictedInitialToneConversationConfig(
        run_result=tmp_path / "result.json",
        fixtures=tmp_path / "fixtures.json",
        asr_transcripts=tmp_path / "initial_asr.jsonl",
        followup_transcripts=tmp_path / "followup_asr.jsonl",
        predictions=tmp_path / "predictions.jsonl",
        classifier_report=tmp_path / "classifier.json",
        output_directory=tmp_path / "evaluation",
        source_git_commit="commit42",
    )
    loaded, proof = prepare_conversation_tone(configuration)
    assert loaded == panel
    assert len(proof.initial_inputs) == 6
    assert isinstance(proof.inference_configuration.decoding, GreedyDecodingConfig)
    assert proof.inference_configuration.history_turns == 6
    assert proof.inference_configuration.max_history_tokens == 8192
    assert proof.inference_configuration.max_new_tokens == 256
    assert proof.artifacts[5] == file_artifact(configuration.predictions)
    configuration.predictions.write_text("unbound changed predictions\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not bound"):
        prepare_conversation_tone(configuration)


def test_configuration_paths_serialize_portably() -> None:
    configuration = PredictedInitialToneConversationConfig(
        run_result=Path("/workspace/result.json"),
        fixtures=Path("/workspace/fixtures.json"),
        asr_transcripts=Path("/workspace/asr.jsonl"),
        followup_transcripts=Path("/workspace/followup.jsonl"),
        predictions=Path("/workspace/predictions.jsonl"),
        classifier_report=Path("/workspace/classifier.json"),
        output_directory=Path("/workspace/tone_conversation"),
        source_git_commit="commit42",
    )
    assert "/workspace/result.json" in configuration.model_dump_json()
    assert "\\\\workspace" not in configuration.model_dump_json()
