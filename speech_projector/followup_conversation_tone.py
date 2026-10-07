"""Recognized conversation words with a predicted cue attached only to the initial user."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from pydantic import Field, field_serializer

from scripts.inventory_results import write_record
from speech_projector.followup_conversation import (
    PipelineConversationReply,
    conversation_history,
    render_pipeline_conversations,
    summary,
    user_turns,
)
from speech_projector.followup_evaluation import bind_provenance, file_artifact
from speech_projector.followup_tone_baselines import PredictedToneInput, verify_prediction_file
from speech_projector.inputs import TranscriptInput
from speech_projector.journal import append_record, read_journal
from speech_projector.llm import ConversationRequest, FrozenQwen, example_prompt
from speech_projector.models import (
    AsrTranscript,
    EvaluationCondition,
    FileArtifact,
    GreedyDecodingConfig,
    Record,
    Role,
    RunConfig,
    RunResult,
    Split,
    SystemPromptConfig,
    Turn,
)
from speech_projector.overnight_conversation import DelayedCueFixtures
from speech_projector.overnight_conversation_execution import (
    ControlledConversationReply,
    ConversationHistoryMode,
    RolloutConversationReply,
    _assert_history_retained,
    reply_identifier,
    response_text,
)
from speech_projector.tone_classifier import ToneClassifierPrediction, ToneClassifierReport


class PredictedInitialToneConversationConfig(Record):
    run_result: Path
    fixtures: Path
    asr_transcripts: Path
    followup_transcripts: Path
    predictions: Path
    classifier_report: Path
    output_directory: Path
    source_git_commit: str = Field(min_length=7)
    history_tokens: int = Field(default=8192, ge=1)

    @field_serializer(
        "run_result",
        "fixtures",
        "asr_transcripts",
        "followup_transcripts",
        "predictions",
        "classifier_report",
        "output_directory",
    )
    def portable_path(self, path: Path) -> str:
        return path.as_posix()


class PredictedInitialToneConversationProvenance(Record):
    configuration: PredictedInitialToneConversationConfig
    inference_configuration: RunConfig
    artifacts: tuple[FileArtifact, ...]
    initial_inputs: tuple[PredictedToneInput, ...]
    transcripts: tuple[AsrTranscript, ...]
    fixtures_sha256: str


def initial_user_annotation(item: PredictedToneInput) -> str:
    return (
        f"[User delivery metadata: {item.prediction.predicted_tone.value}]\n{item.transcript.text}"
    )


def build_initial_inputs(
    fixtures: DelayedCueFixtures,
    transcripts: Sequence[AsrTranscript],
    predictions: Sequence[ToneClassifierPrediction],
    report: ToneClassifierReport,
) -> tuple[PredictedToneInput, ...]:
    recognized = {row.example_id: row for row in transcripts}
    predicted = {row.example_id: row for row in predictions}
    if len(recognized) != len(transcripts) or len(predicted) != len(predictions):
        raise ValueError("ASR and classifier prediction identities must be unique")
    inputs: list[PredictedToneInput] = []
    for scenario in fixtures.scenarios:
        for example, source in zip(scenario.initial, scenario.sources, strict=True):
            identifier = example.example_id
            if example.split == Split.TRAIN or source.example_id != identifier:
                raise ValueError("Initial tone cues require genuinely held-out Neu examples")
            if identifier not in report.selected_example_ids:
                raise ValueError(f"Classifier report does not cover initial fixture {identifier}")
            if identifier not in predicted or identifier not in recognized:
                raise ValueError(f"Initial fixture lacks a prediction or ASR words: {identifier}")
            inputs.append(
                PredictedToneInput(
                    example=example,
                    transcript=recognized[identifier],
                    prediction=predicted[identifier],
                )
            )
        for followup in scenario.followups:
            if followup.clip.case.case_id not in recognized:
                raise ValueError("A neutral follow-up lacks its saved ASR transcription")
    identifiers = tuple(item.example.example_id for item in inputs)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Initial tone fixtures repeat an example identity")
    return tuple(inputs)


def prepare_conversation_tone(
    configuration: PredictedInitialToneConversationConfig,
) -> tuple[DelayedCueFixtures, PredictedInitialToneConversationProvenance]:
    result = RunResult.model_validate_json(configuration.run_result.read_bytes())
    fixtures = DelayedCueFixtures.model_validate_json(configuration.fixtures.read_bytes())
    report = ToneClassifierReport.model_validate_json(configuration.classifier_report.read_bytes())
    prediction_artifact = verify_prediction_file(report, configuration.predictions)
    predictions = read_journal(configuration.predictions, ToneClassifierPrediction)
    transcripts = read_journal(configuration.asr_transcripts, AsrTranscript) + read_journal(
        configuration.followup_transcripts, AsrTranscript
    )
    inputs = build_initial_inputs(fixtures, transcripts, predictions, report)
    inference_configuration = result.config.model_copy(
        update={
            "history_turns": 6,
            "max_history_tokens": configuration.history_tokens,
            "max_new_tokens": 256,
            "decoding": GreedyDecodingConfig(),
        }
    )
    for item in inputs:
        match example_prompt(item.example, inference_configuration):
            case SystemPromptConfig():
                pass
            case _:
                raise ValueError("Neu conversation requires its original generic system policy")
    provenance = PredictedInitialToneConversationProvenance(
        configuration=configuration,
        inference_configuration=inference_configuration,
        artifacts=tuple(
            file_artifact(path)
            for path in (
                configuration.run_result,
                configuration.fixtures,
                configuration.asr_transcripts,
                configuration.followup_transcripts,
                configuration.classifier_report,
            )
        )
        + (prediction_artifact, report.model),
        initial_inputs=inputs,
        transcripts=transcripts,
        fixtures_sha256=hashlib.sha256(fixtures.model_dump_json().encode()).hexdigest(),
    )
    return fixtures, provenance


@torch.no_grad()
def evaluate_predicted_initial_tone(
    wrapper: FrozenQwen,
    fixtures: DelayedCueFixtures,
    provenance: PredictedInitialToneConversationProvenance,
) -> tuple[PipelineConversationReply, ...]:
    if wrapper.config != provenance.inference_configuration:
        raise ValueError("Conversation inference differs from its recorded configuration")
    if (
        hashlib.sha256(fixtures.model_dump_json().encode()).hexdigest()
        != provenance.fixtures_sha256
    ):
        raise ValueError("Conversation fixtures differ from their recorded provenance")
    directory = provenance.configuration.output_directory
    bind_provenance(directory / "provenance.json", provenance)
    journal = directory / "replies.jsonl"
    replies = list(read_journal(journal, PipelineConversationReply))
    saved = {reply_identifier(row.reply): row for row in replies}
    if len(saved) != len(replies):
        raise ValueError("Predicted-initial-tone conversation has duplicate saved replies")
    by_id = {item.example.example_id: item for item in provenance.initial_inputs}
    expected: set[str] = set()
    for scenario in fixtures.scenarios:
        for branch, initial in enumerate(scenario.initial):
            recognized = user_turns(
                scenario, branch, EvaluationCondition.ASR, provenance.transcripts
            )
            users = (initial_user_annotation(by_id[initial.example_id]),) + recognized[1:]
            truth = user_turns(scenario, branch, EvaluationCondition.TEXT, ())
            for turn_index in (2, 3):
                identifier = f"controlled:{initial.example_id}:text_history:{turn_index}"
                expected.add(identifier)
                if identifier in saved:
                    existing = saved[identifier]
                    if (
                        existing.user_words != users
                        or existing.condition != EvaluationCondition.ASR
                    ):
                        raise ValueError("Saved controlled conversation input differs")
                    continue
                request = ConversationRequest(
                    identifier=identifier,
                    prompt=example_prompt(initial, wrapper.config),
                    history=conversation_history(users, scenario.fixed_assistants, turn_index),
                    current=TranscriptInput(users[turn_index]),
                )
                _assert_history_retained(wrapper, request)
                started = time.perf_counter()
                generation = wrapper.generate_conversation(request, wrapper.config.max_new_tokens)
                reply = ControlledConversationReply(
                    scenario_base_id=scenario.base_id,
                    initial_example_id=initial.example_id,
                    current_case_id=scenario.followups[turn_index - 1].clip.case.case_id,
                    history_for_judge=conversation_history(
                        truth, scenario.fixed_assistants, turn_index
                    ),
                    history_mode=ConversationHistoryMode.TEXT_HISTORY,
                    turn_index=turn_index,
                    generation=generation,
                    generation_seconds=time.perf_counter() - started,
                )
                row = PipelineConversationReply(
                    condition=EvaluationCondition.ASR, user_words=users, reply=reply
                )
                append_record(journal, row)
                replies.append(row)
                saved[identifier] = row
            assistants: list[Turn] = []
            for turn_index in range(4):
                identifier = f"rollout:{initial.example_id}:{turn_index}"
                expected.add(identifier)
                existing = saved.get(identifier)
                if existing is None:
                    request = ConversationRequest(
                        identifier=identifier,
                        prompt=example_prompt(initial, wrapper.config),
                        history=conversation_history(users, assistants, turn_index),
                        current=TranscriptInput(users[turn_index]),
                    )
                    _assert_history_retained(wrapper, request)
                    started = time.perf_counter()
                    generation = wrapper.generate_conversation(
                        request, wrapper.config.max_new_tokens
                    )
                    reply = RolloutConversationReply(
                        scenario_base_id=scenario.base_id,
                        initial_example_id=initial.example_id,
                        current_case_id=initial.example_id
                        if turn_index == 0
                        else scenario.followups[turn_index - 1].clip.case.case_id,
                        history_for_judge=conversation_history(truth, assistants, turn_index),
                        turn_index=turn_index,
                        generation=generation,
                        generation_seconds=time.perf_counter() - started,
                    )
                    existing = PipelineConversationReply(
                        condition=EvaluationCondition.ASR, user_words=users, reply=reply
                    )
                    append_record(journal, existing)
                    replies.append(existing)
                    saved[identifier] = existing
                if existing.user_words != users or existing.condition != EvaluationCondition.ASR:
                    raise ValueError(
                        "Saved conversation uses a different recognized or annotated input"
                    )
                assistants.append(
                    Turn(role=Role.ASSISTANT, text=response_text(existing.reply.generation))
                )
    if set(saved) != expected:
        raise ValueError("Predicted-initial-tone journal has unexpected fixture coverage")
    write_record(directory / "summary.json", summary(EvaluationCondition.ASR, replies))
    introduction = (
        "# ASR + predicted initial delivery\n\n"
        "Every user turn uses saved ASR words. Only the initial USER message carries "
        "`[User delivery metadata: <classifier prediction>]`; that same message remains "
        "in later history. Neutral follow-ups have no delivery annotation. "
        "The generic system policy is unchanged. This is a prediction, not an oracle label. "
        "Judge histories retain true words for the matched evaluation; "
        "`user_words` records the actual recognized and annotated model input.\n\n"
    )
    (directory / "conversation.md").write_text(
        introduction + render_pipeline_conversations(replies), encoding="utf-8"
    )
    return tuple(replies)


def run_predicted_initial_tone(configuration: PredictedInitialToneConversationConfig) -> None:
    fixtures, provenance = prepare_conversation_tone(configuration)
    bind_provenance(configuration.output_directory / "provenance.json", provenance)
    wrapper = FrozenQwen(provenance.inference_configuration, torch.device("cuda"))
    evaluate_predicted_initial_tone(wrapper, fixtures, provenance)
