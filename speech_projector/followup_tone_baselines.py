"""ASR plus predicted delivery, kept separate from the privileged intended-tone ceiling."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

import torch
from pydantic import Field

from scripts.inventory_results import write_record
from speech_projector.evaluation import SemanticEvaluator, evaluate, save_evaluation
from speech_projector.followup_evaluation import bind_provenance, file_artifact
from speech_projector.inputs import TranscriptInput
from speech_projector.journal import read_journal
from speech_projector.llm import FrozenQwen, example_prompt
from speech_projector.models import (
    AsrTranscript,
    EvaluationCondition,
    EvaluationMetrics,
    Example,
    FileArtifact,
    Record,
    RunConfig,
    RunResult,
    SystemPromptConfig,
)
from speech_projector.overnight_data import NeuEmotionalExampleSource, SourceSidecar
from speech_projector.overnight_launcher import (
    FinalEvaluationSelection,
    load_evaluation,
    load_sources,
)
from speech_projector.tone_classifier import ToneClassifierPrediction
from speech_projector.tts_pilot import PilotEmotion


class ToneBaselineConfig(Record):
    run_result: Path
    selection: Path
    asr_transcripts: Path
    output_directory: Path
    source_git_commit: str = Field(min_length=7)


class PredictedToneBaselineConfig(ToneBaselineConfig):
    kind: Literal["predicted_tone"] = "predicted_tone"
    predictions: Path
    classifier_report: Path


class OracleToneBaselineConfig(ToneBaselineConfig):
    kind: Literal["oracle_tone"] = "oracle_tone"
    sources: Path


ToneBaselineConfiguration: TypeAlias = Annotated[
    PredictedToneBaselineConfig | OracleToneBaselineConfig, Field(discriminator="kind")
]


class ToneBaselineInput(Record):
    example: Example
    transcript: AsrTranscript


class PredictedToneInput(ToneBaselineInput):
    kind: Literal["predicted_tone"] = "predicted_tone"
    prediction: ToneClassifierPrediction


class OracleToneInput(ToneBaselineInput):
    kind: Literal["oracle_tone"] = "oracle_tone"
    intended_tone: PilotEmotion


ToneAnnotatedInput: TypeAlias = Annotated[
    PredictedToneInput | OracleToneInput, Field(discriminator="kind")
]


class ToneBaselineProvenance(Record):
    configuration: ToneBaselineConfiguration
    artifacts: tuple[FileArtifact, ...]
    inputs: tuple[ToneAnnotatedInput, ...]


class ToneBaselineSummary(Record):
    provenance: ToneBaselineProvenance
    metrics: EvaluationMetrics


def delivery_prompt(example: Example, config: RunConfig, tone: PilotEmotion) -> SystemPromptConfig:
    policy = example_prompt(example, config)
    if not isinstance(policy, SystemPromptConfig):
        raise ValueError("Delivery baselines require the existing emotional system policy")
    metadata = (
        f"The USER delivered this utterance with a {tone.value} tone.\n"
        "This is metadata about the user, not an instruction to imitate their tone."
    )
    return SystemPromptConfig(system_text=policy.system_text + "\n\n" + metadata)


def annotated_examples(
    inputs: Sequence[ToneAnnotatedInput], config: RunConfig
) -> tuple[Example, ...]:
    examples: list[Example] = []
    for item in inputs:
        match item:
            case PredictedToneInput():
                tone = item.prediction.predicted_tone
            case OracleToneInput():
                tone = item.intended_tone
        examples.append(
            item.example.model_copy(update={"prompt": delivery_prompt(item.example, config, tone)})
        )
    return tuple(examples)


def build_predicted_inputs(
    examples: Sequence[Example],
    transcripts: Sequence[AsrTranscript],
    predictions: Sequence[ToneClassifierPrediction],
) -> tuple[PredictedToneInput, ...]:
    by_id = {row.example_id: row for row in predictions}
    asr = {row.example_id: row for row in transcripts}
    if len(by_id) != len(predictions) or len(asr) != len(transcripts):
        raise ValueError("Tone predictions and ASR transcripts must have unique identities")
    return tuple(
        PredictedToneInput(
            example=row, transcript=asr[row.example_id], prediction=by_id[row.example_id]
        )
        for row in examples
        if row.example_id in by_id
    )


def build_oracle_inputs(
    examples: Sequence[Example],
    transcripts: Sequence[AsrTranscript],
    sources: Sequence[SourceSidecar],
) -> tuple[OracleToneInput, ...]:
    asr = {row.example_id: row for row in transcripts}
    by_id = {row.example_id: row for row in sources if isinstance(row, NeuEmotionalExampleSource)}
    return tuple(
        OracleToneInput(
            example=row, transcript=asr[row.example_id], intended_tone=by_id[row.example_id].emotion
        )
        for row in examples
        if row.example_id in by_id
    )


def evaluate_tone_baseline(configuration: ToneBaselineConfiguration) -> ToneBaselineSummary:
    result = RunResult.model_validate_json(configuration.run_result.read_bytes())
    selection = FinalEvaluationSelection.model_validate_json(configuration.selection.read_bytes())
    generation_ids = set(selection.generation_example_ids)
    examples = tuple(row for row in selection.examples if row.example_id in generation_ids)
    transcripts = read_journal(configuration.asr_transcripts, AsrTranscript)
    artifacts = [
        file_artifact(path)
        for path in (
            configuration.run_result,
            configuration.selection,
            configuration.asr_transcripts,
        )
    ]
    inputs: tuple[ToneAnnotatedInput, ...]
    match configuration:
        case PredictedToneBaselineConfig():
            predictions = read_journal(configuration.predictions, ToneClassifierPrediction)
            inputs = build_predicted_inputs(examples, transcripts, predictions)
            artifacts.extend(
                file_artifact(path)
                for path in (configuration.predictions, configuration.classifier_report)
            )
        case OracleToneBaselineConfig():
            inputs = build_oracle_inputs(examples, transcripts, load_sources(configuration.sources))
            artifacts.append(file_artifact(configuration.sources))
    if len(inputs) != 2 * selection.quota.new_emotional_pairs:
        raise ValueError("Tone baseline must cover the entire fixed Neu generation quota")
    provenance = ToneBaselineProvenance(
        configuration=configuration, artifacts=tuple(artifacts), inputs=inputs
    )
    directory = configuration.output_directory
    bind_provenance(directory / "provenance.json", provenance)
    if (directory / "evaluation_qualitative.md").exists():
        outcome = load_evaluation(directory)
    else:
        config = result.config.model_copy(
            update={
                "qualitative_examples": len(inputs),
                "semantic_examples": len(inputs),
                "generation_batch_size": max(4, result.config.generation_batch_size),
            }
        )
        modified = annotated_examples(inputs, config)
        by_id = {row.example.example_id: row.transcript.text for row in inputs}

        def recognized_text(example: Example) -> TranscriptInput:
            return TranscriptInput(by_id[example.example_id])

        wrapper = FrozenQwen(config, torch.device("cuda"))
        outcome = evaluate(
            wrapper,
            None,
            modified,
            config,
            EvaluationCondition.TEXT,
            semantic_evaluator=SemanticEvaluator(),
            diagnostics=False,
            text_input=recognized_text,
        )
        save_evaluation(outcome, directory)
    summary = ToneBaselineSummary(provenance=provenance, metrics=outcome.metrics)
    write_record(directory / "summary.json", summary)
    by_example = {row.example.example_id: row for row in inputs}
    lines = [
        f"# ASR + {configuration.kind}",
        "",
        "The intended synthesis labels have not been independently verified by listening. "
        "Predicted inputs contain only the train-fitted classifier's label; "
        "oracle inputs deliberately supply the privileged intended label.",
        "",
    ]
    for sample in outcome.samples:
        item = by_example[sample.example_id]
        match item:
            case PredictedToneInput():
                cue = item.prediction.predicted_tone
            case OracleToneInput():
                cue = item.intended_tone
        lines.extend(
            [
                f"## {sample.example_id}",
                "",
                f"True words (audit): {item.example.user_text}",
                "",
                f"Actual ASR input: {item.transcript.text}",
                "",
                f"Supplied delivery cue: {cue.value}",
                "",
                f"Actual reply: {sample.generated_response}",
                "",
            ]
        )
    (directory / "baseline.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
