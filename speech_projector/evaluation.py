"""Held-out losses, fixed generations, baselines, and audio-conditioning controls."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import TYPE_CHECKING

import torch
from pydantic import TypeAdapter
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch.nn import functional

from speech_projector.inputs import SpeechInput, TranscriptInput, UtteranceInput
from speech_projector.models import (
    AsrTranscript,
    EvaluationCondition,
    EvaluationMetrics,
    Example,
    Record,
    RunConfig,
    SampleGeneration,
)

if TYPE_CHECKING:
    from speech_projector.llm import FrozenQwen
    from speech_projector.projectors import Projector


class ExampleLoss(Record):
    example_id: str
    dialogue_id: str
    condition: EvaluationCondition
    cross_entropy: float
    target_tokens: int


class ConditioningDiagnostic(Record):
    correct_condition: EvaluationCondition
    control_condition: EvaluationCondition
    paired_examples: int
    mean_control_minus_correct_ce: float
    standard_error: float
    fraction_correct_audio_lower_loss: float


@dataclass(frozen=True)
class EvaluationOutcome:
    metrics: EvaluationMetrics
    samples: tuple[SampleGeneration, ...]
    example_losses: tuple[ExampleLoss, ...]
    diagnostics: tuple[ConditioningDiagnostic, ...]


@dataclass(frozen=True)
class LossObservation:
    negative_log_likelihood: float
    target_tokens: int


def summarize_losses(observations: Sequence[LossObservation]) -> tuple[float, float, int]:
    if not observations:
        raise ValueError("Evaluation requires at least one observation")
    tokens = sum(observation.target_tokens for observation in observations)
    if tokens <= 0:
        raise ValueError("Evaluation requires positive assistant target token counts")
    cross_entropy = (
        sum(observation.negative_log_likelihood for observation in observations) / tokens
    )
    perplexity = math.exp(cross_entropy) if cross_entropy < 709 else math.inf
    return cross_entropy, perplexity, tokens


def conditioning_diagnostic(
    losses: Sequence[ExampleLoss],
    correct_condition: EvaluationCondition,
    control_condition: EvaluationCondition,
) -> ConditioningDiagnostic:
    correct = [loss for loss in losses if loss.condition == correct_condition]
    control = [loss for loss in losses if loss.condition == control_condition]
    if len(correct) != len(control) or not correct:
        raise ValueError("Conditioning comparisons require equal nonempty paired evaluations")
    differences: list[float] = []
    for correct_loss, control_loss in zip(correct, control, strict=True):
        if (correct_loss.example_id, correct_loss.target_tokens) != (
            control_loss.example_id,
            control_loss.target_tokens,
        ):
            raise ValueError("Conditioning comparisons must match examples and targets")
        differences.append(control_loss.cross_entropy - correct_loss.cross_entropy)
    return ConditioningDiagnostic(
        correct_condition=correct_condition,
        control_condition=control_condition,
        paired_examples=len(differences),
        mean_control_minus_correct_ce=mean(differences),
        standard_error=stdev(differences) / math.sqrt(len(differences))
        if len(differences) > 1
        else 0.0,
        fraction_correct_audio_lower_loss=sum(difference > 0 for difference in differences)
        / len(differences),
    )


class SemanticEvaluator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name, device="cpu")

    def similarities(self, responses: Sequence[str], references: Sequence[str]) -> list[float]:
        if len(responses) != len(references):
            raise ValueError("Responses and references must have equal lengths")
        if not responses:
            return []
        response_embeddings = self.model.encode(
            list(responses),
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        reference_embeddings = self.model.encode(
            list(references),
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        similarities = torch.sum(response_embeddings * reference_embeddings, dim=-1).tolist()
        return [float(similarity) for similarity in similarities]


def load_asr_transcripts(path: Path) -> tuple[AsrTranscript, ...]:
    return tuple(
        AsrTranscript.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _features(example: Example, device: torch.device) -> Tensor:
    return torch.load(example.feature_path, map_location="cpu", weights_only=True).to(device)


def _asr_text(example: Example, transcripts: Sequence[AsrTranscript]) -> str:
    for transcript in transcripts:
        if transcript.example_id == example.example_id:
            return transcript.text
    raise ValueError(f"Missing ASR transcript for {example.example_id}")


def _mismatched_example(examples: Sequence[Example], index: int) -> Example:
    for offset in range(1, len(examples)):
        candidate = examples[(index + offset) % len(examples)]
        if candidate.dialogue_id != examples[index].dialogue_id:
            return candidate
    raise ValueError("Shuffled-audio diagnostics require at least two different dialogues")


def match_feature_length(shuffled_features: Tensor, target_length: int) -> Tensor:
    """Hold pseudo-token count fixed while replacing current-utterance information."""
    if target_length < 1:
        raise ValueError("Matched speech feature length must be positive")
    if shuffled_features.shape[0] == target_length:
        return shuffled_features
    interpolated = functional.interpolate(
        shuffled_features.transpose(0, 1).unsqueeze(0).float(),
        size=target_length,
        mode="linear",
        align_corners=False,
    )
    return interpolated[0].transpose(0, 1).to(shuffled_features.dtype)


def _record_loss(
    wrapper: FrozenQwen,
    example: Example,
    condition: EvaluationCondition,
    utterance: UtteranceInput,
) -> ExampleLoss:
    return ExampleLoss(
        example_id=example.example_id,
        dialogue_id=example.dialogue_id,
        condition=condition,
        cross_entropy=float(wrapper.loss(example, utterance)),
        target_tokens=wrapper.target_token_count(example),
    )


def _mean_ce(losses: Sequence[ExampleLoss], condition: EvaluationCondition) -> float:
    cross_entropy, _, _ = summarize_losses(
        [
            LossObservation(loss.cross_entropy * loss.target_tokens, loss.target_tokens)
            for loss in losses
            if loss.condition == condition
        ]
    )
    return cross_entropy


def original_dialogue_text(example: Example) -> TranscriptInput:
    return TranscriptInput(example.user_text)


@torch.no_grad()
def evaluate(
    wrapper: FrozenQwen,
    projector: Projector | None,
    examples: Sequence[Example],
    config: RunConfig,
    condition: EvaluationCondition,
    asr_transcripts: Sequence[AsrTranscript] = (),
    semantic_evaluator: SemanticEvaluator | None = None,
    diagnostics: bool = True,
    *,
    text_input: Callable[[Example], TranscriptInput] = original_dialogue_text,
) -> EvaluationOutcome:
    if not examples:
        raise ValueError("Evaluation requires examples")
    if condition == EvaluationCondition.SPEECH and projector is None:
        raise ValueError("Speech evaluation requires a projector")
    if condition not in (
        EvaluationCondition.TEXT,
        EvaluationCondition.ASR,
        EvaluationCondition.SPEECH,
    ):
        raise ValueError("Select text, ASR, or speech as the primary evaluation condition")
    wrapper.model.eval()
    if projector is not None:
        projector.eval()
    started = time.perf_counter()
    losses: list[ExampleLoss] = []
    samples: list[SampleGeneration] = []
    control_samples: list[SampleGeneration] = []
    generation_seconds = 0.0
    generated_tokens = 0
    for index, example in enumerate(examples):
        utterance: UtteranceInput
        match condition:
            case EvaluationCondition.TEXT:
                utterance = text_input(example)
            case EvaluationCondition.ASR:
                utterance = TranscriptInput(_asr_text(example, asr_transcripts))
            case EvaluationCondition.SPEECH:
                assert projector is not None
                current_features = _features(example, wrapper.device)
                utterance = SpeechInput(projector(current_features))
        losses.append(_record_loss(wrapper, example, condition, utterance))
        match utterance:
            case SpeechInput(embeddings=embeddings) if diagnostics and index < min(
                config.conditioning_examples, len(examples)
            ):
                assert projector is not None
                shuffled = projector(
                    match_feature_length(
                        _features(_mismatched_example(examples, index), wrapper.device),
                        current_features.shape[0],
                    )
                )
                no_history = example.model_copy(update={"history": ()})
                controls = (
                    (EvaluationCondition.SHUFFLED_SPEECH, example, shuffled),
                    (EvaluationCondition.ZERO_SPEECH, example, torch.zeros_like(embeddings)),
                    (EvaluationCondition.SPEECH_NO_HISTORY, no_history, embeddings),
                    (EvaluationCondition.SHUFFLED_SPEECH_NO_HISTORY, no_history, shuffled),
                )
                for control_condition, control_example, control_embeddings in controls:
                    losses.append(
                        _record_loss(
                            wrapper,
                            control_example,
                            control_condition,
                            SpeechInput(control_embeddings),
                        )
                    )
                    if index < min(8, config.qualitative_examples) and control_condition in (
                        EvaluationCondition.SHUFFLED_SPEECH,
                        EvaluationCondition.SPEECH_NO_HISTORY,
                        EvaluationCondition.SHUFFLED_SPEECH_NO_HISTORY,
                    ):
                        control_response = wrapper.generate(
                            control_example, SpeechInput(control_embeddings)
                        )
                        control_samples.append(
                            SampleGeneration(
                                example_id=example.example_id,
                                dialogue_id=example.dialogue_id,
                                condition=control_condition,
                                history=control_example.history,
                                user_transcript=example.user_text,
                                gold_response=example.target_text,
                                generated_response=control_response,
                                duration=example.duration,
                                pseudo_tokens=control_embeddings.shape[0],
                            )
                        )
        if index < max(config.qualitative_examples, config.semantic_examples):
            generation_started = time.perf_counter()
            response = wrapper.generate(example, utterance)
            elapsed = time.perf_counter() - generation_started
            tokens = len(wrapper.tokenizer.encode(response, add_special_tokens=False))
            generation_seconds += elapsed
            generated_tokens += tokens
            match utterance:
                case SpeechInput(embeddings=embeddings):
                    pseudo_tokens = embeddings.shape[0]
                    recognized_transcript = None
                    user_transcript = example.user_text
                case TranscriptInput(text=text):
                    pseudo_tokens = None
                    recognized_transcript = text if condition == EvaluationCondition.ASR else None
                    user_transcript = (
                        text if condition == EvaluationCondition.TEXT else example.user_text
                    )
            samples.append(
                SampleGeneration(
                    example_id=example.example_id,
                    dialogue_id=example.dialogue_id,
                    condition=condition,
                    history=example.history,
                    user_transcript=user_transcript,
                    asr_transcript=recognized_transcript,
                    gold_response=example.target_text,
                    generated_response=response,
                    duration=example.duration,
                    pseudo_tokens=pseudo_tokens,
                )
            )
    similarity: float | None = None
    if semantic_evaluator is not None and samples:
        scores = semantic_evaluator.similarities(
            [sample.generated_response for sample in samples],
            [sample.gold_response for sample in samples],
        )
        samples = [
            sample.model_copy(update={"semantic_similarity": score})
            for sample, score in zip(samples, scores, strict=True)
        ]
        similarity = mean(scores)
    primary_losses = [loss for loss in losses if loss.condition == condition]
    cross_entropy, perplexity, target_tokens = summarize_losses(
        [
            LossObservation(loss.cross_entropy * loss.target_tokens, loss.target_tokens)
            for loss in primary_losses
        ]
    )
    controls_enabled = condition == EvaluationCondition.SPEECH and diagnostics
    metrics = EvaluationMetrics(
        examples=len(examples),
        target_tokens=target_tokens,
        cross_entropy=cross_entropy,
        perplexity=perplexity,
        semantic_similarity=similarity,
        generated_examples=len(samples),
        generation_seconds=generation_seconds,
        generated_tokens=generated_tokens,
        shuffled_audio_cross_entropy=_mean_ce(losses, EvaluationCondition.SHUFFLED_SPEECH)
        if controls_enabled
        else None,
        zero_audio_cross_entropy=_mean_ce(losses, EvaluationCondition.ZERO_SPEECH)
        if controls_enabled
        else None,
        no_history_cross_entropy=_mean_ce(losses, EvaluationCondition.SPEECH_NO_HISTORY)
        if controls_enabled
        else None,
        no_history_shuffled_cross_entropy=_mean_ce(
            losses, EvaluationCondition.SHUFFLED_SPEECH_NO_HISTORY
        )
        if controls_enabled
        else None,
        evaluation_seconds=time.perf_counter() - started,
    )
    comparisons = (
        (EvaluationCondition.SPEECH, EvaluationCondition.SHUFFLED_SPEECH),
        (EvaluationCondition.SPEECH, EvaluationCondition.ZERO_SPEECH),
        (EvaluationCondition.SPEECH_NO_HISTORY, EvaluationCondition.SHUFFLED_SPEECH_NO_HISTORY),
    )
    paired_losses = [
        loss
        for loss in losses
        if loss.example_id
        in {example.example_id for example in examples[: config.conditioning_examples]}
    ]
    diagnostic_results = (
        tuple(
            conditioning_diagnostic(paired_losses, correct, control)
            for correct, control in comparisons
        )
        if controls_enabled
        else ()
    )
    print(
        f"Evaluation {condition.value}: CE={cross_entropy:.4f}, "
        f"seconds={metrics.evaluation_seconds:.1f}",
        flush=True,
    )
    return EvaluationOutcome(
        metrics, tuple(samples + control_samples), tuple(losses), diagnostic_results
    )


def save_evaluation(
    outcome: EvaluationOutcome, output_directory: Path, name: str = "evaluation"
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / f"{name}.json").write_text(
        outcome.metrics.model_dump_json(indent=2), encoding="utf-8"
    )
    (output_directory / f"{name}_conditioning.json").write_bytes(
        TypeAdapter(tuple[ConditioningDiagnostic, ...]).dump_json(outcome.diagnostics, indent=2)
    )
    with (output_directory / f"{name}_losses.jsonl").open("w", encoding="utf-8") as output:
        for loss in outcome.example_losses:
            output.write(loss.model_dump_json() + "\n")
    with (output_directory / f"{name}_generations.jsonl").open("w", encoding="utf-8") as output:
        for sample in outcome.samples:
            output.write(sample.model_dump_json() + "\n")
    lines = ["# Fixed qualitative sample set", ""]
    for sample in outcome.samples:
        history = "\n".join(f"{turn.role.value}: {turn.text}" for turn in sample.history)
        lines.extend(
            [
                f"## {sample.example_id} (dialogue {sample.dialogue_id})",
                "",
                f"History: {history or '[none]'}",
                "",
                f"User transcript (audit only): {sample.user_transcript}",
                "",
                f"Gold response: {sample.gold_response}",
                "",
                f"**{sample.condition.value}**: {sample.generated_response}",
                "",
            ]
        )
        if sample.asr_transcript is not None:
            lines.extend([f"ASR input transcript: {sample.asr_transcript}", ""])
    (output_directory / f"{name}_qualitative.md").write_text("\n".join(lines), encoding="utf-8")
