"""Actual cached Qwen transcript responses as the distillation text reference."""

import time
from collections.abc import Sequence
from pathlib import Path

import torch

from scripts.package_results import FileArtifact, file_digest
from speech_projector.evaluation import (
    EvaluationOutcome,
    ExampleLoss,
    LossObservation,
    SemanticEvaluator,
    evaluate,
    load_asr_transcripts,
    save_evaluation,
    summarize_losses,
)
from speech_projector.inputs import TranscriptInput
from speech_projector.journal import read_journal
from speech_projector.llm import FrozenQwen
from speech_projector.models import (
    EvaluationCondition,
    EvaluationMetrics,
    Example,
    GenerationDetails,
    GenerationKind,
    Record,
    RunConfig,
    SampleGeneration,
    Split,
)
from speech_projector.teacher import TeacherProvenance, TeacherTarget, target_example


class TeacherBaselineConfig(Record):
    manifest: Path
    teacher_directory: Path
    output_root: Path


class TeacherBaselineProvenance(Record):
    evaluation_config: RunConfig
    teacher: TeacherProvenance
    target_journal: FileArtifact
    examples: tuple[Example, ...]


@torch.no_grad()
def evaluate_cached_teacher(
    wrapper: FrozenQwen,
    examples: Sequence[Example],
    targets: Sequence[TeacherTarget],
) -> EvaluationOutcome:
    if not examples:
        raise ValueError("Cached teacher baseline requires examples")
    selected = {item.example.example_id: item for item in targets}
    if len(selected) != len(targets):
        raise ValueError("Teacher journal contains duplicate example IDs")
    started = time.perf_counter()
    losses: list[ExampleLoss] = []
    samples: list[SampleGeneration] = []
    generated_tokens = 0
    for example in examples:
        if example.example_id not in selected:
            raise ValueError(f"Teacher response missing for {example.example_id}")
        teacher = selected[example.example_id]
        if target_example(teacher, wrapper) != example:
            raise ValueError("Cached teacher input/history/target differs from evaluation manifest")
        tokens = wrapper.target_token_count(example)
        losses.append(
            ExampleLoss(
                example_id=example.example_id,
                dialogue_id=example.dialogue_id,
                condition=EvaluationCondition.TEXT,
                cross_entropy=float(wrapper.loss(example, TranscriptInput(example.user_text))),
                target_tokens=tokens,
            )
        )
        if len(samples) < wrapper.config.semantic_examples:
            generated_tokens += len(teacher.response.token_ids)
            samples.append(
                SampleGeneration(
                    example_id=example.example_id,
                    dialogue_id=example.dialogue_id,
                    condition=EvaluationCondition.TEXT,
                    history=example.history,
                    user_transcript=example.user_text,
                    gold_response=example.target_text,
                    generated_response=teacher.response.text,
                    generation=GenerationDetails(
                        kind=GenerationKind.COMPLETED,
                        token_ids=teacher.response.token_ids,
                    ),
                    generation=GenerationDetails(
                        kind=teacher.response.kind, token_ids=teacher.response.token_ids
                    ),
                    duration=example.duration,
                    semantic_similarity=1.0,
                )
            )
    cross_entropy, perplexity, target_tokens = summarize_losses(
        tuple(
            LossObservation(item.cross_entropy * item.target_tokens, item.target_tokens)
            for item in losses
        )
    )
    return EvaluationOutcome(
        metrics=EvaluationMetrics(
            examples=len(examples),
            target_tokens=target_tokens,
            cross_entropy=cross_entropy,
            perplexity=perplexity,
            semantic_similarity=1.0,
            generated_examples=len(samples),
            generated_tokens=generated_tokens,
            completed_generations=len(samples),
            token_limited_generations=0,
            generation_seconds=0.0,
            evaluation_seconds=time.perf_counter() - started,
        ),
        samples=tuple(samples),
        example_losses=tuple(losses),
        diagnostics=(),
    )


def run_teacher_baselines(
    wrapper: FrozenQwen,
    validation: Sequence[Example],
    test: Sequence[Example],
    config: TeacherBaselineConfig,
    semantic_evaluator: SemanticEvaluator,
) -> None:
    teacher = TeacherProvenance.model_validate_json(
        (config.teacher_directory / "provenance.json").read_bytes()
    )
    teacher_run = teacher.config.run
    current = wrapper.config
    if (
        teacher_run.model_name,
        teacher_run.prompt,
        teacher_run.history_turns,
        teacher_run.max_history_tokens,
    ) != (current.model_name, current.prompt, current.history_turns, current.max_history_tokens):
        raise ValueError("Cached teacher uses different model, prompt or history settings")
    journal_path = config.teacher_directory / "targets.jsonl"
    targets = read_journal(journal_path, TeacherTarget)
    target_artifact = FileArtifact(
        path=journal_path,
        source_path=journal_path,
        bytes=journal_path.stat().st_size,
        sha256=file_digest(journal_path),
    )
    asr = load_asr_transcripts(config.manifest.parent / "asr_transcripts.jsonl")
    for split, examples in ((Split.VALIDATION, validation), (Split.TEST, test)):
        directory = config.output_root / "baseline" / split.value
        directory.mkdir(parents=True, exist_ok=True)
        provenance = TeacherBaselineProvenance(
            evaluation_config=current,
            teacher=teacher,
            target_journal=target_artifact,
            examples=tuple(examples),
        )
        provenance_path = directory / "cached_teacher_provenance.json"
        if provenance_path.exists():
            if (
                TeacherBaselineProvenance.model_validate_json(provenance_path.read_bytes())
                != provenance
            ):
                raise ValueError(
                    "Saved teacher baseline belongs to different cached responses/inputs"
                )
        else:
            if (directory / "text.json").exists():
                raise ValueError("Existing text baseline lacks cached-teacher provenance")
            provenance_path.write_text(
                provenance.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
        if not (directory / "text_qualitative.md").exists():
            save_evaluation(evaluate_cached_teacher(wrapper, examples, targets), directory, "text")
        if not (directory / "asr_qualitative.md").exists():
            asr_outcome = evaluate(
                wrapper,
                None,
                examples,
                current,
                EvaluationCondition.ASR,
                asr_transcripts=asr,
                semantic_evaluator=semantic_evaluator,
            )
            save_evaluation(asr_outcome, directory, "asr")
        (directory / "baseline_interpretation.md").write_text(
            "# Transcript reference: actual cached Qwen teacher responses\n\n"
            "TEXT generations reuse the actual completed transcript-conditioned Qwen outputs "
            "that define the distillation targets. Semantic self-cosine 1 is an identity by "
            "construction, not a quality score. TEXT CE is freshly computed under that saved "
            "target prefix. BF16 batch/unpadded near ties can change free-running trajectories "
            "and transcript target accuracy; no regenerated trajectory substitutes for the "
            "cached reference. ASR is generated fresh with the same prompt/history. The "
            "independent judge assesses teacher response quality instead of assigning it "
            "automatic perfection. Provenance retains journal SHA and original teacher model "
            "revision/config.\n",
            encoding="utf-8",
        )
