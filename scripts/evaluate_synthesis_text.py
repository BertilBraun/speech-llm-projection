"""Supplementary documented-synthesis-text reference with unchanged examples and targets."""

import argparse
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from scripts.audit_synthesis_alignment import (
    SynthesisAlignment,
    SynthesisAlignmentAudit,
    load_alignments,
)
from scripts.package_results import FileArtifact, ModelRevision, file_digest, write_record
from speech_projector.data import load_examples
from speech_projector.evaluation import SemanticEvaluator, evaluate, save_evaluation
from speech_projector.inputs import TranscriptInput
from speech_projector.llm import FrozenQwen
from speech_projector.models import (
    EvaluationCondition,
    EvaluationMetrics,
    Example,
    Record,
    RunConfig,
    Split,
)
from speech_projector.training import weights_digest


@dataclass(frozen=True)
class SynthesisTextInputs:
    alignments: tuple[SynthesisAlignment, ...]

    def __call__(self, example: Example) -> TranscriptInput:
        for alignment in self.alignments:
            if alignment.example.example_id == example.example_id:
                if alignment.example != example:
                    raise ValueError("Evaluation example differs from its audited canonical record")
                return TranscriptInput(alignment.cleaned_text)
        raise ValueError(f"Missing synthesis alignment: {example.example_id}")


class SynthesisTextEvaluation(Record):
    config: RunConfig
    evaluation_git_commit: str
    input_artifacts: tuple[FileArtifact, ...]
    model_revisions: tuple[ModelRevision, ...]
    frozen_llm_weights_sha256: str
    validation: EvaluationMetrics
    test: EvaluationMetrics
    evaluation_seconds: float
    peak_vram_gb: float


def input_artifact(path: Path) -> FileArtifact:
    return FileArtifact(
        path=path, source_path=path, bytes=path.stat().st_size, sha256=file_digest(path)
    )


def verified_examples(
    manifest: Path, inputs: SynthesisTextInputs, split: Split, count: int
) -> tuple[Example, ...]:
    examples = tuple(load_examples(manifest, split, count))
    alignments = tuple(item for item in inputs.alignments if item.example.split == split)
    if tuple(item.split_index for item in alignments) != tuple(range(count)):
        raise ValueError("Synthesis audit must cover the complete ordered fixed evaluation split")
    if tuple(item.example for item in alignments) != examples:
        raise ValueError("Synthesis audit does not match the unchanged manifest examples")
    return examples


def evaluate_synthesis_text(
    configuration_path: Path,
    manifest: Path,
    audit_directory: Path,
    model_revisions_path: Path,
    output_directory: Path,
    device: torch.device,
) -> SynthesisTextEvaluation:
    configuration = RunConfig.model_validate_json(configuration_path.read_bytes())
    audit_path = audit_directory / "synthesis_alignment_audit.json"
    alignments_path = audit_directory / "synthesis_alignment_heldout.jsonl"
    audit = SynthesisAlignmentAudit.model_validate_json(audit_path.read_bytes())
    if file_digest(manifest) != audit.manifest_sha256:
        raise ValueError("Synthesis audit manifest SHA does not match the selected manifest")
    inputs = SynthesisTextInputs(load_alignments(alignments_path))
    validation_examples = verified_examples(
        manifest, inputs, Split.VALIDATION, configuration.validation_examples
    )
    test_examples = verified_examples(manifest, inputs, Split.TEST, configuration.test_examples)
    revisions = tuple(
        ModelRevision.model_validate_json(line)
        for line in model_revisions_path.read_bytes().splitlines()
    )
    if configuration.model_name not in {item.model_name for item in revisions}:
        raise ValueError("Frozen Qwen revision is absent from the model revision inventory")
    artifacts = tuple(
        input_artifact(path)
        for path in (
            configuration_path,
            manifest,
            audit_path,
            alignments_path,
            model_revisions_path,
        )
    )
    summary_path = output_directory / "summary.json"
    if summary_path.exists():
        saved = SynthesisTextEvaluation.model_validate_json(summary_path.read_bytes())
        if (
            saved.config != configuration
            or saved.input_artifacts != artifacts
            or saved.model_revisions != revisions
        ):
            raise ValueError("Saved supplementary synthesis evaluation does not match its inputs")
        return saved
    started = time.monotonic()
    wrapper = FrozenQwen(configuration, device)
    frozen_digest = weights_digest(wrapper.model)
    semantic_evaluator = SemanticEvaluator()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    validation = evaluate(
        wrapper,
        None,
        validation_examples,
        configuration,
        EvaluationCondition.TEXT,
        semantic_evaluator=semantic_evaluator,
        text_input=inputs,
    )
    save_evaluation(validation, output_directory / "validation")
    tested = evaluate(
        wrapper,
        None,
        test_examples,
        configuration,
        EvaluationCondition.TEXT,
        semantic_evaluator=semantic_evaluator,
        text_input=inputs,
    )
    save_evaluation(tested, output_directory / "test")
    summary = SynthesisTextEvaluation(
        config=configuration,
        evaluation_git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        input_artifacts=artifacts,
        model_revisions=revisions,
        frozen_llm_weights_sha256=frozen_digest,
        validation=validation.metrics,
        test=tested.metrics,
        evaluation_seconds=time.monotonic() - started,
        peak_vram_gb=torch.cuda.max_memory_allocated(device) / 1e9
        if device.type == "cuda"
        else 0.0,
    )
    write_record(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-directory", type=Path, required=True)
    parser.add_argument("--model-revisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    summary = evaluate_synthesis_text(
        arguments.config,
        arguments.manifest,
        arguments.audit_directory,
        arguments.model_revisions,
        arguments.output,
        torch.device(arguments.device),
    )
    print(summary.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
