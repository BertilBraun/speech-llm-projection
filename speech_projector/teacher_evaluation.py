"""Teacher-prefix fidelity, separate from free-running response quality."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional

from speech_projector.evaluation import match_feature_length
from speech_projector.inputs import SpeechInput, TranscriptInput
from speech_projector.journal import append_record, read_journal
from speech_projector.models import EvaluationCondition, Example, Record, RunConfig

if TYPE_CHECKING:
    from speech_projector.llm import FrozenQwen, TargetScores
    from speech_projector.projectors import Projector


class TeacherFidelity(Record):
    example_id: str
    dialogue_id: str
    condition: EvaluationCondition
    target_tokens: int
    input_cross_entropy: float
    transcript_cross_entropy: float
    excess_cross_entropy: float
    top1_agreement: float
    input_target_accuracy: float
    transcript_target_accuracy: float
    teacher_to_input_kl: float
    first_token_agreement: float
    first_token_target_accuracy: float
    first_8_tokens: int
    first_8_token_agreement: float
    first_8_token_target_accuracy: float


class FidelitySummary(Record):
    condition: EvaluationCondition
    examples: int
    target_tokens: int
    input_cross_entropy: float
    transcript_cross_entropy: float
    excess_cross_entropy: float
    top1_agreement: float
    input_target_accuracy: float
    transcript_target_accuracy: float
    teacher_to_input_kl: float
    first_token_agreement: float
    first_token_target_accuracy: float
    first_8_tokens: int
    first_8_token_agreement: float
    first_8_token_target_accuracy: float


class FidelityProvenance(Record):
    config: RunConfig
    projector_weights_sha256: str
    examples_sha256: str


def projector_digest(projector: Projector) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(projector.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\x00")
        digest.update(str(parameter.dtype).encode("ascii") + b"\x00")
        digest.update(str(tuple(parameter.shape)).encode("ascii") + b"\x00")
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def exact_teacher_kl(teacher_logits: Tensor, input_logits: Tensor, chunk_tokens: int = 16) -> float:
    """KL(teacher || input) normalized per token, without persistent vocabulary tensors."""
    if chunk_tokens < 1:
        raise ValueError("KL chunk size must be positive")
    assert teacher_logits.shape == input_logits.shape
    total = torch.zeros((), device=teacher_logits.device, dtype=torch.float64)
    for offset in range(0, teacher_logits.shape[0], chunk_tokens):
        teacher_log = functional.log_softmax(
            teacher_logits[offset : offset + chunk_tokens].float(), -1
        )
        input_log = functional.log_softmax(input_logits[offset : offset + chunk_tokens].float(), -1)
        total += (teacher_log.exp() * (teacher_log - input_log)).double().sum()
    return float(total / teacher_logits.shape[0])


def compare_target_scores(
    example: Example,
    condition: EvaluationCondition,
    input_scores: TargetScores,
    transcript_scores: TargetScores,
) -> TeacherFidelity:
    if not torch.equal(input_scores.target_token_ids, transcript_scores.target_token_ids):
        raise ValueError("Fidelity comparisons require identical teacher-forced target prefixes")
    targets = input_scores.target_token_ids
    assert targets.ndim == 1 and targets.numel() > 0
    assert input_scores.logits.shape == transcript_scores.logits.shape
    assert input_scores.logits.shape[0] == targets.numel()
    input_ce = float(functional.cross_entropy(input_scores.logits.float(), targets))
    transcript_ce = float(functional.cross_entropy(transcript_scores.logits.float(), targets))
    input_top1 = input_scores.logits.argmax(-1)
    transcript_top1 = transcript_scores.logits.argmax(-1)
    return TeacherFidelity(
        example_id=example.example_id,
        dialogue_id=example.dialogue_id,
        condition=condition,
        target_tokens=targets.numel(),
        input_cross_entropy=input_ce,
        transcript_cross_entropy=transcript_ce,
        excess_cross_entropy=input_ce - transcript_ce,
        top1_agreement=float((input_top1 == transcript_top1).float().mean()),
        input_target_accuracy=float((input_top1 == targets).float().mean()),
        transcript_target_accuracy=float((transcript_top1 == targets).float().mean()),
        teacher_to_input_kl=exact_teacher_kl(transcript_scores.logits, input_scores.logits),
        first_token_agreement=float(input_top1[0] == transcript_top1[0]),
        first_token_target_accuracy=float(input_top1[0] == targets[0]),
        first_8_tokens=min(8, targets.numel()),
        first_8_token_agreement=float((input_top1[:8] == transcript_top1[:8]).float().mean()),
        first_8_token_target_accuracy=float((input_top1[:8] == targets[:8]).float().mean()),
    )


@torch.no_grad()
def evaluate_teacher_fidelity(
    wrapper: FrozenQwen,
    projector: Projector,
    examples: Sequence[Example],
    output_directory: Path,
) -> tuple[TeacherFidelity, ...]:
    if not examples:
        raise ValueError("Teacher fidelity evaluation requires examples")
    if wrapper.config.conditioning_examples and len({item.dialogue_id for item in examples}) < 2:
        raise ValueError("Wrong-audio controls require at least two different dialogues")
    wrapper.model.eval()
    projector.eval()
    output_directory.mkdir(parents=True, exist_ok=True)
    journal = output_directory / "teacher_fidelity.jsonl"
    provenance_path = output_directory / "teacher_fidelity_provenance.json"
    provenance = FidelityProvenance(
        config=wrapper.config,
        projector_weights_sha256=projector_digest(projector),
        examples_sha256=hashlib.sha256(
            "".join(item.model_dump_json() + "\n" for item in examples).encode("utf-8")
        ).hexdigest(),
    )
    if provenance_path.exists():
        if FidelityProvenance.model_validate_json(provenance_path.read_bytes()) != provenance:
            raise ValueError(
                "Fidelity journal differs in projector weights, config or teacher examples"
            )
    elif journal.exists():
        raise ValueError("Existing fidelity journal has no matching provenance")
    else:
        provenance_path.write_text(provenance.model_dump_json(indent=2) + "\n", encoding="utf-8")
    observations = list(read_journal(journal, TeacherFidelity))
    completed = {(item.example_id, item.condition) for item in observations}
    if len(completed) != len(observations):
        raise ValueError("Fidelity journal contains duplicate example/condition records")
    example_ids = {item.example_id for item in examples}
    if any(item.example_id not in example_ids for item in observations):
        raise ValueError("Fidelity journal contains examples outside the selected subset")
    for index, example in enumerate(examples):
        conditions = (
            (
                EvaluationCondition.SPEECH,
                EvaluationCondition.SHUFFLED_SPEECH,
                EvaluationCondition.ZERO_SPEECH,
            )
            if index < wrapper.config.conditioning_examples
            else (EvaluationCondition.SPEECH,)
        )
        if all((example.example_id, condition) in completed for condition in conditions):
            continue
        features: Tensor = torch.load(example.feature_path, map_location="cpu", weights_only=True)
        embeddings = projector(features.to(wrapper.device))
        transcript = wrapper.score_target(example, TranscriptInput(example.user_text))
        for condition in conditions:
            if (example.example_id, condition) in completed:
                continue
            match condition:
                case EvaluationCondition.SPEECH:
                    current = embeddings
                case EvaluationCondition.ZERO_SPEECH:
                    current = torch.zeros_like(embeddings)
                case EvaluationCondition.SHUFFLED_SPEECH:
                    donor = next(
                        examples[(index + offset) % len(examples)]
                        for offset in range(1, len(examples))
                        if examples[(index + offset) % len(examples)].dialogue_id
                        != example.dialogue_id
                    )
                    wrong: Tensor = torch.load(
                        donor.feature_path, map_location="cpu", weights_only=True
                    )
                    current = projector(
                        match_feature_length(wrong.to(wrapper.device), features.shape[0])
                    )
                case _:
                    raise AssertionError("Unexpected fidelity condition")
            scores = wrapper.score_target(example, SpeechInput(current))
            observation = compare_target_scores(example, condition, scores, transcript)
            append_record(journal, observation)
            observations.append(observation)
    return tuple(observations)


def summarize_fidelity(
    observations: Sequence[TeacherFidelity],
    condition: EvaluationCondition = EvaluationCondition.SPEECH,
) -> FidelitySummary:
    selected = tuple(item for item in observations if item.condition == condition)
    if not selected:
        raise ValueError("Fidelity summary requires observations")
    observations = selected
    tokens = sum(item.target_tokens for item in observations)
    early_tokens = sum(item.first_8_tokens for item in observations)
    return FidelitySummary(
        condition=condition,
        examples=len(observations),
        target_tokens=tokens,
        input_cross_entropy=sum(
            item.input_cross_entropy * item.target_tokens for item in observations
        )
        / tokens,
        transcript_cross_entropy=sum(
            item.transcript_cross_entropy * item.target_tokens for item in observations
        )
        / tokens,
        excess_cross_entropy=sum(
            item.excess_cross_entropy * item.target_tokens for item in observations
        )
        / tokens,
        top1_agreement=sum(item.top1_agreement * item.target_tokens for item in observations)
        / tokens,
        input_target_accuracy=sum(
            item.input_target_accuracy * item.target_tokens for item in observations
        )
        / tokens,
        transcript_target_accuracy=sum(
            item.transcript_target_accuracy * item.target_tokens for item in observations
        )
        / tokens,
        teacher_to_input_kl=sum(
            item.teacher_to_input_kl * item.target_tokens for item in observations
        )
        / tokens,
        first_token_agreement=sum(item.first_token_agreement for item in observations)
        / len(observations),
        first_token_target_accuracy=sum(item.first_token_target_accuracy for item in observations)
        / len(observations),
        first_8_tokens=early_tokens,
        first_8_token_agreement=sum(
            item.first_8_token_agreement * item.first_8_tokens for item in observations
        )
        / early_tokens,
        first_8_token_target_accuracy=sum(
            item.first_8_token_target_accuracy * item.first_8_tokens for item in observations
        )
        / early_tokens,
    )


def save_teacher_fidelity(observations: Sequence[TeacherFidelity], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    summaries = tuple(
        summarize_fidelity(observations, condition)
        for condition in dict.fromkeys(item.condition for item in observations)
    )
    assert all(math.isfinite(item.teacher_to_input_kl) for item in summaries)
    partial = directory / "teacher_fidelity_summary.partial"
    partial.write_text(
        "".join(item.model_dump_json() + "\n" for item in summaries), encoding="utf-8"
    )
    partial.replace(directory / "teacher_fidelity_summary.jsonl")
