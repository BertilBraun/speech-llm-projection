"""Calibrate and run the independent local judge after the main model process exits."""

import argparse
import time
from enum import Enum
from pathlib import Path

import torch

from speech_projector.judge import (
    JudgeConfig,
    JudgeRequest,
    JudgeSuccess,
    LocalJudge,
    evaluate_judge,
    summarize_judge,
)
from speech_projector.models import EvaluationCondition, Record, SampleGeneration


class CalibrationCategory(str, Enum):
    PARAPHRASE = "paraphrase"
    UNRELATED = "unrelated"
    GENERIC = "generic"
    WRONG_ENTITY = "wrong_entity"
    CONVERSATIONAL = "conversational"


class CalibrationCase(Record):
    category: CalibrationCategory
    request: JudgeRequest


class CalibrationGroup(Record):
    category: CalibrationCategory
    examples: int
    valid_examples: int
    acceptable_rate: float


class CalibrationSummary(Record):
    config: JudgeConfig
    cases: tuple[CalibrationCase, ...]
    groups: tuple[CalibrationGroup, ...]
    passed: bool
    seconds: float
    decoded_response_tokens: int


def calibration_cases() -> tuple[CalibrationCase, ...]:
    topics = (
        (
            "Where is the Grand Canyon?",
            "The Grand Canyon is in northern Arizona, United States.",
            "You'll find the Grand Canyon in Arizona in the US.",
            "A piano has eighty-eight keys.",
            "Wow, that's an amazing place! I'm so excited to explore it with you.",
            "The Grand Canyon is in Wyoming, inside Yellowstone National Park.",
        ),
        (
            "Is the Ford Mustang Shelby GT500 an electric car?",
            "No. The Shelby GT500 is a gasoline-powered performance Mustang.",
            "It isn't electric: the Shelby GT500 uses a petrol engine.",
            "Water your houseplants only when they need it.",
            "That car sounds incredible! I love hearing about exciting vehicles.",
            "The Tesla Model3 is fully electric, so yes, that car is electric.",
        ),
        (
            "My succulent leaves are mushy. Should I water it more often?",
            "Mushy succulent leaves can indicate overwatering. "
            "Let the soil dry and check drainage.",
            "More water may make it worse. Allow the soil to dry and ensure the pot drains well.",
            "The NBA playoffs are exciting to follow.",
            "Plants can be such a fun challenge. I'm curious to hear how it goes!",
            "Keep your orchid's roots constantly submerged in water to solve this.",
        ),
    )
    cases: list[CalibrationCase] = []
    for index, (transcript, reference, *responses) in enumerate(topics):
        for category, candidate in zip(
            (
                CalibrationCategory.PARAPHRASE,
                CalibrationCategory.UNRELATED,
                CalibrationCategory.GENERIC,
                CalibrationCategory.WRONG_ENTITY,
            ),
            responses,
            strict=True,
        ):
            cases.append(
                CalibrationCase(
                    category=category,
                    request=JudgeRequest(
                        example_id=f"calibration-{index}-{category.value}",
                        dialogue_id=f"calibration-dialogue-{index}",
                        history=(),
                        transcript=transcript,
                        reference_response=reference,
                        candidate_response=candidate,
                    ),
                )
            )
    cases.append(
        CalibrationCase(
            category=CalibrationCategory.CONVERSATIONAL,
            request=JudgeRequest(
                example_id="calibration-conversation-trip",
                dialogue_id="calibration-conversation",
                history=(),
                transcript="I'm so excited for my trip to Tokyo next week!",
                reference_response=(
                    "That sounds wonderful! I hope you have an amazing time in Tokyo."
                ),
                candidate_response="That sounds really exciting! Hope you have a wonderful trip.",
            ),
        )
    )
    return tuple(cases)


def calibrate(judge: LocalJudge, output: Path) -> CalibrationSummary:
    output.mkdir(parents=True, exist_ok=True)
    cases = calibration_cases()
    summary_path = output / "calibration_summary.json"
    if summary_path.exists():
        existing = CalibrationSummary.model_validate_json(summary_path.read_bytes())
        if existing.config != judge.config or existing.cases != cases:
            raise ValueError("Calibration output belongs to different judge/cases")
        return existing
    started = time.perf_counter()
    outcomes = evaluate_judge(
        tuple(item.request for item in cases), judge, output / "calibration.jsonl"
    )
    groups: list[CalibrationGroup] = []
    for category in CalibrationCategory:
        selected = tuple(
            outcome
            for case, outcome in zip(cases, outcomes, strict=True)
            if case.category == category
        )
        valid = tuple(item for item in selected if isinstance(item, JudgeSuccess))
        groups.append(
            CalibrationGroup(
                category=category,
                examples=len(selected),
                valid_examples=len(valid),
                acceptable_rate=sum(item.verdict.acceptable for item in valid) / len(selected),
            )
        )
    passed = all(
        item.valid_examples == item.examples
        and (
            item.acceptable_rate >= 0.75
            if item.category in (CalibrationCategory.PARAPHRASE, CalibrationCategory.CONVERSATIONAL)
            else item.acceptable_rate <= 0.25
        )
        for item in groups
    )
    summary = CalibrationSummary(
        config=judge.config,
        cases=cases,
        groups=tuple(groups),
        passed=passed,
        seconds=time.perf_counter() - started,
        decoded_response_tokens=sum(
            len(judge.tokenizer.encode(raw, add_special_tokens=False))
            for item in outcomes
            for raw in item.raw_responses
        ),
    )
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return summary


def requests_from_generations(path: Path) -> tuple[JudgeRequest, ...]:
    samples = tuple(
        SampleGeneration.model_validate_json(line) for line in path.read_bytes().splitlines()
    )
    primary = tuple(
        item
        for item in samples
        if item.condition
        in (EvaluationCondition.SPEECH, EvaluationCondition.TEXT, EvaluationCondition.ASR)
    )
    return tuple(
        JudgeRequest(
            example_id=item.example_id,
            dialogue_id=item.dialogue_id,
            history=item.history,
            transcript=item.user_transcript,
            reference_response=item.gold_response,
            candidate_response=item.generated_response,
        )
        for item in primary
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    parser.add_argument("--batch-size", type=int, default=16)
    arguments = parser.parse_args()
    if not arguments.calibrate_only and arguments.generations is None:
        parser.error("--generations is required unless --calibrate-only")
    config = JudgeConfig(
        model_name=arguments.model, revision=arguments.revision, batch_size=arguments.batch_size
    )
    judge = LocalJudge(config, torch.device(arguments.device))
    calibration = calibrate(judge, arguments.calibration)
    print(calibration.model_dump_json(indent=2), flush=True)
    if not calibration.passed:
        raise ValueError("Judge calibration failed; do not treat its full-run scores as reliable")
    if arguments.calibrate_only:
        return
    assert arguments.generations is not None
    requests = requests_from_generations(arguments.generations)
    started = time.perf_counter()
    outcomes = evaluate_judge(requests, judge, arguments.output / "judgments.jsonl")
    summary = summarize_judge(outcomes)
    (arguments.output / "judge_summary.json").write_text(
        summary.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (arguments.output / "judge_config.json").write_text(
        config.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    print(f"Judged {len(outcomes)} examples in {time.perf_counter() - started:.1f}s", flush=True)
    print(summary.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
