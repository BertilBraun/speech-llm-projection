"""Calibrate and run the final-only blinded intended-tone response judge."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import torch
from pydantic import TypeAdapter

from speech_projector.journal import read_journal
from speech_projector.judge import JudgeConfig, JudgeRequest, LocalJudge
from speech_projector.overnight_judge import (
    FinalJudgingProgram,
    FinalJudgingSet,
    NeuToneJudgeRequest,
    ResponseChoice,
    ToneCalibrationCase,
    ToneCalibrationResult,
    ToneJudgeRequest,
    ToneJudgeSuccess,
    TonePairJudgeRequest,
    TonePairSuccess,
    run_final_judging_set,
    run_pair_judgments,
    run_tone_judgments,
    summarize_pair_judgments,
    summarize_tone_judgments,
)
from speech_projector.tts_pilot import PilotEmotion


def calibration_cases() -> tuple[ToneCalibrationCase, ...]:
    anchors = (
        (
            PilotEmotion.HAPPY,
            "The decision came through this morning.",
            "It sounds like you're pleased about it; I'm glad for you.",
            "Stop being so miserable. You should be ashamed of feeling this way.",
        ),
        (
            PilotEmotion.SAD,
            "The room is quiet today.",
            "It sounds like the quiet feels heavy. I'm here to listen if you'd like.",
            "You must be ecstatic! Let's celebrate this amazing achievement.",
        ),
        (
            PilotEmotion.FEARFUL,
            "My turn to speak is next.",
            "That sounds nerve-racking. One slow breath may help before you begin.",
            "Stop worrying and get over it; there's nothing to be afraid of.",
        ),
        (
            PilotEmotion.ANGRY,
            "They changed the deadline again.",
            "I hear how irritating that is. You could ask them for a clear, stable deadline.",
            "Just smile and be grateful they bothered to tell you.",
        ),
        (
            PilotEmotion.SAD,
            "The letter arrived today.",
            "It sounds like this landed heavily. We can take a moment before talking about it.",
            "What fantastic news! Forget your sadness; let's party.",
        ),
        (
            PilotEmotion.FEARFUL,
            "The door opened behind me.",
            "That sounds unsettling. Take a moment to check your surroundings if you need to.",
            "Ha, that's hilarious! You should be enjoying this instead of feeling scared.",
        ),
    )
    return tuple(
        ToneCalibrationCase(
            request=NeuToneJudgeRequest(
                utterance=JudgeRequest(
                    example_id=f"tone_calibration_{index}_{variant}",
                    dialogue_id=f"tone_calibration_family_{index}",
                    history=(),
                    transcript=transcript,
                    candidate_response=response,
                ),
                intended_tone=tone,
            ),
            expected_acceptable=acceptable,
        )
        for index, (tone, transcript, good, bad) in enumerate(anchors)
        for variant, response, acceptable in (("good", good, True), ("bad", bad, False))
    )


def calibrate(judge: LocalJudge, directory: Path) -> ToneCalibrationResult:
    cases = calibration_cases()
    case_bytes = ("\n".join(item.model_dump_json() for item in cases) + "\n").encode()
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    saved_result = (
        ToneCalibrationResult.model_validate_json(result_path.read_bytes())
        if result_path.exists()
        else None
    )
    path = directory / "cases.jsonl"
    if path.exists() and path.read_bytes() != case_bytes:
        raise ValueError("Calibration directory contains different cases")
    path.write_bytes(case_bytes)
    if judge.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(judge.device)
    started = time.perf_counter()
    outcomes = run_tone_judgments(
        judge, tuple(item.request for item in cases), directory / "single"
    )
    valid = tuple(item for item in outcomes if isinstance(item, ToneJudgeSuccess))
    expected = {item.request.utterance.example_id: item.expected_acceptable for item in cases}
    correct = sum(
        item.verdict.acceptable == expected[item.request.utterance.example_id] for item in valid
    )
    pairs: list[TonePairJudgeRequest] = []
    for index in range(0, len(cases), 2):
        good, bad = cases[index], cases[index + 1]
        swap = index // 2 % 2 == 1
        pairs.append(
            TonePairJudgeRequest(
                context=bad.request if swap else good.request,
                alternative_response=good.request.utterance.candidate_response
                if swap
                else bad.request.utterance.candidate_response,
                matching_response=ResponseChoice.B if swap else ResponseChoice.A,
            )
        )
    pair_outcomes = run_pair_judgments(judge, pairs, directory / "paired")
    pair_correct = sum(
        item.verdict.choice == item.request.matching_response
        for item in pair_outcomes
        if isinstance(item, TonePairSuccess)
    )
    result = ToneCalibrationResult(
        configuration=judge.config,
        cases_sha256=hashlib.sha256(case_bytes).hexdigest(),
        requested=len(cases),
        valid=len(valid),
        correct_acceptability=correct,
        preference_requests=len(pairs),
        preference_correct=pair_correct,
        passed=correct == len(cases) and pair_correct == len(pairs),
        runtime_seconds=time.perf_counter() - started,
        peak_pytorch_allocated_decimal_gb=torch.cuda.max_memory_allocated(judge.device) / 1e9
        if judge.device.type == "cuda"
        else 0,
    )
    if saved_result is not None:
        restored_clocks = result.model_copy(
            update={
                "runtime_seconds": saved_result.runtime_seconds,
                "peak_pytorch_allocated_decimal_gb": saved_result.peak_pytorch_allocated_decimal_gb,
            }
        )
        if restored_clocks != saved_result:
            raise ValueError(
                "Saved calibration summary differs from recomputed judgments or inputs"
            )
        return saved_result
    pending_path = result_path.with_suffix(".json.part")
    pending_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    pending_path.replace(result_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibrate-only", action="store_true")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--requests", type=Path)
    inputs.add_argument("--request-set", type=Path)
    inputs.add_argument("--program", type=Path)
    parser.add_argument("--pair-requests", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    arguments = parser.parse_args()
    configuration = JudgeConfig(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        revision="cdbee75f17c01a7cc42f958dc650907174af0554",
        batch_size=arguments.batch_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    judge = LocalJudge(configuration, device)
    calibration = calibrate(judge, arguments.calibration)
    print(calibration.model_dump_json())
    if arguments.calibrate_only:
        return
    if not calibration.passed:
        raise ValueError("Tone judge calibration failed; quality rates are not reliable")
    if arguments.program is not None:
        program = FinalJudgingProgram.model_validate_json(arguments.program.read_bytes())
        for requests in program.sets:
            summary = run_final_judging_set(
                judge, requests, arguments.output / requests.condition.value
            )
            print(summary.model_dump_json(), flush=True)
        return
    if arguments.request_set is not None:
        requests = FinalJudgingSet.model_validate_json(arguments.request_set.read_bytes())
        summary = run_final_judging_set(judge, requests, arguments.output)
        print(summary.model_dump_json())
        return
    if arguments.requests is None:
        raise ValueError("Final judging requires --requests or --calibrate-only")
    requests = read_journal(arguments.requests, TypeAdapter(ToneJudgeRequest))
    outcomes = run_tone_judgments(judge, requests, arguments.output / "single")
    summary = summarize_tone_judgments(outcomes)
    (arguments.output / "summary.json").write_text(
        summary.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    if arguments.pair_requests is not None:
        pairs = read_journal(arguments.pair_requests, TonePairJudgeRequest)
        judged = run_pair_judgments(judge, pairs, arguments.output / "paired")
        preference = summarize_pair_judgments(judged)
        (arguments.output / "pair_summary.json").write_text(
            preference.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
