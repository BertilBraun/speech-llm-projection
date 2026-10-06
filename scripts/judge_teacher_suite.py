"""Calibrated, resumable full-coverage judging after the training process exits."""

import argparse
import gc
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import Field, TypeAdapter

from scripts.challenge_judge import JudgeChallengeReport, run_challenge
from scripts.compare_judgments import JudgmentComparison, compare_journals
from scripts.package_results import FileArtifact, file_digest
from scripts.run_judge import requests_from_generations
from speech_projector.judge import (
    JudgeConfig,
    JudgeJournalProvenance,
    JudgeRequest,
    JudgeSummary,
    LocalJudge,
    evaluate_judge,
    rubric_digest,
    summarize_judge,
)
from speech_projector.models import (
    EvaluationCondition,
    GenerationKind,
    Record,
    RunResult,
    SampleGeneration,
    Split,
    SuiteState,
)


class TeacherJudgingConfig(Record):
    results_root: Path
    output_directory: Path
    calibration_root: Path
    candidates: tuple[JudgeConfig, ...] = Field(min_length=1)


class JudgeSelection(Record):
    attempts: tuple[JudgeChallengeReport, ...]
    selected: JudgeConfig


@dataclass(frozen=True)
class CalibratedJudge:
    judge: LocalJudge
    selection: JudgeSelection


@dataclass(frozen=True)
class GenerationJob:
    name: str
    split: Split
    generations: Path
    expected_examples: int


class JudgmentInputs(Record):
    generations: FileArtifact
    judge: JudgeJournalProvenance


class GenerationCapSummary(Record):
    completed: int
    capped: int
    unknown: int


class JudgedGenerationSet(Record):
    name: str
    split: Split
    inputs: JudgmentInputs
    journal: FileArtifact
    summary: JudgeSummary
    generation_caps: GenerationCapSummary
    seconds: float


class NamedJudgmentComparison(Record):
    left: str
    right: str
    split: Split
    comparison: JudgmentComparison


class TeacherJudgingReport(Record):
    config: TeacherJudgingConfig
    selection: JudgeSelection
    sets: tuple[JudgedGenerationSet, ...]
    comparisons: tuple[NamedJudgmentComparison, ...]
    seconds: float
    peak_vram_gb: float


def artifact(path: Path) -> FileArtifact:
    return FileArtifact(
        path=path, source_path=path, bytes=path.stat().st_size, sha256=file_digest(path)
    )


def select_judge(
    config: TeacherJudgingConfig,
    load: Callable[[JudgeConfig], LocalJudge],
    challenge_model: Callable[[LocalJudge, Path], JudgeChallengeReport] = run_challenge,
) -> CalibratedJudge:
    attempts: list[JudgeChallengeReport] = []
    for index, candidate in enumerate(config.candidates):
        judge = load(candidate)
        directory = config.calibration_root / f"candidate_only_{index}"
        summary = challenge_model(judge, directory)
        attempts.append(summary)
        print(summary.model_dump_json(), flush=True)
        if summary.passed:
            selection = JudgeSelection(attempts=tuple(attempts), selected=candidate)
            config.output_directory.mkdir(parents=True, exist_ok=True)
            (config.output_directory / "judge_selection.json").write_text(
                selection.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            return CalibratedJudge(judge, selection)
        del judge
        gc.collect()
        torch.cuda.empty_cache()
    config.output_directory.mkdir(parents=True, exist_ok=True)
    (config.output_directory / "failed_calibrations.json").write_bytes(
        TypeAdapter(tuple[JudgeChallengeReport, ...]).dump_json(tuple(attempts), indent=2)
    )
    raise ValueError("All candidate-only judges failed calibration; no quality scores produced")


def generation_jobs(root: Path) -> tuple[GenerationJob, ...]:
    state = SuiteState.model_validate_json((root / "suite_state.json").read_bytes())
    if state.running is not None or state.failed:
        raise ValueError("Judge the suite only after all training/evaluation writers have exited")
    results = TypeAdapter(tuple[RunResult, ...]).validate_json(
        (root / "completed_results.json").read_bytes()
    )
    if len({item.config.name for item in results}) != len(results):
        raise ValueError("Completed suite repeats a run name")
    jobs = [
        GenerationJob(
            name=result.config.name,
            split=split,
            generations=root / result.config.name / split.value / "evaluation_generations.jsonl",
            expected_examples=result.config.semantic_examples,
        )
        for result in results
        for split in (Split.VALIDATION, Split.TEST)
    ]
    jobs.extend(
        GenerationJob(
            name=name,
            split=split,
            generations=root / "baseline" / split.value / f"{name}_generations.jsonl",
            expected_examples=512,
        )
        for name in ("text", "asr")
        for split in (Split.VALIDATION, Split.TEST)
    )
    for job in jobs:
        if not job.generations.is_file():
            raise ValueError(f"Completed generations missing: {job.generations}")
    return tuple(jobs)


def cache_requests(
    job: GenerationJob, config: TeacherJudgingConfig, judge: LocalJudge
) -> tuple[tuple[JudgeRequest, ...], JudgmentInputs, Path]:
    requests = requests_from_generations(job.generations)
    if len(requests) != job.expected_examples:
        raise ValueError(f"{job.name}/{job.split.value} has incomplete generation coverage")
    directory = config.output_directory / job.name / job.split.value
    directory.mkdir(parents=True, exist_ok=True)
    inputs = JudgmentInputs(
        generations=artifact(job.generations),
        judge=JudgeJournalProvenance(config=judge.config, rubric_sha256=rubric_digest()),
    )
    provenance_path = directory / "inputs.json"
    requests_path = directory / "requests.jsonl"
    if provenance_path.exists():
        if JudgmentInputs.model_validate_json(provenance_path.read_bytes()) != inputs:
            raise ValueError("Judging resume model/rubric/generation SHA changed")
    else:
        provenance_path.write_text(inputs.model_dump_json(indent=2) + "\n", encoding="utf-8")
    expected = "".join(item.model_dump_json() + "\n" for item in requests)
    if requests_path.exists():
        if requests_path.read_text(encoding="utf-8") != expected:
            raise ValueError("Cached blinded judging requests changed")
    else:
        pending = requests_path.with_suffix(".pending")
        pending.write_text(expected, encoding="utf-8")
        pending.replace(requests_path)
    return requests, inputs, directory


def generation_caps(path: Path) -> GenerationCapSummary:
    completed, capped, unknown = 0, 0, 0
    for line in path.read_bytes().splitlines():
        sample = SampleGeneration.model_validate_json(line)
        if sample.condition not in (
            EvaluationCondition.SPEECH,
            EvaluationCondition.TEXT,
            EvaluationCondition.ASR,
        ):
            continue
        if sample.generation is None:
            unknown += 1
        else:
            match sample.generation.kind:
                case GenerationKind.COMPLETED:
                    completed += 1
                case GenerationKind.TOKEN_LIMIT:
                    capped += 1
    return GenerationCapSummary(completed=completed, capped=capped, unknown=unknown)


def judge_generation_set(
    job: GenerationJob, config: TeacherJudgingConfig, judge: LocalJudge
) -> JudgedGenerationSet:
    requests, inputs, directory = cache_requests(job, config, judge)
    summary_path = directory / "summary.json"
    if summary_path.exists():
        previous = JudgedGenerationSet.model_validate_json(summary_path.read_bytes())
        if previous.inputs != inputs or artifact(previous.journal.path) != previous.journal:
            raise ValueError("Completed judgment summary/provenance changed")
        return previous
    started = time.perf_counter()
    path = directory / "judgments.jsonl"
    outcomes = evaluate_judge(requests, judge, path)
    summary = JudgedGenerationSet(
        name=job.name,
        split=job.split,
        inputs=inputs,
        journal=artifact(path),
        summary=summarize_judge(outcomes),
        generation_caps=generation_caps(job.generations),
        seconds=time.perf_counter() - started,
    )
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(summary.model_dump_json(), flush=True)
    return summary


def compare_sets(
    sets: Sequence[JudgedGenerationSet], speech_reference: str
) -> tuple[NamedJudgmentComparison, ...]:
    comparisons: list[NamedJudgmentComparison] = []
    for split in (Split.VALIDATION, Split.TEST):
        for reference_name in ("text", speech_reference):
            reference = next(
                item for item in sets if item.name == reference_name and item.split == split
            )
            for item in sets:
                if item.split != split or item.name == reference_name:
                    continue
                comparisons.append(
                    NamedJudgmentComparison(
                        left=item.name,
                        right=reference_name,
                        split=split,
                        comparison=compare_journals(item.journal.path, reference.journal.path),
                    )
                )
    return tuple(comparisons)


def render_report(report: TeacherJudgingReport) -> str:
    lines = [
        "# Independent candidate-only response quality",
        "",
        f"Judge: {report.selection.selected.model_name}, "
        f"revision {report.selection.selected.revision}. Source labels and teacher reference "
        "answers are excluded from the judging prompt. The actual cleaned user transcript "
        "and history are provided. Acceptance requires each rubric dimension to be >=2/3.",
        "",
        "The 13-case calibration and 30 independent cases are minimal gates, not a certification. "
        "A compact automated judge may miss subtle errors; inspect qualitative outputs. "
        "Failed JSON judgments are excluded from valid-rate and paired estimates; requested "
        "acceptance conservatively treats those cases as unaccepted. "
        "They are not silently dropped.",
        "",
        "| System | Split | Valid/requested | Acceptable requested | Relevance | "
        "Grounding/detail | Naturalness | Complete/capped/unknown | Attempt seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report.sets:
        summary = item.summary
        lines.append(
            f"| {item.name} | {item.split.value} | {summary.valid_examples}/"
            f"{summary.requested_examples} | {summary.acceptable_rate_requested:.3f} | "
            f"{summary.relevance:.3f} | {summary.grounded_detail:.3f} | "
            f"{summary.naturalness:.3f} | {item.generation_caps.completed}/"
            f"{item.generation_caps.capped}/{item.generation_caps.unknown} | {item.seconds:.1f} |"
        )
    lines.extend(
        [
            "",
            "Paired differences are left minus right reference, with "
            "95% percentile confidence intervals from 2,000 dialogue-cluster bootstrap draws. "
            "References are actual cached transcript teacher and the speech run selected by "
            "lowest validation CE among full512-evaluated runs. They quantify heldout sampling "
            "uncertainty, not judge bias or training-seed uncertainty. Unknown completion means "
            "the historical artifact did not retain EOS/cap metadata; "
            "it is not inferred from text.",
            "",
            "| System | Reference | Split | Metric | Paired valid | Difference [95% CI] |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for item in report.comparisons:
        for metric in item.comparison.metrics:
            interval = metric.left_minus_right
            lines.append(
                f"| {item.left} | {item.right} | {item.split.value} | {metric.metric.value} | "
                f"{interval.examples} | {interval.estimate:+.3f} "
                f"[{interval.lower:+.3f}, {interval.upper:+.3f}] |"
            )
    return "\n".join(lines) + "\n"


def run_judging(config: TeacherJudgingConfig, calibrated: CalibratedJudge) -> TeacherJudgingReport:
    started = time.perf_counter()
    jobs = generation_jobs(config.results_root)
    sets = tuple(judge_generation_set(job, config, calibrated.judge) for job in jobs)
    results = TypeAdapter(tuple[RunResult, ...]).validate_json(
        (config.results_root / "completed_results.json").read_bytes()
    )
    comparable = tuple(item for item in results if item.config.validation_examples == 512)
    if not comparable:
        raise ValueError("Quality comparison requires a speech run evaluated on all512 cases")
    speech_reference = min(comparable, key=lambda item: item.validation.cross_entropy)
    report = TeacherJudgingReport(
        config=config,
        selection=calibrated.selection,
        sets=sets,
        comparisons=compare_sets(sets, speech_reference.config.name),
        seconds=time.perf_counter() - started,
        peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
    )
    (config.output_directory / "quality_report.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (config.output_directory / "quality_report.md").write_text(
        render_report(report), encoding="utf-8"
    )
    return report


def load_cuda_judge(config: JudgeConfig) -> LocalJudge:
    return LocalJudge(config, torch.device("cuda"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--calibrate-only", action="store_true")
    arguments = parser.parse_args()
    config = TeacherJudgingConfig(
        results_root=arguments.results_root,
        output_directory=arguments.output,
        calibration_root=arguments.calibration_root,
        candidates=(
            JudgeConfig(batch_size=arguments.batch_size),
            JudgeConfig(
                model_name="Qwen/Qwen3-4B",
                revision="1cfa9a7208912126459214e8b04321603b3df60c",
                batch_size=arguments.batch_size,
            ),
            JudgeConfig(
                model_name="Qwen/Qwen3-4B-Instruct-2507",
                revision="cdbee75f17c01a7cc42f958dc650907174af0554",
                batch_size=arguments.batch_size,
            ),
        ),
    )
    calibrated = select_judge(config, load_cuda_judge)
    if not arguments.calibrate_only:
        run_judging(config, calibrated)


if __name__ == "__main__":
    main()
