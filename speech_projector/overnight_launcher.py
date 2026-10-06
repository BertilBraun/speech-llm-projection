"""Sequential compression sweep, held-out selection, and bounded continuation."""

import argparse
import math
import time
import traceback
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, TypeAdapter
from safetensors.torch import load_file

from scripts.inventory_results import write_record
from speech_projector.cache import CacheConfig, extract_features
from speech_projector.evaluation import (
    ConditioningDiagnostic,
    EvaluationOutcome,
    ExampleLoss,
    SemanticEvaluator,
    evaluate,
    load_asr_transcripts,
    save_evaluation,
)
from speech_projector.launcher import git_revision, initialize_projector
from speech_projector.llm import FrozenQwen
from speech_projector.models import (
    AsrTranscript,
    EvaluationCondition,
    EvaluationMetrics,
    Example,
    ExperimentFailure,
    GradientCheck,
    Record,
    RunConfig,
    RunResult,
    SampleGeneration,
    Split,
    SuiteState,
)
from speech_projector.overnight_configuration import sweep_runs
from speech_projector.overnight_continuation import prepare_continuation
from speech_projector.overnight_data import Cohort, SourceSidecar
from speech_projector.overnight_evaluation import (
    SweepCandidate,
    SweepDecision,
    SweepSelectionPolicy,
    build_emotion_pairs,
    evaluate_emotion_pairs,
    select_sweep,
    summarize_cohort_validation,
    summarize_preferences,
)
from speech_projector.overnight_judge import FinalJudgingQuota
from speech_projector.overnight_preparation import load_records
from speech_projector.projectors import Projector
from speech_projector.teacher_evaluation import evaluate_teacher_fidelity
from speech_projector.training import (
    BatchedGradientCheck,
    BatchedParityPolicy,
    TrainingState,
    batched_gradient_sanity,
    gradient_sanity,
    train_run,
    validate_batched_parity,
)


class OvernightSuiteConfig(Record):
    data_root: Path
    output_root: Path
    microbatch_size: Literal[1, 2, 4] = 1
    deadline_unix_time: float = Field(gt=0)
    finalization_reserve_seconds: float = Field(default=3600, ge=0)
    continuation_epoch_ceiling: int = Field(default=3, ge=1, le=3)
    continuation_minimum_improvement: float = Field(default=0.01, ge=0)
    selection: SweepSelectionPolicy = SweepSelectionPolicy()
    batched_parity: BatchedParityPolicy = BatchedParityPolicy()


class ContinuationStop(Record):
    selected_sweep: str
    last_completed_run: str
    reason: str


class OvernightCompletion(Record):
    decision: SweepDecision
    final_runs: tuple[str, ...]
    continuation_stops: tuple[ContinuationStop, ...]
    completed_at: float


class FinalEvaluationSelection(Record):
    quota: FinalJudgingQuota
    examples: tuple[Example, ...]
    generation_example_ids: tuple[str, ...]


def select_final_evaluation(
    examples: tuple[Example, ...], sources: tuple[SourceSidecar, ...], quota: FinalJudgingQuota
) -> FinalEvaluationSelection:
    sources_by_id = {source.example_id: source for source in sources}
    ordinary = tuple(
        example
        for example in examples
        if sources_by_id[example.example_id].cohort == Cohort.ORDINARY
    )
    old_pairs = build_emotion_pairs(examples, sources, Cohort.QWEN_EMOTIONAL, Split.TEST)
    new_pairs = build_emotion_pairs(examples, sources, Cohort.NEU_EMOTIONAL, Split.TEST)
    if (
        len(ordinary) < quota.ordinary
        or len(old_pairs) < quota.old_emotional_pairs
        or len(new_pairs) < quota.new_emotional_pairs
    ):
        raise ValueError("Final test examples do not cover the precommitted judging quotas")
    selected = ordinary[: quota.ordinary] + tuple(
        example
        for pairs, count in (
            (old_pairs, quota.old_emotional_pairs),
            (new_pairs, quota.new_emotional_pairs),
        )
        for pair in pairs[:count]
        for example in (pair.first, pair.second)
    )
    identifiers = tuple(example.example_id for example in selected)
    selected_ids = set(identifiers)
    return FinalEvaluationSelection(
        quota=quota,
        examples=selected
        + tuple(example for example in examples if example.example_id not in selected_ids),
        generation_example_ids=identifiers,
    )


def final_evaluation_config(
    configuration: RunConfig, selection: FinalEvaluationSelection
) -> RunConfig:
    count = len(selection.generation_example_ids)
    return configuration.model_copy(
        update={"qualitative_examples": count, "semantic_examples": count}
    )


def load_sources(path: Path) -> tuple[SourceSidecar, ...]:
    adapter = TypeAdapter(SourceSidecar)
    return tuple(adapter.validate_json(line) for line in path.read_bytes().splitlines())


def load_evaluation(directory: Path, name: str = "evaluation") -> EvaluationOutcome:
    return EvaluationOutcome(
        metrics=EvaluationMetrics.model_validate_json((directory / f"{name}.json").read_bytes()),
        samples=load_records(directory / f"{name}_generations.jsonl", SampleGeneration),
        example_losses=load_records(directory / f"{name}_losses.jsonl", ExampleLoss),
        diagnostics=TypeAdapter(tuple[ConditioningDiagnostic, ...]).validate_json(
            (directory / f"{name}_conditioning.json").read_bytes()
        ),
    )


def cached_evaluation(
    wrapper: FrozenQwen,
    projector: Projector | None,
    examples: tuple[Example, ...],
    directory: Path,
    semantic: SemanticEvaluator,
    condition: EvaluationCondition = EvaluationCondition.SPEECH,
    transcripts: tuple[AsrTranscript, ...] = (),
) -> EvaluationOutcome:
    if directory.exists():
        return load_evaluation(directory)
    pending = directory.with_name(directory.name + ".pending")
    if pending.exists():
        pending.replace(pending.with_name(pending.name + f".interrupted-{time.time_ns()}"))
    outcome = evaluate(
        wrapper,
        projector,
        examples,
        wrapper.config,
        condition,
        asr_transcripts=transcripts,
        semantic_evaluator=semantic,
    )
    save_evaluation(outcome, pending)
    pending.replace(directory)
    return outcome


def write_incremental_summary(root: Path, completed: list[str]) -> None:
    candidates = tuple(
        SweepCandidate.model_validate_json((root / name / "candidate.json").read_bytes())
        for name in completed
    )
    pending = root / "study_summary.pending.json"
    pending.write_bytes(TypeAdapter(tuple[SweepCandidate, ...]).dump_json(candidates, indent=2))
    pending.replace(root / "study_summary.json")
    lines = [
        "# Overnight experiment progress",
        "",
        "Validation only; no final quality claim before held-out evaluation.",
        "",
        "| Run | Steps | Rate/s | Macro CE | Ordinary CE | New Neu raw / resized margin |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        result = RunResult.model_validate_json(
            (root / candidate.configuration.name / "result.json").read_bytes()
        )
        lines.append(
            f"| {candidate.configuration.name} | {result.steps} | "
            f"{candidate.pseudo_tokens_per_second:g} | "
            f"{candidate.validation.macro_cross_entropy:.4f} | "
            f"{candidate.validation.old_ordinary.cross_entropy:.4f} | "
            f"{candidate.new_neu_preference.matching_margin.estimate:.4f} / "
            f"{candidate.new_neu_preference.resized_matching_margin.estimate:.4f} |"
        )
    (root / "study_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_candidate(
    config: RunConfig,
    training: list[Example],
    validation: tuple[Example, ...],
    sources: tuple[SourceSidecar, ...],
    wrapper: FrozenQwen,
    semantic: SemanticEvaluator,
    output_root: Path,
) -> SweepCandidate:
    directory = output_root / config.name
    candidate_path = directory / "candidate.json"
    if candidate_path.exists():
        candidate = SweepCandidate.model_validate_json(candidate_path.read_bytes())
        if candidate.configuration != config:
            raise ValueError("A completed candidate has a different experiment configuration")
        result = RunResult.model_validate_json((directory / "result.json").read_bytes())
        state = TrainingState.model_validate_json(
            (directory / "checkpoint" / "state.json").read_bytes()
        )
        if state.step != result.steps or not result.checkpoint_path.is_file():
            raise ValueError("Completed candidate checkpoint/state is missing or inconsistent")
        return candidate
    wrapper.config = config
    source_revision = git_revision()
    projector = initialize_projector(config, wrapper.device)
    smoke_path = output_root / "smoke" / config.name / "gradient_check.json"
    if smoke_path.exists():
        checked = GradientCheck.model_validate_json(smoke_path.read_bytes())
        assert (
            checked.projector_changed
            and checked.llm_weights_unchanged
            and not checked.llm_has_gradients
        )
    else:
        checked = gradient_sanity(wrapper, projector, training[0])
        write_record(smoke_path, checked)
        projector = initialize_projector(config, wrapper.device)
    outcome = train_run(config, training, list(validation), directory, wrapper, projector)
    evaluated = cached_evaluation(
        wrapper, projector, validation, directory / "validation", semantic
    )
    cohorts = summarize_cohort_validation(evaluated, sources, EvaluationCondition.SPEECH)
    old_pairs = build_emotion_pairs(validation, sources, Cohort.QWEN_EMOTIONAL, Split.VALIDATION)
    new_pairs = build_emotion_pairs(validation, sources, Cohort.NEU_EMOTIONAL, Split.VALIDATION)
    old_preference = summarize_preferences(
        evaluate_emotion_pairs(
            wrapper, projector, old_pairs, directory / "old_emotional_preference"
        )
    )
    new_preference = summarize_preferences(
        evaluate_emotion_pairs(wrapper, projector, new_pairs, directory / "new_neu_preference")
    )
    result = RunResult(
        config=config,
        git_commit=source_revision,
        train_examples=len(training),
        validation_examples=len(validation),
        test_examples=0,
        projector_parameters=projector.parameter_count,
        pseudo_tokens_per_second=config.projector.native_rate / config.projector.compression_factor,
        mean_pseudo_tokens=sum(
            math.ceil(
                example.duration
                * config.projector.native_rate
                / config.projector.compression_factor
            )
            for example in validation
        )
        / len(validation),
        steps=outcome.steps,
        runtime_seconds=outcome.runtime_seconds,
        peak_vram_gb=outcome.peak_vram_gb,
        examples_per_second=outcome.examples_seen / outcome.runtime_seconds,
        target_tokens_per_second=outcome.target_tokens_seen / outcome.runtime_seconds,
        initial_validation_loss=outcome.initial_validation_loss,
        initial_training_loss=outcome.initial_training_loss,
        final_fixed_training_loss=outcome.final_fixed_training_loss,
        final_training_loss=outcome.final_training_loss,
        validation=evaluated.metrics,
        checkpoint_path=outcome.checkpoint_path,
    )
    write_record(directory / "result.json", result)
    candidate = SweepCandidate(
        configuration=config,
        validation=cohorts,
        old_emotional_preference=old_preference,
        new_neu_preference=new_preference,
        pseudo_tokens_per_second=result.pseudo_tokens_per_second,
        training_seconds_per_update=outcome.runtime_seconds / outcome.steps,
    )
    write_record(candidate_path, candidate)
    return candidate


def execute_sweep(
    config: OvernightSuiteConfig, *, stop_after_shortlist: bool = False
) -> OvernightCompletion | SweepDecision:
    examples = load_records(config.data_root / "examples.jsonl", Example)
    training = [example for example in examples if example.split == Split.TRAIN]
    validation = load_records(config.data_root / "fixed_validation.jsonl", Example)
    sources = load_sources(config.data_root / "fixed_validation_sources.jsonl")
    if any(not example.feature_path.is_file() for example in examples):
        raise ValueError("All missing frozen Whisper features must be cached before Qwen is loaded")
    configurations = sweep_runs(len(training), config.microbatch_size)
    config.output_root.mkdir(parents=True, exist_ok=True)
    suite_path = config.output_root / "suite_config.json"
    if (
        suite_path.exists()
        and OvernightSuiteConfig.model_validate_json(suite_path.read_bytes()) != config
    ):
        raise ValueError("Cannot resume an overnight queue with changed config/deadline")
    write_record(suite_path, config)
    state_path = config.output_root / "suite_state.json"
    previous = (
        SuiteState.model_validate_json(state_path.read_bytes())
        if state_path.exists()
        else SuiteState(
            completed=(), running=None, failed=(), started_at=time.time(), updated_at=time.time()
        )
    )
    completed = list(previous.completed)
    failed = list(previous.failed)
    wrapper = FrozenQwen(configurations[0], torch.device("cuda"))
    semantic = SemanticEvaluator()
    parity_path = config.output_root / "smoke" / "batched_parity.json"
    if parity_path.exists():
        parity = BatchedGradientCheck.model_validate_json(parity_path.read_bytes())
    else:
        ordinary = next(
            example for example in training if example.example_id.startswith("ordinary:")
        )
        emotional = next(example for example in training if example.example_id.startswith("neu:"))
        parity = batched_gradient_sanity(
            wrapper, initialize_projector(configurations[0], wrapper.device), [ordinary, emotional]
        )
        write_record(parity_path, parity)
    assert (
        parity.llm_weights_unchanged
        and parity.projector_weights_unchanged
        and not parity.llm_has_gradients
        and parity.reference_gradient_norm > 0
        and parity.batched_gradient_norm > 0
    )
    try:
        validate_batched_parity(parity, config.batched_parity)
    except ValueError as error:
        failure = ExperimentFailure(
            run_name="batched_gradient_parity",
            attempt=1,
            exception_type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
            timestamp=time.time(),
        )
        write_record(config.output_root / "smoke" / "batched_parity_failure.json", failure)
        if config.microbatch_size > 1:
            raise
        print(
            "Batched parity failed; configured microbatch1 uses the official individual loss path.",
            flush=True,
        )

    def save_state(running: str | None) -> None:
        write_record(
            state_path,
            SuiteState(
                completed=tuple(completed),
                running=running,
                failed=tuple(failed),
                started_at=previous.started_at,
                updated_at=time.time(),
            ),
        )

    def execute(configuration: RunConfig) -> SweepCandidate:
        save_state(configuration.name)
        for attempt in range(1, 3):
            try:
                candidate = run_candidate(
                    configuration,
                    training,
                    validation,
                    sources,
                    wrapper,
                    semantic,
                    config.output_root,
                )
                if configuration.name not in completed:
                    completed.append(configuration.name)
                if configuration.name in failed:
                    failed.remove(configuration.name)
                write_incremental_summary(config.output_root, completed)
                save_state(None)
                return candidate
            except Exception as error:
                failure = ExperimentFailure(
                    run_name=configuration.name,
                    attempt=attempt,
                    exception_type=type(error).__name__,
                    message=str(error),
                    traceback=traceback.format_exc(),
                    timestamp=time.time(),
                )
                with (config.output_root / "failures.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(failure.model_dump_json() + "\n")
                print(failure.model_dump_json(), flush=True)
                torch.cuda.empty_cache()
                if attempt == 2:
                    if configuration.name not in failed:
                        failed.append(configuration.name)
                    save_state(None)
                    raise
        raise AssertionError("Retry loop must complete or raise")

    candidates = tuple(execute(configuration) for configuration in configurations)
    decision = select_sweep(candidates, config.selection)
    write_record(config.output_root / "decision.json", decision)
    selected_names = (decision.quality_leader,) + decision.compact_winners
    latest = {
        candidate.configuration.name: candidate
        for candidate in candidates
        if candidate.configuration.name in selected_names
    }
    stops: list[ContinuationStop] = []
    updates_per_epoch = math.ceil(len(training) / 8)

    def continue_to(source: SweepCandidate, name: str, updates: int) -> SweepCandidate | None:
        source_config = source.configuration
        source_result = RunResult.model_validate_json(
            (config.output_root / source_config.name / "result.json").read_bytes()
        )
        continuation_config = source_config.model_copy(
            update={"name": name, "max_optimizer_updates": updates}
        )
        destination = config.output_root / name
        current_steps = (
            TrainingState.model_validate_json(
                (destination / "checkpoint" / "state.json").read_bytes()
            ).step
            if (destination / "checkpoint" / "state.json").exists()
            else source_result.steps
        )
        estimate = (updates - current_steps) * source.training_seconds_per_update + 120
        if (
            not (destination / "candidate.json").exists()
            and time.time() + estimate + config.finalization_reserve_seconds
            > config.deadline_unix_time
        ):
            stops.append(
                ContinuationStop(
                    selected_sweep=source_config.name,
                    last_completed_run=source_config.name,
                    reason="Insufficient budget for continuation plus finalization reserve",
                )
            )
            write_record(config.output_root / "budget_stop.json", stops[-1])
            return None
        prepare_continuation(
            config.output_root / source_config.name, destination, continuation_config
        )
        return execute(continuation_config)

    for name in selected_names:
        continued = continue_to(latest[name], f"{name}_updates4000", 4000)
        if continued is not None:
            latest[name] = continued
    shortlist = select_sweep(tuple(latest.values()), config.selection)
    write_record(config.output_root / "shortlist_decision.json", shortlist)
    if stop_after_shortlist:
        save_state(None)
        return shortlist
    winner = next(
        candidate
        for candidate in latest.values()
        if candidate.configuration.name == shortlist.quality_leader
    )
    trajectory = [winner]
    for epoch in range(1, config.continuation_epoch_ceiling + 1):
        updates = epoch * updates_per_epoch
        current_steps = RunResult.model_validate_json(
            (config.output_root / winner.configuration.name / "result.json").read_bytes()
        ).steps
        if current_steps >= updates:
            continue
        continued = continue_to(winner, f"{shortlist.quality_leader}_epoch{epoch}", updates)
        if continued is None:
            break
        improvement = (
            winner.validation.macro_cross_entropy - continued.validation.macro_cross_entropy
        )
        trajectory.append(continued)
        winner = continued
        if improvement < config.continuation_minimum_improvement:
            stops.append(
                ContinuationStop(
                    selected_sweep=shortlist.quality_leader,
                    last_completed_run=winner.configuration.name,
                    reason=f"Validation macro CE improvement {improvement:.6f} below threshold "
                    f"{config.continuation_minimum_improvement:.6f}",
                )
            )
            break
    if (
        RunResult.model_validate_json(
            (config.output_root / winner.configuration.name / "result.json").read_bytes()
        ).steps
        < updates_per_epoch
    ):
        raise ValueError("Final winner has not completed one full training pass; extend the budget")
    eligible_trajectory = tuple(
        candidate
        for candidate in trajectory
        if RunResult.model_validate_json(
            (config.output_root / candidate.configuration.name / "result.json").read_bytes()
        ).steps
        >= updates_per_epoch
    )
    final_decision = select_sweep(eligible_trajectory, config.selection)
    write_record(config.output_root / "final_selection.json", final_decision)
    final_winner = next(
        candidate
        for candidate in eligible_trajectory
        if candidate.configuration.name == final_decision.quality_leader
    )
    finalists = (final_winner,) + tuple(
        candidate
        for candidate in latest.values()
        if candidate.configuration.name in shortlist.compact_winners
    )
    test = load_records(config.data_root / "fixed_test.jsonl", Example)
    test_sources = load_sources(config.data_root / "fixed_test_sources.jsonl")
    final_selection = select_final_evaluation(test, test_sources, FinalJudgingQuota())
    write_record(config.output_root / "final_evaluation_selection.json", final_selection)
    test = final_selection.examples
    for candidate in finalists:
        wrapper.config = final_evaluation_config(candidate.configuration, final_selection)
        projector = initialize_projector(candidate.configuration, wrapper.device)
        directory = config.output_root / candidate.configuration.name
        result = RunResult.model_validate_json((directory / "result.json").read_bytes())
        write_record(directory / "final_evaluation_config.json", wrapper.config)
        projector.load_state_dict(
            load_file(str(result.checkpoint_path), device=str(wrapper.device))
        )
        tested = cached_evaluation(wrapper, projector, test, directory / "test", semantic)
        write_record(
            directory / "final_result.json",
            result.model_copy(update={"test": tested.metrics, "test_examples": len(test)}),
        )
        for cohort in (Cohort.QWEN_EMOTIONAL, Cohort.NEU_EMOTIONAL):
            pairs = build_emotion_pairs(test, test_sources, cohort, Split.TEST)
            evaluate_emotion_pairs(
                wrapper, projector, pairs, directory / "test" / f"{cohort.value}_preference"
            )
        evaluate_teacher_fidelity(wrapper, projector, validation, directory / "validation")
        evaluate_teacher_fidelity(wrapper, projector, test, directory / "test")
    wrapper.config = configurations[0]
    transcripts = load_asr_transcripts(config.data_root / "asr_transcripts.jsonl")
    for split, fixed in (("validation", validation), ("test", test)):
        wrapper.config = (
            final_evaluation_config(configurations[0], final_selection)
            if split == "test"
            else configurations[0]
        )
        for condition in (EvaluationCondition.TEXT, EvaluationCondition.ASR):
            cached_evaluation(
                wrapper,
                None,
                fixed,
                config.output_root / "baseline" / split / condition.value,
                semantic,
                condition,
                transcripts if condition == EvaluationCondition.ASR else (),
            )
    result = OvernightCompletion(
        decision=final_decision,
        final_runs=tuple(candidate.configuration.name for candidate in finalists),
        continuation_stops=tuple(stops),
        completed_at=time.time(),
    )
    write_record(config.output_root / "completion.json", result)
    save_state(None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--stop-after-shortlist", action="store_true")
    arguments = parser.parse_args()
    configuration = OvernightSuiteConfig.model_validate_json(arguments.config.read_bytes())
    if arguments.cache_only:
        examples = load_records(configuration.data_root / "examples.jsonl", Example)
        count = sum(example.split == Split.TRAIN for example in examples)
        print(
            extract_features(
                CacheConfig(
                    root=configuration.data_root, train_examples=count, transcribe_heldout=True
                )
            ).model_dump_json(indent=2),
            flush=True,
        )
    else:
        print(
            execute_sweep(
                configuration, stop_after_shortlist=arguments.stop_after_shortlist
            ).model_dump_json(indent=2),
            flush=True,
        )


if __name__ == "__main__":
    main()
