"""Render recorded mixed-corpus results without treating unrun tests as measurements."""

from collections.abc import Sequence
from pathlib import Path

from pydantic import TypeAdapter

from speech_projector.judge import JudgeOutcome
from speech_projector.models import (
    EvaluationCondition,
    EvaluationMetrics,
    Record,
    RunResult,
    SampleGeneration,
)
from speech_projector.overnight_conversation_execution import (
    ConversationEvaluationProvenance,
    ConversationEvaluationSummary,
)
from speech_projector.overnight_data import (
    Cohort,
    NeuEmotionalExampleSource,
    OrdinaryExampleSource,
    QwenEmotionalExampleSource,
    SourceSidecar,
)
from speech_projector.overnight_evaluation import SweepDecision
from speech_projector.overnight_judge import (
    FinalJudgingSummary,
    ToneCalibrationResult,
    ToneJudgeOutcome,
    paired_acceptability_difference,
)
from speech_projector.overnight_preparation import CombinedPreparation


class OvernightReportConfig(Record):
    results_root: Path
    preparation_path: Path
    decision_path: Path
    output_directory: Path
    final_run_directories: tuple[Path, ...] = ()
    operational_notes: tuple[str, ...] = ()
    judge_calibration_directory: Path | None = None


def _margin(estimate: float, lower: float, upper: float) -> str:
    return f"{estimate:+.4f} [{lower:+.4f}, {upper:+.4f}]"


def render_overnight_report(
    preparation: CombinedPreparation,
    decision: SweepDecision,
    sweep_results: Sequence[RunResult],
    final_results: Sequence[RunResult],
    operational_notes: Sequence[str],
) -> str:
    recorded = {item.config.name: item for item in sweep_results}
    expected = {item.configuration.name for item in decision.candidates}
    if len(recorded) != len(sweep_results) or set(recorded) != expected:
        raise ValueError("Report needs exact unique result coverage for all selection candidates")
    for candidate in decision.candidates:
        if recorded[candidate.configuration.name].config != candidate.configuration:
            raise ValueError("Selection candidate and recorded experiment configurations differ")
    lines = [
        "# Overnight mixed-corpus speech-projector results",
        "",
        f"Validation recommendation: **{decision.quality_leader}**. "
        + (
            "Compact recommendation: **" + ", ".join(decision.compact_winners) + "**."
            if decision.compact_winners
            else "No candidate meets the compact-selection criteria."
        ),
        "",
        "These results separate words, intended emotional delivery and free-response quality. "
        "Positive teacher-preference margins are supporting evidence, not verified emotion "
        "recognition or acceptable conversational behavior. Same-word controls remove literal "
        "text differences; duration, voice, synthesizer and corpus remain potential shortcuts.",
        "",
        "## Dataset and supervision",
        "",
        f"Immutable combined manifest SHA256: `{preparation.manifest.sha256}`; "
        f"source-identity sidecar SHA256: `{preparation.sidecar.sha256}`. "
        f"Preparation source: `{preparation.source_commit}`.",
        "",
        "Existing ordinary audio and Qwen targets, and existing Qwen emotional audio/targets, "
        "remain source artifacts. New Neu emotional data is a separate addition. The ordinary "
        "teacher uses native chat; each emotional cohort retains its own generic concise "
        "response policy. Emotional targets were generated with intended-tone metadata. "
        "The speech student receives the generic policy and audio, never the current words "
        "or actual tone annotation. Words-only text/ASR references therefore have less "
        "delivery information than the privileged annotated teacher targets.",
        "",
        "| Cohort | Split | Examples | Mean duration s | Median s | P90 s |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in preparation.coverage:
        duration = item.duration_seconds
        lines.append(
            f"| {item.cohort.value} | {item.split.value} | {item.examples} | "
            f"{duration.mean:.3f} | {duration.median:.3f} | {duration.p90:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Whole-pair duration exclusions: {len(preparation.exclusions)}. "
            "Excluded source files are preserved, not truncated or regenerated. "
            f"Feature reuse at preparation: {preparation.cached_features_reused}; "
            f"features missing at that snapshot: {preparation.missing_features}. "
            "These preparation counts are not final cache-storage or extraction-time measurements.",
            "",
            f"Copied ordinary TRAIN prompt-overlap exclusions: "
            f"{len(preparation.ordinary_prompt_exclusions)} whole dialogues / "
            f"{preparation.ordinary_prompt_excluded_examples} examples. "
            "Original source records remain preserved; held-out examples are unchanged. "
            "The preparation receipt records protected validation/test identities.",
            "",
            "Synthetic role/content and intended-tone limitations remain. There is no independent "
            "natural-speaker holdout. Family grouping prevents exact paired-family leakage, "
            "but does not establish broad paraphrase or synthesizer generalization.",
            "",
            "## Fixed-budget validation sweep",
            "",
            "Fixed validation covers 48 ordinary examples and 20 complete paired bases from each "
            "emotional cohort (128 clips total); the first 24 generated responses are balanced "
            "8 per cohort, with both recordings of each selected emotional base. Ordinary examples "
            "are history-stratified when possible; emotional selection balances contrast types "
            "and prefers distinct families. The saved selection artifact contains "
            "exact membership.",
            "",
            "Each cohort CE weights its target tokens. Macro CE weights the three cohorts equally; "
            "it prevents longer ordinary teacher replies from dominating the mixture. "
            "Paired margins average target-normalized four-way differences by base. "
            "Intervals resample whole families, retaining both tones together (95% percentile "
            "bootstrap, 2,000 draws); they omit training-seed, synthesis and judge uncertainty.",
            "",
            "| Run | Speech tokens/s | Params | Updates | Ordinary CE | Old emotion CE | "
            "Neu emotion CE | Macro CE | Neu raw margin [95% CI] | Neu resized margin [95% CI] |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for candidate in decision.candidates:
        result = recorded[candidate.configuration.name]
        cohorts = candidate.validation
        natural = candidate.new_neu_preference.matching_margin
        resized = candidate.new_neu_preference.resized_matching_margin
        lines.append(
            f"| {candidate.configuration.name} | {candidate.pseudo_tokens_per_second:g} | "
            f"{result.projector_parameters:,} | {result.steps} | "
            f"{cohorts.old_ordinary.cross_entropy:.4f} | "
            f"{cohorts.old_emotional.cross_entropy:.4f} | "
            f"{cohorts.new_neu_emotional.cross_entropy:.4f} | {cohorts.macro_cross_entropy:.4f} | "
            f"{_margin(natural.estimate, natural.lower, natural.upper)} | "
            f"{_margin(resized.estimate, resized.lower, resized.upper)} |"
        )
    lines.extend(
        [
            "",
            "Wrong-tone states are also linearly resized to the correct state's length before "
            "projection, holding pseudo-token count fixed. This changes feature statistics. "
            "The raw and resized margins answer complementary questions and neither is a "
            "standalone proof of semantic tone understanding.",
            "",
            decision.rationale,
            "",
            "Precommitted ordinary guard: best ordinary CE + "
            f"{decision.policy.ordinary_ce_tolerance:g}; "
            f"macro tie band {decision.policy.macro_ce_tie_band:g}. "
            "Within that band the lower of raw/resized Neu matching margins breaks the tie, "
            "then training speed. "
            f"Compact tolerance {decision.policy.compact_macro_ce_tolerance:g} macro nats with "
            f"at least {decision.policy.minimum_token_rate_reduction:g}× token-rate reduction. "
            "Test results do not affect this recommendation.",
            "",
            "| Run | Old paired bases / families | Old raw margin [95% CI] | "
            "Neu paired bases / families | Neu win rate | Neu ties | Identical Neu targets |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for candidate in decision.candidates:
        old = candidate.old_emotional_preference
        new = candidate.new_neu_preference
        margin = old.matching_margin
        lines.append(
            f"| {candidate.configuration.name} | {old.pairs}/{old.family_clusters} | "
            f"{_margin(margin.estimate, margin.lower, margin.upper)} | "
            f"{new.pairs}/{new.family_clusters} | {new.matching_win_rate.estimate:.1%} | "
            f"{new.tie_rate:.1%} | {new.pairs - new.distinct_target_pairs}/{new.pairs} |"
        )
    lines.extend(
        [
            "",
            "The older emotional corpus is reported separately because its targets and audible "
            "contrasts may be weak. Identical teacher targets necessarily carry no contrastive "
            "target preference; distinct strings still do not guarantee useful adaptation. "
            "Teacher fidelity and appropriate responses must be assessed separately.",
            "",
            "## Selected final runs and test coverage",
            "",
            "| Run | Updates | Validation CE | Test examples | Test CE | "
            "Test semantic similarity |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in final_results:
        match result.test:
            case None:
                lines.append(
                    f"| {result.config.name} | {result.steps} | "
                    f"{result.validation.cross_entropy:.4f} | Not evaluated | — | — |"
                )
            case test:
                semantic = (
                    "—" if test.semantic_similarity is None else f"{test.semantic_similarity:.4f}"
                )
                lines.append(
                    f"| {result.config.name} | {result.steps} | "
                    f"{result.validation.cross_entropy:.4f} | {test.examples} | "
                    f"{test.cross_entropy:.4f} | {semantic} |"
                )
    if not final_results:
        lines.extend(["", "No completed selected final-run record was supplied."])
    lines.extend(
        [
            "",
            "Semantic cosine is a secondary response-similarity proxy. Inspect actual generations "
            "and blinded rubric judgments; do not equate similar embeddings or teacher tokens "
            "with content correctness, emotional appropriateness or human approval.",
            "",
            "## Measured training resources",
            "",
            "| Run | Recorded training wall s | Seconds/update | "
            "Peak PyTorch allocated decimal GB |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for result in tuple(sweep_results) + tuple(final_results):
        lines.append(
            f"| {result.config.name} | {result.runtime_seconds:.1f} | "
            f"{result.runtime_seconds / result.steps:.3f} | {result.peak_vram_gb:.3f} |"
        )
    lines.extend(
        [
            "",
            "Training clocks do not include all source generation, TTS, cache preparation, "
            "loads, final evaluation or judge work. They are neither measured GPU-busy hours "
            "nor billing duration. Per-cohort evaluation clocks cannot be decomposed from the "
            "saved combined outcome and are unmeasured.",
            "",
            "## Failures and operational notes",
            "",
        ]
    )
    if operational_notes:
        lines.extend("- " + note for note in operational_notes)
    else:
        lines.append("No operational failure notes were supplied to this renderer.")
    lines.extend(
        [
            "",
            "## Conversation diagnostic",
            "",
            "Four-turn and delayed-acoustic-cue results require explicit per-turn evidence. "
            "Identical fixed assistant messages isolate delayed audio context; generated "
            "assistant histories are separate practical rollouts because their text can reveal "
            "tone. Report speech-history, text-history and omitted-initial-turn controls "
            "separately. This summary does not infer conversation success from single-turn scores.",
            "",
        ]
    )
    return "\n".join(lines)


def render_generations(
    samples: Sequence[SampleGeneration], sources: Sequence[SourceSidecar]
) -> str:
    sources_by_id = {item.example_id: item for item in sources}
    lines = ["# Fixed generated replies", ""]
    for sample in samples:
        source = sources_by_id[sample.example_id]
        label = source.cohort.value
        match source:
            case OrdinaryExampleSource():
                delivery = "Ordinary conversation; no delivery annotation supplied."
            case QwenEmotionalExampleSource() | NeuEmotionalExampleSource():
                delivery = (
                    f"Intended user delivery: {source.emotion.value}; not verified perception."
                )
        termination = "unknown" if sample.generation is None else sample.generation.kind.value
        lines.extend(
            [
                f"## {sample.example_id} — {label} — {sample.condition.value}",
                "",
                delivery,
                "",
                f"User words: {sample.user_transcript}",
                "",
                "Textual history:",
                "",
                *(f"- {turn.role.value}: {turn.text}" for turn in sample.history),
                "",
                "Saved supervision (annotation-privileged for emotional examples):",
                "",
                "> " + sample.gold_response.replace("\n", "\n> "),
                "",
                "Actual generated reply:",
                "",
                "> " + sample.generated_response.replace("\n", "\n> "),
                "",
                f"Generation termination: {termination}.",
                "",
            ]
        )
    return "\n".join(lines)


def render_baselines(results_root: Path) -> str:
    lines = [
        "## Recorded words-only references",
        "",
        "The ordinary transcript branch has the textual input used by its teacher. Emotional "
        "text and ASR branches receive words and the generic response policy, without the "
        "intended-tone metadata used to construct their teacher targets. They are words-only "
        "comparators, not information-complete emotional upper bounds. Aggregate CE weights "
        "target tokens and should be compared on identical saved example membership.",
        "",
        "| Split | Input | CE examples | CE | Semantic cosine | Generations | Completed / capped |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for split in ("validation", "test"):
        for condition in (EvaluationCondition.TEXT, EvaluationCondition.ASR):
            path = results_root / "baseline" / split / condition.value / "evaluation.json"
            if not path.exists():
                lines.append(f"| {split} | {condition.value} | Not measured | — | — | — | — |")
                continue
            metrics = EvaluationMetrics.model_validate_json(path.read_bytes())
            semantic = (
                "—" if metrics.semantic_similarity is None else f"{metrics.semantic_similarity:.4f}"
            )
            completed = (
                "unknown"
                if metrics.completed_generations is None
                else str(metrics.completed_generations)
            )
            capped = (
                "unknown"
                if metrics.token_limited_generations is None
                else str(metrics.token_limited_generations)
            )
            lines.append(
                f"| {split} | {condition.value} | {metrics.examples} | "
                f"{metrics.cross_entropy:.4f} | {semantic} | {metrics.generated_examples} | "
                f"{completed} / {capped} |"
            )
    return "\n".join(lines)


def render_final_judging(directory: Path, calibration: ToneCalibrationResult | None) -> str:
    if calibration is None:
        return (
            f"Blinded final response judging for {directory.name}: calibration result not "
            "supplied. Quality rates are not presented without the tone-calibration gate."
        )
    gate = (
        f"Tone calibration: **{'passed' if calibration.passed else 'FAILED'}**; "
        f"{calibration.correct_acceptability}/{calibration.requested} candidate decisions "
        f"({calibration.valid} valid JSON outputs), "
        f"{calibration.preference_correct}/{calibration.preference_requests} paired decisions. "
        f"Judge: `{calibration.configuration.model_name}` at "
        f"`{calibration.configuration.revision}`. Calibration wall seconds "
        f"{calibration.runtime_seconds:.1f}; peak PyTorch allocated decimal GB "
        f"{calibration.peak_pytorch_allocated_decimal_gb:.3f}."
    )
    if not calibration.passed:
        return gate + " Quality rates are suppressed; raw failed calibration evidence is retained."
    lines = [
        f"## Blinded final response judgments — {directory.name}",
        "",
        gate,
        "",
        "These are compact-model rubric judgments, not human ratings or acoustic emotion "
        "recognition. The judge sees true words, history and intended emotional annotation, "
        "even though the words-only generators did not receive that annotation. A successful "
        "summary requires the dedicated tone calibration gate; the calibration evidence "
        "and raw judgments remain necessary to assess reliability. Acceptance includes "
        "failed JSON outputs in its denominator. Emotional intervals resample families "
        "(95% percentile bootstrap); they omit judge bias and training-seed uncertainty.",
        "",
        "| Input | Cohort | Requested / valid / failed | Acceptable | Tone score / 3 | "
        "Acceptable 95% CI |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    summaries: list[FinalJudgingSummary] = []
    for condition in (
        EvaluationCondition.SPEECH,
        EvaluationCondition.TEXT,
        EvaluationCondition.ASR,
    ):
        path = directory / "judge" / condition.value / "summary.json"
        if not path.exists():
            lines.append(f"| {condition.value} | All | Not measured | — | — | — |")
            continue
        summary = FinalJudgingSummary.model_validate_json(path.read_bytes())
        if summary.condition != condition:
            raise ValueError("Final judge summary condition differs from its output directory")
        summaries.append(summary)
        ordinary = summary.ordinary
        lines.append(
            f"| {condition.value} | ordinary | {ordinary.requested_examples} / "
            f"{ordinary.valid_examples} / {ordinary.failed_examples} | "
            f"{ordinary.acceptable_rate_requested:.1%} | — | Not recorded |"
        )
        for cohort, judged in (
            ("old emotional", summary.old_emotional),
            ("Neu emotional", summary.new_emotional),
        ):
            interval = judged.acceptable_interval
            lines.append(
                f"| {condition.value} | {cohort} | {judged.requested} / {judged.valid} / "
                f"{judged.failed} | {judged.acceptable_rate_requested:.1%} | "
                f"{judged.tone_appropriateness:.3f} | "
                f"[{interval.lower:.1%}, {interval.upper:.1%}] |"
            )
    lines.extend(
        [
            "",
            "Paired A/B judgments compare the two actual responses to the same words under each "
            "intended delivery. Slots are blinded and deterministically shuffled. Identical "
            "responses are counted as ties regardless of an arbitrary judge slot choice. "
            "This measures whether different responses fit the annotation, not whether listeners "
            "actually hear that tone.",
            "",
            "| Input | Cohort | A/B requested / valid / failed | Matches / ties / mismatches | "
            "Matching rate 95% CI |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for summary in summaries:
        for cohort, judged in (
            ("old emotional", summary.old_preferences),
            ("Neu emotional", summary.new_preferences),
        ):
            interval = judged.matching_win_rate
            lines.append(
                f"| {summary.condition.value} | {cohort} | {judged.requested} / {judged.valid} / "
                f"{judged.failed} | {judged.matching_wins} / {judged.ties} / "
                f"{judged.matching_losses} | {interval.estimate:.1%} "
                f"[{interval.lower:.1%}, {interval.upper:.1%}] |"
            )
    lines.extend(
        [
            "",
            "| Input | Main judging wall s | Additional A/B judging wall s | Sum s |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for summary in summaries:
        lines.append(
            f"| {summary.condition.value} | {summary.main_judging_seconds:.1f} | "
            f"{summary.paired_judging_seconds:.1f} | "
            f"{summary.main_judging_seconds + summary.paired_judging_seconds:.1f} |"
        )
    lines.extend(
        ["", "These clocks exclude model loading and calibration; they are not GPU-busy hours."]
    )
    lines.extend(["", render_paired_quality(directory)])
    return "\n".join(lines)


def _load_quality_journal(
    path: Path, cohort: Cohort
) -> tuple[JudgeOutcome | ToneJudgeOutcome, ...]:
    contents = path.read_bytes()
    if contents and not contents.endswith(b"\n"):
        raise ValueError("Final quality reporting requires a complete immutable judgment journal")
    if cohort == Cohort.ORDINARY:
        ordinary_adapter = TypeAdapter(JudgeOutcome)
        return tuple(ordinary_adapter.validate_json(line) for line in contents.splitlines())
    tone_adapter = TypeAdapter(ToneJudgeOutcome)
    return tuple(tone_adapter.validate_json(line) for line in contents.splitlines())


def render_paired_quality(directory: Path) -> str:
    lines = [
        "### Paired acceptable-rate differences",
        "",
        "Speech minus words-only reference, using only cases with valid judgments in both "
        "conditions. This conditional matched-valid denominator differs from the requested-case "
        "acceptance rates above. Whole dialogue/family resampling retains paired recordings; "
        "95% intervals do not capture judge calibration uncertainty or systematic bias.",
        "",
        "| Reference | Cohort | Jointly valid / speech requested / reference requested | "
        "Dialogue or family clusters | Difference [95% CI] |",
        "| --- | --- | --- | ---: | --- |",
    ]
    speech = directory / "judge" / EvaluationCondition.SPEECH.value
    for condition in (EvaluationCondition.TEXT, EvaluationCondition.ASR):
        reference = directory / "judge" / condition.value
        if not (speech / "summary.json").exists() or not (reference / "summary.json").exists():
            lines.append(f"| {condition.value} | All | Not measured | — | — |")
            continue
        for cohort, subdirectory in (
            (Cohort.ORDINARY, "ordinary"),
            (Cohort.QWEN_EMOTIONAL, "old_emotional"),
            (Cohort.NEU_EMOTIONAL, "new_emotional"),
        ):
            primary = _load_quality_journal(speech / subdirectory / "judgments.jsonl", cohort)
            other = _load_quality_journal(reference / subdirectory / "judgments.jsonl", cohort)
            interval = paired_acceptability_difference(
                EvaluationCondition.SPEECH, condition, primary, other
            )
            lines.append(
                f"| {condition.value} | {cohort.value} | {interval.examples} / {len(primary)} / "
                f"{len(other)} | {interval.dialogues} | {interval.estimate:+.1%} "
                f"[{interval.lower:+.1%}, {interval.upper:+.1%}] |"
            )
    return "\n".join(lines)


def render_conversation_measurements(directory: Path) -> str:
    path = directory / "conversation" / "summary.json"
    if not path.exists():
        return f"Conversation measurements for {directory.name}: not recorded."
    summary = ConversationEvaluationSummary.model_validate_json(path.read_bytes())
    provenance_path = directory / "conversation" / "provenance.json"
    provenance = ConversationEvaluationProvenance.model_validate_json(provenance_path.read_bytes())
    evidence_path = directory / "conversation" / "conversation_evidence.md"
    return (
        f"Recorded conversation measurements for {directory.name}: {summary.scenarios} paired "
        f"scenarios, {summary.controlled_responses} controlled replies, {summary.rollouts} "
        f"practical rollouts ({summary.rollout_responses} replies); "
        f"{summary.completed_responses} completed and {summary.token_limited_responses} capped "
        f"responses. Recorded generation wall time: {summary.generation_seconds:.1f} seconds. "
        f"Evaluation source `{provenance.source_commit}`, projector SHA256 "
        f"`{provenance.projector_weights_sha256}`, fixture SHA256 `{provenance.fixtures_sha256}`. "
        f"History budget {provenance.configuration.history_turns} turns / "
        f"{provenance.configuration.max_history_tokens} tokens. "
        f"Inspect [saved replies]({(directory / 'conversation' / 'replies.jsonl').as_posix()}) "
        "and "
        f"[readable evidence]({evidence_path.as_posix()}); "
        "counts alone do not establish delayed-cue use. Controlled branches share fixed "
        "assistant text; practical rollouts can carry the initial cue in generated text."
    )


def report_overnight(configuration: OvernightReportConfig) -> Path:
    preparation = CombinedPreparation.model_validate_json(
        configuration.preparation_path.read_bytes()
    )
    decision = SweepDecision.model_validate_json(configuration.decision_path.read_bytes())
    results = tuple(
        RunResult.model_validate_json(
            (configuration.results_root / item.configuration.name / "result.json").read_bytes()
        )
        for item in decision.candidates
    )
    finals = tuple(
        RunResult.model_validate_json((directory / "final_result.json").read_bytes())
        for directory in configuration.final_run_directories
    )
    text = render_overnight_report(
        preparation, decision, results, finals, configuration.operational_notes
    )
    text += (
        f"\n\nRecorded selection values: [decision.json]({configuration.decision_path.as_posix()})."
    )
    for name in ("compression_sweep.png", "compression_sweep.pdf", "compression_values.csv"):
        path = configuration.output_directory / name
        if path.exists():
            text += f"\n\n[{name}]({path.as_posix()})"
    text += "\n\n" + render_baselines(configuration.results_root)
    calibration = (
        None
        if configuration.judge_calibration_directory is None
        else ToneCalibrationResult.model_validate_json(
            (configuration.judge_calibration_directory / "result.json").read_bytes()
        )
    )
    for directory in configuration.final_run_directories:
        text += "\n\n" + render_final_judging(directory, calibration)
        text += "\n\n" + render_conversation_measurements(directory)
    configuration.output_directory.mkdir(parents=True, exist_ok=True)
    destination = configuration.output_directory / "research_report.md"
    destination.write_text(text, encoding="utf-8")
    (configuration.output_directory / "report_config.json").write_text(
        configuration.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return destination
