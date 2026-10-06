"""Render recorded mixed-corpus results without treating unrun tests as measurements."""

from collections.abc import Sequence
from pathlib import Path

from speech_projector.models import Record, RunResult, SampleGeneration
from speech_projector.overnight_data import (
    NeuEmotionalExampleSource,
    OrdinaryExampleSource,
    QwenEmotionalExampleSource,
    SourceSidecar,
)
from speech_projector.overnight_evaluation import SweepDecision
from speech_projector.overnight_preparation import CombinedPreparation


class OvernightReportConfig(Record):
    results_root: Path
    preparation_path: Path
    decision_path: Path
    output_directory: Path
    final_run_directories: tuple[Path, ...] = ()
    operational_notes: tuple[str, ...] = ()


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
        RunResult.model_validate_json((directory / "result.json").read_bytes())
        for directory in configuration.final_run_directories
    )
    text = render_overnight_report(
        preparation, decision, results, finals, configuration.operational_notes
    )
    configuration.output_directory.mkdir(parents=True, exist_ok=True)
    destination = configuration.output_directory / "research_report.md"
    destination.write_text(text, encoding="utf-8")
    (configuration.output_directory / "report_config.json").write_text(
        configuration.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return destination
