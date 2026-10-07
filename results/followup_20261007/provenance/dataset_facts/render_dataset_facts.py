"""Report-ready facts from sealed source evidence; no model calls or data changes."""

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

from pydantic import TypeAdapter

from artifacts.summarize_combined_dataset import CombinedDatasetAudit
from scripts.inventory_results import stable_digest, write_record
from scripts.prepare_teacher_data import TeacherDataPreparation
from speech_projector.cache import CacheStatistics
from speech_projector.evaluation import ExampleLoss
from speech_projector.models import (
    EvaluationCondition,
    Example,
    FileArtifact,
    Record,
    SampleGeneration,
    Split,
)
from speech_projector.overnight_data import Cohort, SourceSidecar


class PanelCount(Record):
    scope: str
    cohort: Cohort
    examples: int


class HardwareFacts(Record):
    measured_at: datetime
    gpu_name: str
    gpu_memory_mib: int
    nvidia_driver: str
    workspace_capacity_bytes: int
    workspace_free_bytes: int
    container_memory_limit_bytes: int
    container_memory_charged_bytes: int
    container_memory_limit_path: str
    host_memory_bytes: int
    host_memory_available_bytes: int
    swap_bytes: int


class TeacherCleaningCounts(Record):
    split: Split
    candidate_examples: int
    selected_examples: int
    selected_distinct_dialogues: int
    material_alignment_exclusions: int
    lexical_substitution_exclusions: int
    missing_synthesis_text_exclusions: int
    collision_dialogues_excluded: int


class DatasetFacts(Record):
    evidence: tuple[FileArtifact, ...]
    prepared: CombinedDatasetAudit
    cache: CacheStatistics
    panels: tuple[PanelCount, ...]
    full_train_target_tokens_including_eos: int
    maximum_full_target_tokens_including_eos: int
    hardware: HardwareFacts
    ordinary_cleaning: tuple[TeacherCleaningCounts, ...]
    ordinary_transcript_reference: str
    ordinary_subset_definition: str
    scope_notes: tuple[str, ...]


def evidence(path: Path) -> FileArtifact:
    size, digest = stable_digest(path)
    return FileArtifact(path=path, source_path=path.resolve(), bytes=size, sha256=digest)


def read_rows(path: Path, adapter: TypeAdapter[Example]) -> tuple[Example, ...]:
    return tuple(adapter.validate_json(line) for line in path.read_bytes().splitlines())


def build_facts(replay: Path, hardware: HardwareFacts) -> DatasetFacts:
    data = replay / "speech-projector/data_overnight_20261006"
    audit_path = replay / "provenance/combined_dataset/summary.json"
    cache_path = data / "cache_stats_38193.json"
    exposure_path = (
        replay / "provenance/checkpoint_operations/token_exposure_and_continuation_audit.md"
    )
    audit = CombinedDatasetAudit.model_validate_json(audit_path.read_bytes())
    cache = CacheStatistics.model_validate_json(cache_path.read_bytes())
    teacher_preparation_path = replay / "speech-projector/data_teacher/preparation.json"
    teacher_preparation = TeacherDataPreparation.model_validate_json(
        teacher_preparation_path.read_bytes()
    )
    sidecar_adapter = TypeAdapter(SourceSidecar)
    sources = tuple(
        sidecar_adapter.validate_json(line)
        for line in (data / "sources.jsonl").read_bytes().splitlines()
    )
    owners = {row.example_id: row.cohort for row in sources}
    panels: list[PanelCount] = []
    paths: list[Path] = [
        audit_path,
        cache_path,
        exposure_path,
        data / "sources.jsonl",
        teacher_preparation_path,
    ]
    for scope, filename in (
        ("fixed validation selection", "fixed_validation.jsonl"),
        ("prepared fixed test selection bank", "fixed_test.jsonl"),
    ):
        path = data / filename
        paths.append(path)
        rows = read_rows(path, TypeAdapter(Example))
        counts = Counter(owners[row.example_id] for row in rows)
        panels.extend(
            PanelCount(scope=scope, cohort=cohort, examples=counts[cohort]) for cohort in Cohort
        )
    test_path = (
        replay / "speech-projector/results_overnight_20261006"
        "/overnight_mean_2p5hz_updates4000_epoch1/test/evaluation_losses.jsonl"
    )
    paths.append(test_path)
    observations = tuple(
        ExampleLoss.model_validate_json(line) for line in test_path.read_bytes().splitlines()
    )
    counts = Counter(
        owners[row.example_id]
        for row in observations
        if row.condition == EvaluationCondition.SPEECH
    )
    panels.extend(
        PanelCount(
            scope="prior completed primary test CE panel", cohort=cohort, examples=counts[cohort]
        )
        for cohort in Cohort
    )
    generation_path = test_path.parent / "evaluation_generations.jsonl"
    paths.append(generation_path)
    generations = tuple(
        SampleGeneration.model_validate_json(line)
        for line in generation_path.read_bytes().splitlines()
    )
    counts = Counter(
        owners[row.example_id] for row in generations if row.condition == EvaluationCondition.SPEECH
    )
    panels.extend(
        PanelCount(
            scope="prior completed primary test generation panel",
            cohort=cohort,
            examples=counts[cohort],
        )
        for cohort in Cohort
    )
    return DatasetFacts(
        evidence=tuple(evidence(path) for path in paths),
        prepared=audit,
        cache=cache,
        panels=tuple(panels),
        full_train_target_tokens_including_eos=4_314_521,
        maximum_full_target_tokens_including_eos=1260,
        hardware=hardware,
        ordinary_cleaning=tuple(
            TeacherCleaningCounts(
                split=row.split,
                candidate_examples=row.candidate_examples,
                selected_examples=row.selected_examples,
                selected_distinct_dialogues=row.distinct_dialogues,
                material_alignment_exclusions=row.material_alignment_exclusions,
                lexical_substitution_exclusions=row.lexical_substitution_exclusions,
                missing_synthesis_text_exclusions=row.missing_synthesis_text_exclusions,
                collision_dialogues_excluded=len(row.collision_dialogue_exclusions),
            )
            for row in teacher_preparation.splits
        ),
        ordinary_transcript_reference=teacher_preparation.transcript_reference,
        ordinary_subset_definition=teacher_preparation.subset_definition,
        scope_notes=(
            "No frozen-model inference, audio transfer, new labels or training in this audit.",
            "Target token counts reuse actual Transformers 5.13.0/tokenizers 0.22.2 "
            "exposure evidence; per-cohort token quantiles/modes were not recorded "
            "and are unknown here.",
            "Native cache bytes include reused ordinary plus newly extracted emotional files; "
            "do not count all files as new extraction.",
            "Encoder extraction, cache wall and ASR timers have nested scopes; do not add them.",
            "Hardware values are point observations, not training peaks "
            "or allocated rental duration.",
            "Intended synthetic delivery labels are not human-verified emotional gold.",
            "Neu classifier and Neu audio are one Paul voice; vocoder/prosody signatures may help "
            "classification and do not establish natural-speech or speaker generalization.",
            "The production classifier has four synthetic intended classes: happy, sad, "
            "angry and fearful. It has no neutral/OOD class or calibrated abstention; "
            "perfect synthetic-label performance does not establish natural-emotion readiness.",
            "Teacher targets are model-generated; role/physical-action/tone mistakes "
            "remain preserved.",
            "Current ordinary inputs were selected after material alignment/substitution "
            "filters; historical raw V0 known mismatch rows are not deliberately "
            "carried into this clean ordinary cohort. Cleaning counters need not be "
            "mutually exclusive and must not be summed as unique excluded examples.",
            "Exact cross-cohort design families share one split; domains and pragmatic intents "
            "are intentionally reused, so these are not unseen-domain or unseen-intent tests.",
            "Prepared test bank and actual primary generation panel have separate denominators; "
            "the full-pass follow-up has now executed the same fixed TEST panel: 256 CE "
            "examples and 136 primary speech generations.",
        ),
    )


def render(facts: DatasetFacts) -> str:
    lines = [
        "# Dataset and hardware facts for the follow-up",
        "",
        "Source is the verified prior replay; inputs/supervision remain unchanged.",
        "",
        "| Cohort | Split | Examples | Audio mean/median/p90/max s | "
        "User words mean/median/p90/max | Teacher words mean/median/p90/max |",
        "|---|---|---:|---|---|---|",
    ]
    for row in facts.prepared.cohorts:
        cells = []
        for values in (row.duration_seconds, row.user_words, row.teacher_words):
            cells.append(
                f"{values.mean:.3f}/{values.median:.3f}/{values.p90:.3f}/{values.maximum:.3f}"
            )
        lines.append(
            f"| {row.cohort.value} | {row.split.value} | {row.examples} | "
            + " | ".join(cells)
            + " |"
        )
    lines.extend(("", "## Actual panels", ""))
    for row in facts.panels:
        lines.append(f"- {row.scope}: {row.cohort.value} {row.examples} examples.")
    lines.extend(
        (
            "",
            "## Ordinary clean-source construction",
            "",
            f"Reference: {facts.ordinary_transcript_reference}.",
            "Material original-turn versus audio-original lexical differences and lexical "
            "TTS substitutions were removed before Qwen teacher supervision was generated. "
            "The preparation's original targets were placeholders; current training targets "
            "come from its later completed teacher_examples.jsonl.",
            "",
            "| Split | Candidate examples | Selected | Dialogues | Material alignment "
            "exclusions | Lexical substitution exclusions | Missing synthesis text | "
            "Collision dialogues excluded |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in facts.ordinary_cleaning:
        lines.append(
            f"| {row.split.value} | {row.candidate_examples} | {row.selected_examples} | "
            f"{row.selected_distinct_dialogues} | {row.material_alignment_exclusions} | "
            f"{row.lexical_substitution_exclusions} | {row.missing_synthesis_text_exclusions} | "
            f"{row.collision_dialogues_excluded} |"
        )
    lines.extend(
        (
            "",
            facts.ordinary_subset_definition,
            "",
            "Exclusion counters describe candidate pools, not selected examples. "
            "Alignment and substitution counts may overlap. Collision counts are whole "
            "dialogues in each candidate pool. A later combined-copy normalized prompt "
            "audit removed one additional TRAIN dialogue/example; original source assets "
            "remain preserved. Cleaned synthesis text is a documented generation reference, "
            "not an independently verified waveform transcript; occasional synthesis/ASR "
            "anomalies remain possible.",
        )
    )
    cache = facts.cache
    hardware = facts.hardware
    lines.extend(
        (
            "",
            "## Cache and training targets",
            "",
            f"- Raw native cache: {cache.feature_count:,} files / {cache.feature_bytes:,} bytes "
            f"({cache.feature_bytes / 1e9:.3f} GB decimal); "
            f"{cache.total_audio_seconds:,.3f} audio s.",
            f"- Incremental extraction: {cache.extracted_count:,} new files / "
            f"{cache.extracted_audio_seconds:,.3f} audio s; synchronized encoder scope "
            f"{cache.extraction_seconds:.3f} s, "
            f"overall cache wall {cache.cache_wall_seconds:.3f} s, "
            f"nested heldout ASR {cache.asr_seconds:.3f} s.",
            "- Final normalized Whisper Small states: 768 dimensions, 50 Hz, BF16, "
            "76,800 bytes/audio second before serialization overhead. Encoder attends standard "
            "30-second padded log-mels; retained ceil(actual seconds×50) states "
            "define downstream masks.",
            f"- One full training pass: {facts.full_train_target_tokens_including_eos:,} target "
            "tokens including EOS over 38,193 examples; maximum stored target 1260 tokens, "
            "configured cap 4097. Target token modes/quantiles per cohort "
            "are unknown in saved evidence.",
            "- Ordinary history: TRAIN 7251 empty/12748 two-turn; VAL 202/310; TEST 184/328. "
            "Both emotional cohorts have zero history turns.",
            "- Old paired identical targets: 112 exact/125 normalized of 4997 remaining pairs; "
            "Neu 93 exact/98 normalized of 5000 pairs. "
            "Equality is diagnostic, not a rejection criterion.",
            "- Excluded only in combined copy: 3 over-30-second old pairs = 6 clips, "
            "and 1 ordinary TRAIN "
            "dialogue/example protecting normalized test-prompt overlap. Original files preserved.",
            "",
            "## Point-in-time hardware observation",
            "",
            f"At {hardware.measured_at.isoformat()}: {hardware.gpu_name}, "
            f"{hardware.gpu_memory_mib} MiB VRAM, driver {hardware.nvidia_driver}.",
            f"Workspace filesystem capacity {hardware.workspace_capacity_bytes:,} bytes "
            f"({hardware.workspace_capacity_bytes / 2**30:.1f} GiB); free "
            f"{hardware.workspace_free_bytes:,} bytes "
            f"({hardware.workspace_free_bytes / 1e9:.3f} GB).",
            f"Container limit {hardware.container_memory_limit_bytes:,} bytes "
            f"({hardware.container_memory_limit_bytes / 2**30:.3f} GiB), from exact "
            f"`{hardware.container_memory_limit_path}`. Charged usage "
            f"{hardware.container_memory_charged_bytes:,} bytes includes filesystem cache.",
            f"Host counters separately: total {hardware.host_memory_bytes:,} bytes, "
            f"available {hardware.host_memory_available_bytes:,} bytes; "
            f"swap {hardware.swap_bytes}. "
            "Do not add host available memory to the container allocation limit.",
            "PyTorch allocated peaks, reserved memory and NVIDIA/NVML observed usage are "
            "different measurements; current facts do not substitute one for another.",
            "",
            "## Limits and scope",
            "",
            *(f"- {note}" for note in facts.scope_notes),
            "",
            "## Hashed source evidence",
            "",
            *(
                f"- `{entry.path}`: {entry.bytes:,} bytes; SHA256 `{entry.sha256}`."
                for entry in facts.evidence
            ),
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    hardware = HardwareFacts.model_validate_json(arguments.hardware.read_bytes())
    facts = build_facts(arguments.replay, hardware)
    write_record(arguments.output, facts)
    arguments.output.with_suffix(".md").write_text(render(facts), encoding="utf-8")


if __name__ == "__main__":
    main()
