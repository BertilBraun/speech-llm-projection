"""Read-only audit of completed teacher targets and fixed conversational examples."""

import argparse
import hashlib
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from scripts.package_results import FileArtifact
from speech_projector.data import Distribution, distribution
from speech_projector.models import Example, Record, Split
from speech_projector.teacher import (
    TeacherFailure,
    TeacherProgress,
    TeacherProvenance,
    TeacherTarget,
)


@dataclass(frozen=True)
class TargetAuditConfig:
    journal_directory: Path
    source_manifest: Path
    output_directory: Path


class DomainSplitCount(Record):
    split: Split
    domain: str
    examples: int


class SplitCount(Record):
    split: Split
    examples: int


class ResponseLengths(Record):
    words: Distribution
    target_tokens_with_eos: Distribution


class HistoryLengths(Record):
    selected_turns: Distribution
    selected_text_tokens: Distribution
    current_user_tokens: Distribution
    empty_history_examples: int


class HistoryGroup(str, Enum):
    EMPTY = "empty"
    PRESENT = "present"


class HistoryResponseLengths(Record):
    history: HistoryGroup
    examples: int
    teacher: ResponseLengths


class TargetIssueKind(str, Enum):
    MISSING_FINAL_EOS = "missing_final_eos"
    EARLY_EOS = "early_eos"
    EMPTY_RESPONSE = "empty_response"
    OVER_TARGET_BUDGET = "over_target_budget"
    SOURCE_MISMATCH = "source_mismatch"
    DUPLICATE_IDENTIFIER = "duplicate_identifier"


class TargetIssue(Record):
    example_id: str
    kind: TargetIssueKind


class TeacherTargetAudit(Record):
    provenance: TeacherProvenance
    input_journal: FileArtifact
    source_manifest: FileArtifact
    source_examples: int
    completed_examples: int
    incomplete_suffix_bytes: int
    splits: tuple[SplitCount, ...]
    domains: tuple[DomainSplitCount, ...]
    original_dataset_response: ResponseLengths
    teacher_response: ResponseLengths
    generated_tokens_with_eos: Distribution
    history: HistoryLengths
    history_groups: tuple[HistoryResponseLengths, ...]
    completed_without_retry: int
    completed_after_retry: int
    eos_completed_examples: int
    issues: tuple[TargetIssue, ...]
    failures: tuple[TeacherFailure, ...]
    progress: tuple[TeacherProgress, ...]
    fixed_samples: tuple[TeacherTarget, ...]
    token_count_definition: str
    history_count_definition: str


def artifact(path: Path, content: bytes) -> FileArtifact:
    return FileArtifact(
        path=path,
        source_path=path,
        bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def response_lengths(texts: tuple[str, ...], tokenizer: PreTrainedTokenizerBase) -> ResponseLengths:
    return ResponseLengths(
        words=distribution([float(len(text.split())) for text in texts]),
        target_tokens_with_eos=distribution(
            [float(len(tokenizer.encode(text, add_special_tokens=False)) + 1) for text in texts]
        ),
    )


def fixed_samples(
    targets: tuple[TeacherTarget, ...], sources: tuple[Example, ...]
) -> tuple[TeacherTarget, ...]:
    desired: list[str] = []
    for split, count in ((Split.TRAIN, 3), (Split.VALIDATION, 3), (Split.TEST, 4)):
        desired.extend(
            example.example_id
            for example in tuple(item for item in sources if item.split == split)[:count]
        )
    by_identifier = {target.example.example_id: target for target in targets}
    return tuple(by_identifier[identifier] for identifier in desired if identifier in by_identifier)


def completed_lines(content: bytes) -> tuple[tuple[bytes, ...], int]:
    complete = content.rfind(b"\n") + 1
    return tuple(content[:complete].splitlines()), len(content) - complete


def summarize_targets(
    configuration: TargetAuditConfig, tokenizer: PreTrainedTokenizerBase
) -> TeacherTargetAudit:
    provenance = TeacherProvenance.model_validate_json(
        (configuration.journal_directory / "provenance.json").read_bytes()
    )
    source_content = configuration.source_manifest.read_bytes()
    source_artifact = artifact(configuration.source_manifest, source_content)
    if source_artifact.sha256 != provenance.input_manifest.sha256:
        raise ValueError("Audit source manifest differs from the generation provenance")
    sources = tuple(Example.model_validate_json(line) for line in source_content.splitlines())
    source_by_identifier = {example.example_id: example for example in sources}
    if len(source_by_identifier) != len(sources):
        raise ValueError("Source manifest has duplicate identifiers")
    journal_path = configuration.journal_directory / "targets.jsonl"
    journal_content = journal_path.read_bytes()
    lines, incomplete_bytes = completed_lines(journal_content)
    targets = tuple(TeacherTarget.model_validate_json(line) for line in lines)
    if not targets:
        raise ValueError("No completed teacher targets are available for a length audit")
    if tokenizer.eos_token_id is None:
        raise ValueError("Audit tokenizer must define the actual teacher EOS")
    issues: list[TargetIssue] = []
    identifiers: set[str] = set()
    completed_eos = 0
    for target in targets:
        identifier = target.example.example_id
        if identifier in identifiers:
            issues.append(
                TargetIssue(example_id=identifier, kind=TargetIssueKind.DUPLICATE_IDENTIFIER)
            )
        identifiers.add(identifier)
        if (
            identifier not in source_by_identifier
            or target.example != source_by_identifier[identifier]
        ):
            issues.append(TargetIssue(example_id=identifier, kind=TargetIssueKind.SOURCE_MISMATCH))
        if not target.response.token_ids or target.response.token_ids[-1] != tokenizer.eos_token_id:
            issues.append(
                TargetIssue(example_id=identifier, kind=TargetIssueKind.MISSING_FINAL_EOS)
            )
        else:
            completed_eos += 1
        if tokenizer.eos_token_id in target.response.token_ids[:-1]:
            issues.append(TargetIssue(example_id=identifier, kind=TargetIssueKind.EARLY_EOS))
        if not target.response.text.strip():
            issues.append(TargetIssue(example_id=identifier, kind=TargetIssueKind.EMPTY_RESPONSE))
        tokens = len(tokenizer.encode(target.response.text, add_special_tokens=False)) + 1
        if tokens > provenance.config.run.max_target_tokens:
            issues.append(
                TargetIssue(example_id=identifier, kind=TargetIssueKind.OVER_TARGET_BUDGET)
            )
    split_counts = Counter(target.example.split for target in targets)
    domain_counts = Counter((target.example.split, target.example.domain) for target in targets)
    selected_history = tuple(
        target.example.history[-provenance.config.run.history_turns :]
        if provenance.config.run.history_turns
        else ()
        for target in targets
    )
    groups: list[HistoryResponseLengths] = []
    for history_group in HistoryGroup:
        texts = tuple(
            target.response.text
            for target, history in zip(targets, selected_history, strict=True)
            if bool(history) == (history_group == HistoryGroup.PRESENT)
        )
        if texts:
            groups.append(
                HistoryResponseLengths(
                    history=history_group,
                    examples=len(texts),
                    teacher=response_lengths(texts, tokenizer),
                )
            )
    failure_path = configuration.journal_directory / "failures.jsonl"
    failures = (
        tuple(
            TeacherFailure.model_validate_json(line)
            for line in completed_lines(failure_path.read_bytes())[0]
        )
        if failure_path.exists()
        else ()
    )
    progress_path = configuration.journal_directory / "progress.jsonl"
    progress = (
        tuple(
            TeacherProgress.model_validate_json(line)
            for line in completed_lines(progress_path.read_bytes())[0]
        )
        if progress_path.exists()
        else ()
    )
    retried = sum(bool(target.capped_attempts) for target in targets)
    return TeacherTargetAudit(
        provenance=provenance,
        input_journal=artifact(journal_path, journal_content),
        source_manifest=source_artifact,
        source_examples=len(sources),
        completed_examples=len(targets),
        incomplete_suffix_bytes=incomplete_bytes,
        splits=tuple(SplitCount(split=split, examples=split_counts[split]) for split in Split),
        domains=tuple(
            DomainSplitCount(split=split, domain=domain, examples=count)
            for (split, domain), count in sorted(
                domain_counts.items(), key=lambda entry: (entry[0][0].value, entry[0][1])
            )
        ),
        original_dataset_response=response_lengths(
            tuple(target.example.target_text for target in targets), tokenizer
        ),
        teacher_response=response_lengths(
            tuple(target.response.text for target in targets), tokenizer
        ),
        generated_tokens_with_eos=distribution(
            [float(len(target.response.token_ids)) for target in targets]
        ),
        history=HistoryLengths(
            selected_turns=distribution([float(len(history)) for history in selected_history]),
            selected_text_tokens=distribution(
                [
                    float(
                        sum(
                            len(tokenizer.encode(turn.text, add_special_tokens=False))
                            for turn in history
                        )
                    )
                    for history in selected_history
                ]
            ),
            current_user_tokens=distribution(
                [
                    float(len(tokenizer.encode(target.example.user_text, add_special_tokens=False)))
                    for target in targets
                ]
            ),
            empty_history_examples=sum(not history for history in selected_history),
        ),
        history_groups=tuple(groups),
        completed_without_retry=len(targets) - retried,
        completed_after_retry=retried,
        eos_completed_examples=completed_eos,
        issues=tuple(issues),
        failures=failures,
        progress=progress,
        fixed_samples=fixed_samples(targets, sources),
        token_count_definition=(
            "Re-encoded response text plus one EOS; original dataset responses are measured "
            "before any historical training truncation. Saved generated token IDs include "
            "EOS separately."
        ),
        history_count_definition=(
            "Last configured history_turns, before model history-token budget truncation; "
            "text tokens exclude role delimiters. This measures source history selection, "
            "not exact final model context length."
        ),
    )


def render_audit(audit: TeacherTargetAudit) -> str:
    lines = [
        "# Teacher target audit",
        "",
        f"{audit.completed_examples}/{audit.source_examples} source examples completed; "
        f"{audit.eos_completed_examples} end in EOS, {audit.completed_after_retry} needed retry, "
        f"{len(audit.issues)} checks failed, {len(audit.failures)} failure records retained.",
        "",
        "| Response | Median target tokens | p90 | p99 | Maximum | Mean words |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, lengths in (
        ("Original dataset", audit.original_dataset_response),
        ("Frozen Qwen", audit.teacher_response),
    ):
        tokens = lengths.target_tokens_with_eos
        lines.append(
            f"| {name} | {tokens.median:.1f} | {tokens.p90:.1f} | {tokens.p99:.1f} | "
            f"{tokens.maximum:.0f} | {lengths.words.mean:.1f} |"
        )
    lines.extend(("", audit.token_count_definition, "", audit.history_count_definition, ""))
    for sample in audit.fixed_samples:
        example = sample.example
        lines.extend((f"## {example.example_id} — {example.split.value}/{example.domain}", ""))
        for turn in (
            example.history[-audit.provenance.config.run.history_turns :]
            if audit.provenance.config.run.history_turns
            else ()
        ):
            lines.extend((f"**History {turn.role.value}:** {turn.text}", ""))
        lines.extend(
            (
                f"**True user text:** {example.user_text}",
                "",
                f"**Original dataset response:** {example.target_text}",
                "",
                f"**Frozen Qwen response:** {sample.response.text}",
                "",
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal-directory", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = TargetAuditConfig(
        arguments.journal_directory, arguments.source_manifest, arguments.output
    )
    provenance = TeacherProvenance.model_validate_json(
        (configuration.journal_directory / "provenance.json").read_bytes()
    )
    if provenance.model_revision.main_revision is None:
        raise ValueError("Teacher tokenizer revision is absent from generation provenance")
    tokenizer = AutoTokenizer.from_pretrained(
        provenance.config.run.model_name,
        revision=provenance.model_revision.main_revision,
        local_files_only=True,
    )
    audit = summarize_targets(configuration, tokenizer)
    configuration.output_directory.mkdir(parents=True, exist_ok=True)
    (configuration.output_directory / "target_audit.json").write_text(
        audit.model_dump_json(indent=2), encoding="utf-8"
    )
    (configuration.output_directory / "target_audit.md").write_text(
        render_audit(audit), encoding="utf-8"
    )
    print(
        f"Audited {audit.completed_examples}/{audit.source_examples} completed targets; "
        f"EOS={audit.eos_completed_examples}, retries={audit.completed_after_retry}, "
        f"issues={len(audit.issues)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
