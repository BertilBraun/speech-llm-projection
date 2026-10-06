import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from scripts.package_results import FileArtifact, ModelRevision
from scripts.summarize_teacher_targets import (
    TargetAuditConfig,
    TargetIssueKind,
    fixed_samples,
    render_audit,
    summarize_targets,
)
from speech_projector.generation import CompletedGeneration, TokenLimitedGeneration
from speech_projector.models import Example, Role, Split, Turn
from speech_projector.teacher import (
    TeacherConfig,
    TeacherFailure,
    TeacherFailureReason,
    TeacherProvenance,
    TeacherTarget,
)
from speech_projector.teacher_configuration import teacher_compression_runs


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(WordLevel({"[UNK]": 0, "yes": 1, "no": 2, "[EOS]": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]")


def example(index: int, split: Split) -> Example:
    return Example(
        example_id=f"{split.value}-{index}",
        dialogue_id=f"dialogue-{split.value}-{index}",
        split=split,
        history=()
        if index == 0
        else (Turn(role=Role.USER, text="no no"), Turn(role=Role.ASSISTANT, text="yes")),
        user_text="no",
        target_text="no",
        audio_path=Path("audio.wav"),
        duration=1,
        domain="science" if index % 2 else "fashion",
        emotion="neutral",
        feature_path=Path("feature.pt"),
    )


def create_inputs(directory: Path, targets: tuple[TeacherTarget, ...]) -> TargetAuditConfig:
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "source.jsonl"
    content = "".join(target.example.model_dump_json() + "\n" for target in targets).encode()
    source.write_bytes(content)
    config = TeacherConfig(
        run=teacher_compression_runs()[0].model_copy(update={"max_target_tokens": 4}),
        manifest=source,
        output_directory=directory,
    )
    provenance = TeacherProvenance(
        config=config,
        source_commit="test-source",
        input_manifest=FileArtifact(
            path=source,
            source_path=source,
            bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        ),
        model_revision=ModelRevision(
            model_name=config.run.model_name,
            snapshot_revisions=("test-revision",),
            main_revision="test-revision",
        ),
        frozen_parameter_sha256="test-weights",
        started_at=datetime(2026, 10, 6, 8, tzinfo=timezone.utc),
    )
    (directory / "provenance.json").write_text(provenance.model_dump_json())
    (directory / "targets.jsonl").write_text(
        "".join(target.model_dump_json() + "\n" for target in targets)
    )
    return TargetAuditConfig(directory, source, directory / "audit")


def target(index: int, split: Split = Split.TRAIN) -> TeacherTarget:
    return TeacherTarget(
        example=example(index, split),
        response=CompletedGeneration(text="yes yes yes", token_ids=(1, 1, 1, 3)),
        capped_attempts=(),
    )


def test_audit_compares_lengths_counts_history_and_fixed_readable_examples(
    tmp_path: Path, tokenizer: PreTrainedTokenizerFast
) -> None:
    targets = tuple(target(index, split) for split in Split for index in range(4))
    config = create_inputs(tmp_path, targets)
    audit = summarize_targets(config, tokenizer)
    assert audit.completed_examples == 12
    assert audit.eos_completed_examples == 12
    assert audit.issues == ()
    assert tuple(group.examples for group in audit.splits) == (4, 4, 4)
    assert audit.original_dataset_response.target_tokens_with_eos.median == 2
    assert audit.teacher_response.target_tokens_with_eos.median == 4
    assert audit.history.empty_history_examples == 3
    assert len(audit.fixed_samples) == 10
    report = render_audit(audit)
    assert "History user:" in report
    assert "Original dataset response:" in report
    assert "Frozen Qwen response:" in report


def test_live_snapshot_ignores_partial_tail_without_modifying_journal(
    tmp_path: Path, tokenizer: PreTrainedTokenizerFast
) -> None:
    config = create_inputs(tmp_path, (target(0),))
    path = tmp_path / "targets.jsonl"
    with path.open("ab") as output:
        output.write(b'{"unfinished":')
    original = path.read_bytes()
    audit = summarize_targets(config, tokenizer)
    assert audit.incomplete_suffix_bytes == len(b'{"unfinished":')
    assert audit.completed_examples == 1
    assert path.read_bytes() == original
    assert not (tmp_path / "journal_recovery.jsonl").exists()


@pytest.mark.parametrize(
    ("response", "issue"),
    [
        (CompletedGeneration(text="yes", token_ids=(1,)), TargetIssueKind.MISSING_FINAL_EOS),
        (CompletedGeneration(text="yes", token_ids=(3, 1, 3)), TargetIssueKind.EARLY_EOS),
        (CompletedGeneration(text="", token_ids=(3,)), TargetIssueKind.EMPTY_RESPONSE),
        (
            CompletedGeneration(text="yes yes yes yes", token_ids=(1, 1, 1, 1, 3)),
            TargetIssueKind.OVER_TARGET_BUDGET,
        ),
    ],
)
def test_eos_empty_and_budget_issues_are_explicit(
    tmp_path: Path,
    tokenizer: PreTrainedTokenizerFast,
    response: CompletedGeneration,
    issue: TargetIssueKind,
) -> None:
    selected = target(0).model_copy(update={"response": response})
    audit = summarize_targets(create_inputs(tmp_path, (selected,)), tokenizer)
    assert issue in tuple(record.kind for record in audit.issues)


def test_failure_and_retry_are_retained_without_creating_partial_targets(
    tmp_path: Path, tokenizer: PreTrainedTokenizerFast
) -> None:
    capped = TokenLimitedGeneration(partial_text="no", token_ids=(2,))
    selected = target(0).model_copy(update={"capped_attempts": (capped,)})
    config = create_inputs(tmp_path, (selected,))
    failure = TeacherFailure(
        example=selected.example, reason=TeacherFailureReason.TOKEN_LIMIT, attempts=(capped, capped)
    )
    (tmp_path / "failures.jsonl").write_text(failure.model_dump_json() + "\n")
    audit = summarize_targets(config, tokenizer)
    assert audit.completed_after_retry == 1
    assert audit.completed_examples == 1
    assert audit.failures == (failure,)


def test_fixed_examples_remain_identical_when_journal_grows() -> None:
    sources = tuple(example(index, split) for split in Split for index in range(6))
    bootstrap = tuple(target(index, split) for split in Split for index in range(4))
    full = tuple(target(index, split) for split in Split for index in range(6))
    assert fixed_samples(bootstrap, sources) == fixed_samples(full, sources)


def test_changed_source_hash_is_rejected(
    tmp_path: Path, tokenizer: PreTrainedTokenizerFast
) -> None:
    config = create_inputs(tmp_path, (target(0),))
    config.source_manifest.write_text("changed source")
    with pytest.raises(ValueError, match="generation provenance"):
        summarize_targets(config, tokenizer)
