"""Completed reference reuse must bind actual inputs, all control modes and immutable sources."""

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
import torch
from pydantic import TypeAdapter
from safetensors.torch import save_file

from scripts.package_results import EnvironmentInventory, InstalledPackage, ModelRevision
from speech_projector.cache import AdditionalAudioFeature, CacheStatistics
from speech_projector.conversation_reuse import (
    INFERENCE_FILES,
    ConversationReuseConfig,
    complete_records,
    inference_identity,
    merge_conversation_references,
    verify_reply_coverage,
)
from speech_projector.followup_conversation import (
    FollowupAsrConfig,
    FollowupAsrProvenance,
    FollowupConversationConfig,
    FollowupConversationProvenance,
    PipelineConversationReply,
    evaluate_conversation_baseline,
)
from speech_projector.followup_conversation_tone import (
    PredictedInitialToneConversationConfig,
    evaluate_predicted_initial_tone,
    prepare_conversation_tone,
)
from speech_projector.followup_evaluation import file_artifact
from speech_projector.llm import FrozenQwen
from speech_projector.models import AsrTranscript, EvaluationCondition, OrdinaryResponseKLObjective
from speech_projector.overnight_conversation import DelayedCueFixtures
from speech_projector.overnight_conversation_execution import (
    SavedConversationReply,
    evaluate_conversations,
)
from speech_projector.projectors import Projector
from tests.test_followup_conversation_tone import (
    classifier_report,
    fixtures,
    predictions,
    recognized_words,
)
from tests.test_overnight_conversation_execution import ConversationWrapper
from tests.test_overnight_evaluation import candidate
from tests.test_overnight_report import result


def reuse_fixture(directory: Path, monkeypatch: pytest.MonkeyPatch) -> ConversationReuseConfig:
    original = fixtures()
    audio = directory / "audio"
    audio.mkdir()
    feature_root = directory / "features"
    feature_root.mkdir()
    scenarios = []
    for scenario in original.scenarios:
        initial = []
        followups = []
        for row in scenario.initial:
            waveform = audio / f"{row.example_id}.wav"
            waveform.write_bytes(b"Test-local waveform input fixture")
            features = feature_root / f"{row.example_id}.pt"
            torch.save(torch.ones(2, 768, dtype=torch.bfloat16), features)
            initial.append(
                row.model_copy(update={"audio_path": waveform, "feature_path": features})
            )
        for row in scenario.followups:
            waveform = audio / row.clip.audio_path
            waveform.write_bytes(b"Test-local neutral waveform input fixture")
            features = feature_root / row.feature_path.name
            torch.save(torch.ones(2, 768, dtype=torch.bfloat16), features)
            receipt = AdditionalAudioFeature(
                audio=file_artifact(waveform),
                features=file_artifact(features),
                audio_seconds=0.04,
                valid_frames=2,
            )
            features.with_suffix(".receipt.json").write_text(
                receipt.model_dump_json(), encoding="utf-8"
            )
            followups.append(
                row.model_copy(
                    update={
                        "feature_path": features,
                        "clip": row.clip.model_copy(
                            update={"sha256": file_artifact(waveform).sha256}
                        ),
                    }
                )
            )
        scenarios.append(
            scenario.model_copy(update={"initial": tuple(initial), "followups": tuple(followups)})
        )
    panel = original.model_copy(update={"scenarios": tuple(scenarios)})
    fixture_path = directory / "fixtures.json"
    fixture_path.write_text(panel.model_dump_json(), encoding="utf-8")
    source = directory / "source"
    source.mkdir()
    for relative in INFERENCE_FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# Test-local frozen source identity fixture\n")
    subprocess.check_call(["git", "-C", str(source), "init", "--quiet"])
    subprocess.check_call(["git", "-C", str(source), "add", "."])
    subprocess.check_call(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Fixture",
        ]
    )
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    base = candidate("reference", 1, 1, 0.1, 10).configuration.model_copy(
        update={
            "history_turns": 6,
            "max_history_tokens": 8192,
            "max_new_tokens": 256,
            "max_optimizer_updates": 4775,
        }
    )
    selected_config = base.model_copy(
        update={
            "name": "selected",
            "max_optimizer_updates": 6775,
            "learning_rate": 2e-4,
            "objective": OrdinaryResponseKLObjective(),
        }
    )
    projector = Projector(base.projector)
    parent_checkpoint = directory / "reference.safetensors"
    save_file(projector.state_dict(), parent_checkpoint)
    reference_result_path = directory / "reference_result.json"
    reference_result_path.write_text(
        result()
        .model_copy(
            update={
                "config": base,
                "steps": 4775,
                "checkpoint_path": parent_checkpoint,
                "git_commit": commit,
            }
        )
        .model_dump_json(),
        encoding="utf-8",
    )
    recognized = recognized_words(panel)
    asr_path = directory / "asr.jsonl"
    asr_path.write_bytes(b"".join(row.model_dump_json().encode() + b"\n" for row in recognized[:6]))
    reference_directory = directory / "reference_conversation"
    followup_directory = reference_directory / "followup_asr"
    followup_directory.mkdir(parents=True)
    followup_path = followup_directory / "transcripts.jsonl"
    followup_path.write_bytes(
        b"".join(row.model_dump_json().encode() + b"\n" for row in recognized[6:])
    )
    whisper_revision = "b" * 40
    reference_config = FollowupConversationConfig(
        run_result=reference_result_path,
        fixtures=fixture_path,
        asr_transcripts=asr_path,
        followup_asr=FollowupAsrConfig(revision=whisper_revision, audio_root=audio),
        output_directory=reference_directory,
        source_git_commit=commit,
    )
    reference_proof = FollowupConversationProvenance(
        configuration=reference_config,
        inputs=tuple(
            file_artifact(path)
            for path in (reference_result_path, parent_checkpoint, fixture_path, asr_path)
        ),
    )
    (reference_directory / "provenance.json").write_text(
        reference_proof.model_dump_json(), encoding="utf-8"
    )
    (followup_directory / "provenance.json").write_text(
        FollowupAsrProvenance(
            configuration=reference_config.followup_asr,
            audio=tuple(
                file_artifact(audio / row.clip.audio_path) for row in panel.scenarios[0].followups
            ),
            fixtures_sha256=hashlib.sha256(panel.model_dump_json().encode()).hexdigest(),
        ).model_dump_json(),
        encoding="utf-8",
    )
    wrapper = ConversationWrapper(base)
    for condition in (EvaluationCondition.TEXT, EvaluationCondition.ASR):
        evaluate_conversation_baseline(
            cast(FrozenQwen, wrapper),
            panel,
            condition,
            recognized,
            reference_directory / condition.value,
            source_git_commit=commit,
        )
    with torch.no_grad():
        next(projector.parameters()).add_(0.01)
    selected_checkpoint = directory / "selected.safetensors"
    save_file(projector.state_dict(), selected_checkpoint)
    selected_result_path = directory / "selected_result.json"
    selected_result_path.write_text(
        result()
        .model_copy(
            update={
                "config": selected_config,
                "steps": 6775,
                "checkpoint_path": selected_checkpoint,
                "git_commit": commit,
            }
        )
        .model_dump_json(),
        encoding="utf-8",
    )
    speech_directory = directory / "selected_speech"
    evaluate_conversations(
        cast(FrozenQwen, ConversationWrapper(selected_config)),
        projector,
        panel,
        speech_directory,
        source_commit=commit,
    )
    prediction_path = directory / "predictions.jsonl"
    report = classifier_report(prediction_path, predictions(panel))
    report_path = directory / "classifier.json"
    report_path.write_text(report.model_dump_json(), encoding="utf-8")
    tone_directory = directory / "tone_conversation"
    tone_config = PredictedInitialToneConversationConfig(
        run_result=reference_result_path,
        fixtures=fixture_path,
        asr_transcripts=asr_path,
        followup_transcripts=followup_path,
        predictions=prediction_path,
        classifier_report=report_path,
        output_directory=tone_directory,
        source_git_commit=commit,
    )
    tone_panel, tone_proof = prepare_conversation_tone(tone_config)
    evaluate_predicted_initial_tone(
        cast(FrozenQwen, ConversationWrapper(base)), tone_panel, tone_proof
    )
    hub = directory / "hub"
    revisions = []
    for name, revision in ((base.model_name, "a" * 40), (base.speech_model_name, whisper_revision)):
        repository = hub / ("models--" + name.replace("/", "--"))
        (repository / "refs").mkdir(parents=True)
        (repository / "refs" / "main").write_text(revision, encoding="utf-8")
        snapshot = repository / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "model.safetensors").write_bytes(b"Test-local snapshot identity fixture")
        revisions.append(
            ModelRevision(model_name=name, snapshot_revisions=(revision,), main_revision=revision)
        )
    revisions_path = directory / "model_revisions.jsonl"
    revisions_path.write_bytes(
        b"".join(row.model_dump_json().encode() + b"\n" for row in revisions)
    )
    environment = EnvironmentInventory(
        python_version=sys.version.split()[0],
        python_implementation="CPython",
        python_executable=Path(sys.executable),
        operating_system="test",
        machine="test",
        packages=tuple(
            InstalledPackage(name=name, version="fixture")
            for name in (
                "torch",
                "transformers",
                "safetensors",
                "triton",
                "flash-linear-attention",
                "causal-conv1d",
            )
        ),
    )
    environment_path = directory / "environment.json"
    environment_path.write_text(environment.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr("speech_projector.conversation_reuse.version", lambda name: "fixture")
    cache_path = directory / "cache.json"
    cache_path.write_text(
        CacheStatistics(
            encoder_model="openai/whisper-small",
            hidden_dimension=768,
            native_states_per_second=50,
            selected_hidden_state="final encoder",
            storage_dtype="bfloat16",
            bytes_per_audio_second=76800,
            feature_count=9,
            extracted_count=9,
            total_audio_seconds=1,
            extracted_audio_seconds=1,
            feature_bytes=100,
            extraction_seconds=1,
            cache_wall_seconds=1,
            cache_examples_per_second=9,
            extraction_audio_seconds_per_second=1,
            asr_seconds=1,
            peak_vram_gb=1,
            masking="valid frames",
        ).model_dump_json(),
        encoding="utf-8",
    )
    return ConversationReuseConfig(
        selected_run_result=selected_result_path,
        selected_speech_directory=speech_directory,
        reference_directory=reference_directory,
        predicted_tone_directory=tone_directory,
        fixtures=fixture_path,
        cache_statistics=cache_path,
        model_revisions=revisions_path,
        environment_inventory=environment_path,
        hugging_face_hub=hub,
        source_directory=source,
        git_repository=source,
        output_directory=directory / "merged",
    )


def test_merge_preserves_all_controls_and_sources_and_keeps_tone_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = reuse_fixture(tmp_path, monkeypatch)
    sources = tuple(
        path
        for root in (
            configuration.reference_directory,
            configuration.predicted_tone_directory,
            configuration.selected_speech_directory,
        )
        for path in root.rglob("*")
        if path.is_file()
    )
    before = tuple(file_artifact(path) for path in sources)
    proof = merge_conversation_references(configuration)
    assert (
        proof.speech_responses,
        proof.text_responses,
        proof.asr_responses,
        proof.tone_responses,
    ) == (60, 36, 36, 36)
    merged = complete_records(proof.merged.path, TypeAdapter(PipelineConversationReply))
    tone = complete_records(proof.separate_tone.path, TypeAdapter(PipelineConversationReply))
    assert len(merged) == 132 and len(tone) == 36
    assert sum(row.condition == EvaluationCondition.SPEECH for row in merged) == 60
    assert sum(row.reply.kind == "controlled" for row in merged) == 60
    assert sum(row.reply.kind == "rollout" for row in merged) == 72
    assert all("delivery metadata" not in row.user_words[0] for row in merged)
    assert all("delivery metadata: sad" in row.user_words[0] for row in tone)
    assert tuple(file_artifact(path) for path in sources) == before
    assert len(proof.models) == 2
    assert (
        proof.selected_provenance.configuration.objective != proof.reference_configuration.objective
    )
    with pytest.raises(ValueError, match="new unsealed"):
        merge_conversation_references(configuration)


@pytest.mark.parametrize(
    "change", ("words", "history", "prediction_bytes", "source", "snapshot", "cache")
)
def test_merge_rejects_real_identity_drift_without_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    configuration = reuse_fixture(tmp_path, monkeypatch)
    if change in ("words", "history"):
        path = configuration.reference_directory / "asr" / "replies.jsonl"
        rows = complete_records(path, TypeAdapter(PipelineConversationReply))
        first = rows[0]
        if change == "words":
            first = first.model_copy(
                update={"user_words": ("Wrong initial words.",) + first.user_words[1:]}
            )
        else:
            first = first.model_copy(
                update={"reply": first.reply.model_copy(update={"history_for_judge": ()})}
            )
        path.write_bytes(
            b"".join(row.model_dump_json().encode() + b"\n" for row in (first, *rows[1:]))
        )
    elif change == "prediction_bytes":
        (tmp_path / "predictions.jsonl").write_bytes(b"unbound labels\n")
    elif change == "source":
        (configuration.source_directory / INFERENCE_FILES[0]).write_bytes(b"changed inference\n")
    elif change == "snapshot":
        (configuration.hugging_face_hub / "models--Qwen--Qwen3.5-2B" / "refs" / "main").write_text(
            "c" * 40, encoding="utf-8"
        )
    else:
        panel = DelayedCueFixtures.model_validate_json(configuration.fixtures.read_bytes())
        panel.scenarios[0].followups[0].feature_path.write_bytes(b"changed cached states")
    with pytest.raises(ValueError):
        merge_conversation_references(configuration)
    assert not configuration.output_directory.exists()


def test_incomplete_source_journal_is_not_repaired(tmp_path: Path) -> None:
    path = tmp_path / "transcripts.jsonl"
    original = AsrTranscript(example_id="one", text="recognized").model_dump_json().encode()
    path.write_bytes(original)
    with pytest.raises(ValueError, match="incomplete suffix"):
        complete_records(path, TypeAdapter(AsrTranscript))
    assert path.read_bytes() == original


def test_reply_coverage_failure_is_explicit_and_policy_seed_is_material(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = reuse_fixture(tmp_path, monkeypatch)
    panel = DelayedCueFixtures.model_validate_json(configuration.fixtures.read_bytes())
    replies = complete_records(
        configuration.selected_speech_directory / "replies.jsonl",
        TypeAdapter(SavedConversationReply),
    )
    with pytest.raises(ValueError, match="coverage"):
        verify_reply_coverage(replies[:-1], panel, speech=True)
    inference = candidate("identity", 1, 1, 0.1, 10).configuration.model_copy(
        update={
            "history_turns": 6,
            "max_history_tokens": 8192,
            "max_new_tokens": 256,
        }
    )
    assert inference_identity(inference, panel) != inference_identity(
        inference.model_copy(update={"seed": 43}), panel
    )
