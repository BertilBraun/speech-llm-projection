"""Non-generating, hash-bound merge of a selected speech conversation and frozen references."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import TypeVar

from pydantic import TypeAdapter, field_serializer
from safetensors.torch import load_file

from scripts.inventory_results import stable_digest, write_record
from scripts.package_results import EnvironmentInventory, ModelRevision
from speech_projector.cache import AdditionalAudioFeature, CacheStatistics
from speech_projector.followup_conversation import (
    ConversationBaselineProvenance,
    FollowupAsrProvenance,
    FollowupConversationProvenance,
    PipelineConversationReply,
    conversation_history,
    render_pipeline_conversations,
    user_turns,
)
from speech_projector.followup_conversation_tone import (
    PredictedInitialToneConversationProvenance,
    initial_user_annotation,
    prepare_conversation_tone,
)
from speech_projector.followup_evaluation import file_artifact
from speech_projector.llm import example_prompt
from speech_projector.models import (
    AsrTranscript,
    EvaluationCondition,
    FileArtifact,
    GreedyDecodingConfig,
    PromptConfig,
    Record,
    Role,
    RunConfig,
    RunResult,
    Turn,
)
from speech_projector.overnight_conversation import DelayedCueFixtures
from speech_projector.overnight_conversation_execution import (
    ConversationEvaluationProvenance,
    ConversationHistoryMode,
    SavedConversationReply,
    reply_identifier,
    response_text,
)
from speech_projector.projectors import Projector
from speech_projector.teacher_evaluation import projector_digest

TRecord = TypeVar("TRecord", bound=Record)
INFERENCE_FILES = (
    "speech_projector/llm.py",
    "speech_projector/models.py",
    "speech_projector/inputs.py",
    "speech_projector/prompts.py",
    "speech_projector/decoding.py",
    "speech_projector/generation.py",
    "speech_projector/projectors.py",
)


class ConversationReuseConfig(Record):
    selected_run_result: Path
    selected_speech_directory: Path
    reference_directory: Path
    predicted_tone_directory: Path
    fixtures: Path
    cache_statistics: Path
    model_revisions: Path
    environment_inventory: Path
    hugging_face_hub: Path
    source_directory: Path
    git_repository: Path
    output_directory: Path

    @field_serializer("*")
    def portable_path(self, path: Path) -> str:
        return path.as_posix()


class ConversationInferenceIdentity(Record):
    model_name: str
    seed: int
    prompt: PromptConfig
    decoding: GreedyDecodingConfig
    max_new_tokens: int
    history_turns: int
    max_history_tokens: int
    initial_policies: tuple[PromptConfig, ...]


class BoundModelSnapshot(Record):
    revision: ModelRevision
    artifacts: tuple[FileArtifact, ...]


class ConversationReuseReceipt(Record):
    configuration: ConversationReuseConfig
    identity: ConversationInferenceIdentity
    selected_provenance: ConversationEvaluationProvenance
    reference_configuration: RunConfig
    reference_provenances: tuple[ConversationBaselineProvenance, ...]
    tone_provenance: PredictedInitialToneConversationProvenance
    environment: EnvironmentInventory
    models: tuple[BoundModelSnapshot, ...]
    inputs: tuple[FileArtifact, ...]
    merged: FileArtifact
    separate_tone: FileArtifact
    speech_responses: int
    text_responses: int
    asr_responses: int
    tone_responses: int
    limitations: tuple[str, ...]


def complete_records(path: Path, adapter: TypeAdapter[TRecord]) -> tuple[TRecord, ...]:
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        raise ValueError(f"Completed source journal has an incomplete suffix: {path}")
    return tuple(adapter.validate_json(line) for line in content.splitlines())


def inference_identity(
    configuration: RunConfig, fixtures: DelayedCueFixtures
) -> ConversationInferenceIdentity:
    match configuration.decoding:
        case GreedyDecodingConfig() as decoding:
            pass
        case _:
            raise ValueError("Conversation reuse requires the recorded greedy decoding")
    if (
        configuration.max_new_tokens != 256
        or configuration.history_turns != 6
        or configuration.max_history_tokens != 8192
    ):
        raise ValueError("Conversation reuse requires the same cap256/history6/token budget8192")
    return ConversationInferenceIdentity(
        model_name=configuration.model_name,
        seed=configuration.seed,
        prompt=configuration.prompt,
        decoding=decoding,
        max_new_tokens=configuration.max_new_tokens,
        history_turns=configuration.history_turns,
        max_history_tokens=configuration.max_history_tokens,
        initial_policies=tuple(
            example_prompt(initial, configuration)
            for scenario in fixtures.scenarios
            for initial in scenario.initial
        ),
    )


def verify_inference_sources(
    configuration: ConversationReuseConfig, commits: Sequence[str]
) -> tuple[FileArtifact, ...]:
    records: list[FileArtifact] = []
    for relative_path in INFERENCE_FILES:
        path = configuration.source_directory / relative_path
        actual = file_artifact(path)
        for commit in set(commits):
            contents = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(configuration.git_repository),
                    "show",
                    f"{commit}:{relative_path}",
                ]
            )
            if (
                len(contents) != actual.bytes
                or hashlib.sha256(contents).hexdigest() != actual.sha256
            ):
                raise ValueError(f"Frozen inference source differs at {relative_path}/{commit}")
        records.append(actual)
    return tuple(records)


def bind_model_snapshot(
    hub: Path, model_name: str, revisions: Sequence[ModelRevision]
) -> BoundModelSnapshot:
    matching = tuple(row for row in revisions if row.model_name == model_name)
    if len(matching) != 1 or matching[0].main_revision is None:
        raise ValueError(f"One generation-time pinned model revision is required: {model_name}")
    revision = matching[0]
    repository = hub / ("models--" + model_name.replace("/", "--"))
    main = repository / "refs" / "main"
    if main.read_text(encoding="utf-8").strip() != revision.main_revision:
        raise ValueError(f"Current model cache main revision differs: {model_name}")
    if revision.main_revision not in revision.snapshot_revisions:
        raise ValueError("Model revision record lacks its main snapshot")
    snapshot = repository / "snapshots" / revision.main_revision
    paths = tuple(sorted(path for path in snapshot.rglob("*") if path.is_file()))
    if (
        not paths
        or not (snapshot / "config.json").is_file()
        or not any(
            path.suffix == ".safetensors" or path.name == "pytorch_model.bin" for path in paths
        )
    ):
        raise ValueError(f"Pinned model snapshot is incomplete: {model_name}")
    return BoundModelSnapshot(
        revision=revision, artifacts=tuple(file_artifact(path) for path in (main, *paths))
    )


def verify_runtime_environment(path: Path) -> EnvironmentInventory:
    recorded = EnvironmentInventory.model_validate_json(path.read_bytes())
    if recorded.python_version != sys.version.split()[0]:
        raise ValueError("Python runtime differs from the recorded frozen inference environment")
    for name in (
        "torch",
        "transformers",
        "safetensors",
        "triton",
        "flash-linear-attention",
        "causal-conv1d",
    ):
        packages = tuple(
            row for row in recorded.packages if row.name.lower().replace("_", "-") == name
        )
        if len(packages) != 1 or packages[0].version != version(name):
            raise ValueError(
                f"Inference runtime package differs from generation provenance: {name}"
            )
    return recorded


def verify_fixture_cache(
    configuration: ConversationReuseConfig,
    fixtures: DelayedCueFixtures,
    reference: FollowupConversationProvenance,
) -> tuple[FileArtifact, ...]:
    statistics = CacheStatistics.model_validate_json(configuration.cache_statistics.read_bytes())
    if (
        statistics.encoder_model != "openai/whisper-small"
        or statistics.hidden_dimension != 768
        or statistics.native_states_per_second != 50
        or statistics.storage_dtype != "bfloat16"
    ):
        raise ValueError("Conversation features must use the original frozen Whisper cache")
    paths: set[Path] = {configuration.cache_statistics}
    for scenario in fixtures.scenarios:
        for initial in scenario.initial:
            paths.update((initial.audio_path, initial.feature_path))
        for followup in scenario.followups:
            audio_path = reference.configuration.followup_asr.audio_root / followup.clip.audio_path
            receipt_path = followup.feature_path.with_suffix(".receipt.json")
            receipt = AdditionalAudioFeature.model_validate_json(receipt_path.read_bytes())
            audio = file_artifact(audio_path)
            features = file_artifact(followup.feature_path)
            if (
                audio.sha256 != followup.clip.sha256
                or (audio.bytes, audio.sha256) != (receipt.audio.bytes, receipt.audio.sha256)
                or (features.bytes, features.sha256)
                != (receipt.features.bytes, receipt.features.sha256)
            ):
                raise ValueError(
                    "Neutral follow-up audio/features differ from their extraction receipt"
                )
            paths.update((audio_path, followup.feature_path, receipt_path))
    return tuple(file_artifact(path) for path in sorted(paths))


def verify_reply_coverage(
    replies: Sequence[SavedConversationReply], fixtures: DelayedCueFixtures, *, speech: bool
) -> None:
    indexed = {reply_identifier(row): row for row in replies}
    if len(indexed) != len(replies):
        raise ValueError("A completed conversation repeats reply identifiers")
    expected: set[str] = set()
    modes = tuple(ConversationHistoryMode) if speech else (ConversationHistoryMode.TEXT_HISTORY,)
    for scenario in fixtures.scenarios:
        for initial in scenario.initial:
            expected.update(
                f"controlled:{initial.example_id}:{mode.value}:{turn_index}"
                for mode in modes
                for turn_index in (2, 3)
            )
            expected.update(f"rollout:{initial.example_id}:{turn_index}" for turn_index in range(4))
    if set(indexed) != expected:
        raise ValueError("Conversation references have incomplete or unexpected reply coverage")
    for scenario in fixtures.scenarios:
        for branch, initial in enumerate(scenario.initial):
            truth = user_turns(scenario, branch, EvaluationCondition.TEXT, ())
            for turn_index in (2, 3):
                for mode in modes:
                    identifier = f"controlled:{initial.example_id}:{mode.value}:{turn_index}"
                    expected.add(identifier)
                    row = indexed[identifier]
                    if (
                        row.scenario_base_id != scenario.base_id
                        or row.current_case_id
                        != scenario.followups[turn_index - 1].clip.case.case_id
                        or row.history_for_judge
                        != conversation_history(truth, scenario.fixed_assistants, turn_index)
                    ):
                        raise ValueError("Controlled conversation differs from its fixed fixture")
            assistants: list[Turn] = []
            for turn_index in range(4):
                identifier = f"rollout:{initial.example_id}:{turn_index}"
                expected.add(identifier)
                row = indexed[identifier]
                current_id = (
                    initial.example_id
                    if turn_index == 0
                    else scenario.followups[turn_index - 1].clip.case.case_id
                )
                if (
                    row.scenario_base_id != scenario.base_id
                    or row.current_case_id != current_id
                    or row.history_for_judge != conversation_history(truth, assistants, turn_index)
                ):
                    raise ValueError(
                        "Rollout history differs from its actual preceding assistant replies"
                    )
                assistants.append(Turn(role=Role.ASSISTANT, text=response_text(row.generation)))
    if set(indexed) != expected:
        raise ValueError("Conversation references have incomplete or unexpected reply coverage")


def verify_reference_words(
    rows: Sequence[PipelineConversationReply],
    fixtures: DelayedCueFixtures,
    condition: EvaluationCondition,
    transcripts: Sequence[AsrTranscript],
) -> None:
    expected = {
        initial.example_id: user_turns(scenario, branch, condition, transcripts)
        for scenario in fixtures.scenarios
        for branch, initial in enumerate(scenario.initial)
    }
    for row in rows:
        if (
            row.reply.initial_example_id not in expected
            or row.condition != condition
            or row.user_words != expected[row.reply.initial_example_id]
        ):
            raise ValueError("Reference user words differ from the true or saved recognized input")
    verify_reply_coverage(tuple(row.reply for row in rows), fixtures, speech=False)


def write_rows(path: Path, rows: Sequence[Record]) -> None:
    partial = path.with_suffix(".part")
    partial.write_bytes(b"".join(row.model_dump_json().encode() + b"\n" for row in rows))
    partial.replace(path)


def merge_conversation_references(
    configuration: ConversationReuseConfig,
) -> ConversationReuseReceipt:
    if configuration.output_directory.exists():
        raise ValueError("Reuse output must be a new unsealed directory")
    fixtures = DelayedCueFixtures.model_validate_json(configuration.fixtures.read_bytes())
    if len(fixtures.scenarios) != 3:
        raise ValueError("Final comparison requires the fixed three-family conversation panel")
    fixture_digest = hashlib.sha256(fixtures.model_dump_json().encode()).hexdigest()
    selected_result = RunResult.model_validate_json(configuration.selected_run_result.read_bytes())
    speech_provenance_path = configuration.selected_speech_directory / "provenance.json"
    selected = ConversationEvaluationProvenance.model_validate_json(
        speech_provenance_path.read_bytes()
    )
    expected_config = selected_result.config.model_copy(
        update={"history_turns": 6, "max_history_tokens": 8192}
    )
    if selected.configuration != expected_config or selected.fixtures_sha256 != fixture_digest:
        raise ValueError(
            "Selected speech output has a different inference configuration or fixture"
        )
    projector = Projector(selected_result.config.projector)
    projector.load_state_dict(load_file(selected_result.checkpoint_path))
    if projector_digest(projector) != selected.projector_weights_sha256:
        raise ValueError("Selected conversation was generated by a different projector checkpoint")
    identity = inference_identity(selected.configuration, fixtures)
    reference_path = configuration.reference_directory / "provenance.json"
    reference = FollowupConversationProvenance.model_validate_json(reference_path.read_bytes())
    reference_result = RunResult.model_validate_json(
        reference.configuration.run_result.read_bytes()
    )
    reference_config = reference_result.config.model_copy(
        update={"history_turns": 6, "max_history_tokens": reference.configuration.history_tokens}
    )
    if inference_identity(reference_config, fixtures) != identity:
        raise ValueError("Reference generation has different actual inference settings or policies")
    for record in reference.inputs:
        if stable_digest(record.source_path) != (record.bytes, record.sha256):
            raise ValueError("A saved conversation reference input changed after generation")
    if reference.configuration.fixtures.read_bytes() != configuration.fixtures.read_bytes():
        raise ValueError("Reference and selected test fixture bytes differ")
    asr_path = reference.configuration.asr_transcripts
    followup_asr_path = configuration.reference_directory / "followup_asr" / "transcripts.jsonl"
    transcripts = complete_records(asr_path, TypeAdapter(AsrTranscript)) + complete_records(
        followup_asr_path, TypeAdapter(AsrTranscript)
    )
    followup_asr_provenance_path = (
        configuration.reference_directory / "followup_asr" / "provenance.json"
    )
    followup_asr = FollowupAsrProvenance.model_validate_json(
        followup_asr_provenance_path.read_bytes()
    )
    if (
        followup_asr.configuration != reference.configuration.followup_asr
        or followup_asr.fixtures_sha256 != fixture_digest
        or tuple((row.bytes, row.sha256) for row in followup_asr.audio)
        != tuple(
            stable_digest(reference.configuration.followup_asr.audio_root / item.clip.audio_path)
            for item in fixtures.scenarios[0].followups
        )
    ):
        raise ValueError("Neutral follow-up ASR provenance differs in audio, fixture or encoder")
    speech_path = configuration.selected_speech_directory / "replies.jsonl"
    speech = complete_records(speech_path, TypeAdapter(SavedConversationReply))
    verify_reply_coverage(speech, fixtures, speech=True)
    words = {
        initial.example_id: user_turns(scenario, branch, EvaluationCondition.TEXT, ())
        for scenario in fixtures.scenarios
        for branch, initial in enumerate(scenario.initial)
    }
    merged = [
        PipelineConversationReply(
            condition=EvaluationCondition.SPEECH,
            user_words=words[row.initial_example_id],
            reply=row,
        )
        for row in speech
    ]
    paths = [
        configuration.selected_run_result,
        selected_result.checkpoint_path,
        configuration.fixtures,
        configuration.model_revisions,
        configuration.environment_inventory,
        speech_provenance_path,
        speech_path,
        reference_path,
        reference.configuration.run_result,
        asr_path,
        followup_asr_path,
        followup_asr_provenance_path,
    ]
    baselines: list[ConversationBaselineProvenance] = []
    for condition in (EvaluationCondition.TEXT, EvaluationCondition.ASR):
        directory = configuration.reference_directory / condition.value
        provenance_path = directory / "provenance.json"
        provenance = ConversationBaselineProvenance.model_validate_json(
            provenance_path.read_bytes()
        )
        expected_transcripts = transcripts if condition == EvaluationCondition.ASR else ()
        if (
            provenance.condition != condition
            or provenance.source_git_commit != reference.configuration.source_git_commit
            or provenance.configuration_sha256
            != hashlib.sha256(reference_config.model_dump_json().encode()).hexdigest()
            or provenance.fixtures_sha256 != fixture_digest
            or provenance.transcripts != expected_transcripts
        ):
            raise ValueError("TEXT/ASR reference provenance differs from the actual frozen inputs")
        journal = directory / "replies.jsonl"
        rows = complete_records(journal, TypeAdapter(PipelineConversationReply))
        verify_reference_words(rows, fixtures, condition, transcripts)
        merged.extend(rows)
        baselines.append(provenance)
        paths.extend((provenance_path, journal))
    tone_provenance_path = configuration.predicted_tone_directory / "provenance.json"
    tone = PredictedInitialToneConversationProvenance.model_validate_json(
        tone_provenance_path.read_bytes()
    )
    tone_fixtures, checked_tone = prepare_conversation_tone(tone.configuration)
    if (
        tone != checked_tone
        or tone_fixtures != fixtures
        or tone.transcripts != transcripts
        or inference_identity(tone.inference_configuration, fixtures) != identity
    ):
        raise ValueError(
            "Predicted-initial-tone reference no longer matches its exact actual input"
        )
    tone_path = configuration.predicted_tone_directory / "replies.jsonl"
    tone_rows = complete_records(tone_path, TypeAdapter(PipelineConversationReply))
    verify_reply_coverage(tuple(row.reply for row in tone_rows), fixtures, speech=False)
    tone_inputs = {item.example.example_id: item for item in tone.initial_inputs}
    tone_words = {
        initial.example_id: (initial_user_annotation(tone_inputs[initial.example_id]),)
        + user_turns(scenario, branch, EvaluationCondition.ASR, transcripts)[1:]
        for scenario in fixtures.scenarios
        for branch, initial in enumerate(scenario.initial)
    }
    for row in tone_rows:
        if (
            row.condition != EvaluationCondition.ASR
            or row.user_words != tone_words[row.reply.initial_example_id]
        ):
            raise ValueError(
                "Tone reference user message differs from its historical predicted cue"
            )
    paths.extend((tone_provenance_path, tone_path))
    inputs = tuple(file_artifact(path) for path in paths) + tuple(tone.artifacts)
    inputs += verify_fixture_cache(configuration, fixtures, reference)
    commits = (
        selected.source_commit,
        *(row.source_git_commit for row in baselines),
        tone.configuration.source_git_commit,
    )
    inputs += verify_inference_sources(configuration, commits)
    revisions = complete_records(configuration.model_revisions, TypeAdapter(ModelRevision))
    models = tuple(
        bind_model_snapshot(configuration.hugging_face_hub, name, revisions)
        for name in (identity.model_name, selected_result.config.speech_model_name)
    )
    if models[1].revision.main_revision != reference.configuration.followup_asr.revision:
        raise ValueError("Neutral ASR uses a different Whisper snapshot revision")
    environment = verify_runtime_environment(configuration.environment_inventory)
    if len(merged) != 132 or len(tone_rows) != 36:
        raise ValueError(
            "Final conversation comparison must contain132 plus36 separate tone replies"
        )
    configuration.output_directory.mkdir(parents=True)
    merged_path = configuration.output_directory / "matched_replies.jsonl"
    tone_output = configuration.output_directory / "predicted_initial_tone_replies.jsonl"
    write_rows(merged_path, merged)
    write_rows(tone_output, tone_rows)
    receipt = ConversationReuseReceipt(
        configuration=configuration,
        identity=identity,
        selected_provenance=selected,
        reference_configuration=reference_config,
        reference_provenances=tuple(baselines),
        tone_provenance=tone,
        environment=environment,
        models=models,
        inputs=inputs,
        merged=file_artifact(merged_path),
        separate_tone=file_artifact(tone_output),
        speech_responses=60,
        text_responses=36,
        asr_responses=36,
        tone_responses=36,
        limitations=(
            "Reference responses and timings are reused, "
            "never regenerated or counted as new GPU work.",
            "Speech user_words are audit transcripts, not model inputs; "
            "all three speech history controls remain.",
            "The separate predicted-tone comparator annotates the initial USER, "
            "unlike the single-turn SYSTEM cue.",
            "Full training configs are retained; "
            "objective/LR/name/step differences do not enter inference requests.",
            "Snapshot/source hashes bind current immutable files; "
            "past CUDA kernel/allocator state is not reconstructed.",
            "KV/recurrent caches are fresh within each generation; "
            "no cross-request session cache is reused.",
        ),
    )
    write_record(configuration.output_directory / "reuse_receipt.json", receipt)
    (configuration.output_directory / "comparison.md").write_text(
        "# Selected speech and reused frozen conversation references\n\n"
        "132 replies: speech60, TEXT36, ASR36. The predicted-initial-tone36 comparison "
        "is separate because its prompt placement differs. Original source outputs, "
        "timings and provenance remain unchanged; this merge performs no inference.\n\n"
        + render_pipeline_conversations(merged)
        + "\n\n# Separate predicted initial tone\n\n"
        + render_pipeline_conversations(tone_rows),
        encoding="utf-8",
    )
    return receipt
