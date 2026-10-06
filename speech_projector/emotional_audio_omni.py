"""Resumable CustomVoice synthesis through the isolated vLLM-Omni runtime."""

import hashlib
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

import numpy as np
import soundfile
import torch
from huggingface_hub import snapshot_download
from pydantic import Field, InstanceOf
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import SamplingParams
from vllm_omni import Omni
from vllm_omni.entrypoints.openai.speech_usage import SpeechOutputTokenCounter
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    Qwen3TTSPromptEmbedsBuilder,
)
from vllm_omni.outputs import OmniRequestOutput

from scripts.inventory_results import stable_digest, write_record
from scripts.package_results import FileArtifact, ModelRevision
from speech_projector.emotion_preview import (
    PreviewCase,
    PreviewClip,
    PreviewPlan,
    PreviewSynthesisFailure,
    SynthesizedAudio,
    normalize_waveform,
)
from speech_projector.emotional_audio import (
    AudioGenerationSummary,
    EmotionalAudioConfig,
    archive_uncommitted_audio,
    completed_audio,
)
from speech_projector.journal import append_record
from speech_projector.models import Record


class OmniAudioConfig(Record):
    audio: EmotionalAudioConfig
    deploy_config: Path


class OmniProvenance(Record):
    configuration: OmniAudioConfig
    deployment: FileArtifact
    model_revision: ModelRevision
    module_sha256: str
    torch_version: str
    transformers_version: str
    vllm_version: str
    omni_version: str


class CodecFinish(str, Enum):
    STOP = "stop"
    LENGTH = "length"


class CodecTermination(Record):
    case_id: str
    finish_reason: CodecFinish
    codec_tokens: int = Field(gt=0)
    max_new_tokens: int = Field(ge=4)
    actual_seed: int = Field(ge=0)
    runtime_seconds: float = Field(ge=0)

    @property
    def accepted(self) -> bool:
        return (
            self.finish_reason == CodecFinish.STOP and 0 < self.codec_tokens < self.max_new_tokens
        )


class OmniAttempt(Record):
    session_id: str
    termination: CodecTermination
    waveform: FileArtifact


class OmniBatchTiming(Record):
    session_id: str
    case_ids: tuple[str, ...]
    actual_batch_seed: int
    max_new_tokens: int
    runtime_seconds: float
    gpu_used_gb_after: float


class OmniSession(Record):
    session_id: str
    started_at: datetime
    initialization_seconds: float


class CustomVoiceConditioning(Record):
    task_type: tuple[Literal["CustomVoice"], ...] = ("CustomVoice",)
    text: tuple[str, ...]
    instruct: tuple[str, ...]
    language: tuple[str, ...]
    speaker: tuple[str, ...]
    non_streaming_mode: tuple[bool, ...] = (True,)
    max_new_tokens: tuple[int, ...]


class OmniAudioPayload(Record):
    audio: InstanceOf[torch.Tensor] | list[InstanceOf[torch.Tensor]]
    sr: int = Field(gt=0)


@dataclass(frozen=True)
class PreparedModel:
    snapshot: Path
    tokenizer: PreTrainedTokenizerBase
    model_config: Qwen3TTSConfig


@dataclass(frozen=True)
class GeneratedAudio:
    termination: CodecTermination
    synthesis: SynthesizedAudio


def artifact(path: Path, relative: Path) -> FileArtifact:
    size, digest = stable_digest(path)
    return FileArtifact(path=relative, source_path=path.resolve(), bytes=size, sha256=digest)


def prepare_model(plan: PreviewPlan) -> PreparedModel:
    snapshot = Path(
        snapshot_download(plan.model_name, revision=plan.revision, local_files_only=True)
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model_config = Qwen3TTSConfig.from_pretrained(snapshot, local_files_only=True)
    if model_config.talker_config.codec_eos_token_id != 2150:
        raise ValueError("Pinned CustomVoice codec EOS differs from the release constraint")
    return PreparedModel(snapshot=snapshot, tokenizer=tokenizer, model_config=model_config)


def make_prompt(case: PreviewCase, plan: PreviewPlan, prepared: PreparedModel) -> OmniTokensPrompt:
    conditioning = CustomVoiceConditioning(
        text=(case.text,),
        instruct=(case.instruct,),
        language=(plan.language,),
        speaker=(plan.speaker,),
        max_new_tokens=(plan.max_new_tokens,),
    ).model_dump(mode="json")
    length = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        additional_information=conditioning,
        task_type="CustomVoice",
        tokenize_prompt=lambda text: prepared.tokenizer.encode(text, add_special_tokens=True),
        codec_language_id=prepared.model_config.talker_config.codec_language_id,
        spk_is_dialect=prepared.model_config.talker_config.spk_is_dialect,
    )
    return OmniTokensPrompt(prompt_token_ids=[1] * length, additional_information=conditioning)


def create_engine(prepared: PreparedModel, configuration: OmniAudioConfig) -> Omni:
    return Omni(
        model=str(prepared.snapshot),
        revision=configuration.audio.plan.revision,
        deploy_config=str(configuration.deploy_config),
        trust_remote_code=True,
        stage_init_timeout=300,
        init_timeout=600,
    )


def gpu_used_gb() -> float:
    value = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return int(value) * 1024**2 / 1e9


def decode_output(
    output: OmniRequestOutput, case: PreviewCase, plan: PreviewPlan, seed: int, runtime: float
) -> GeneratedAudio:
    if output.error is not None:
        raise ValueError(f"Omni failed {case.case_id}: {output.error}")
    counter = SpeechOutputTokenCounter()
    counter.observe(output)
    if counter.stage0_finish_reason is None or counter.total() <= 0:
        raise ValueError(f"Missing codec completion/count for {case.case_id}")
    termination = CodecTermination(
        case_id=case.case_id,
        finish_reason=CodecFinish(counter.stage0_finish_reason),
        codec_tokens=counter.total(),
        max_new_tokens=plan.max_new_tokens,
        actual_seed=seed,
        runtime_seconds=runtime,
    )
    payload = OmniAudioPayload.model_validate(output.multimodal_output)
    match payload.audio:
        case torch.Tensor() as waveform:
            combined = waveform
        case list() as chunks:
            combined = torch.cat(chunks, dim=-1)
    values = np.asarray(combined.detach().float().cpu().numpy().reshape(-1), dtype=np.float32)
    synthesis = SynthesizedAudio(
        waveform=values,
        sample_rate=payload.sr,
        codec_tokens=termination.codec_tokens,
        runtime_seconds=runtime,
    )
    return GeneratedAudio(termination=termination, synthesis=synthesis)


def generate_batch(
    engine: Omni,
    prepared: PreparedModel,
    plan: PreviewPlan,
    cases: tuple[PreviewCase, ...],
    seed: int,
) -> tuple[GeneratedAudio, ...]:
    prompts = tuple(make_prompt(case, plan, prepared) for case in cases)
    talker = SamplingParams(
        temperature=plan.temperature,
        top_k=plan.top_k,
        top_p=plan.top_p,
        repetition_penalty=plan.repetition_penalty,
        seed=seed,
        max_tokens=plan.max_new_tokens,
        min_tokens=2,
        stop_token_ids=[2150],
        detokenize=False,
    )
    started = perf_counter()
    outputs = engine.generate(
        prompts,
        sampling_params_list=[talker, SamplingParams(temperature=0, max_tokens=65536)],
        py_generator=False,
        use_tqdm=False,
    )
    elapsed = perf_counter() - started
    ordered = sorted(outputs, key=lambda item: int(item.request_id.partition("_")[0]))
    if tuple(int(item.request_id.partition("_")[0]) for item in ordered) != tuple(
        range(len(cases))
    ):
        raise ValueError("Omni final output indices are incomplete or duplicated")
    return tuple(
        decode_output(output, case, plan, seed, elapsed / len(cases))
        for output, case in zip(ordered, cases, strict=True)
    )


def persist_waveform(directory: Path, case: PreviewCase, generated: GeneratedAudio) -> PreviewClip:
    if not generated.termination.accepted:
        raise ValueError("Refusing to commit audio without stage-0 stop below its cap")
    destination = directory / "audio" / f"{case.case_id}.wav"
    if destination.exists():
        raise ValueError(f"Refusing to overwrite existing audio: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    synthesis = generated.synthesis
    waveform = normalize_waveform(synthesis.waveform)
    soundfile.write(partial, waveform, synthesis.sample_rate, format="WAV", subtype="FLOAT")
    partial.replace(destination)
    clip = PreviewClip(
        case=case,
        audio=artifact(destination, destination.relative_to(directory)),
        sample_rate=synthesis.sample_rate,
        samples=waveform.size,
        duration_seconds=waveform.size / synthesis.sample_rate,
        runtime_seconds=synthesis.runtime_seconds,
        codec_tokens=synthesis.codec_tokens,
        peak_amplitude=float(np.abs(waveform).max()),
    )
    return clip


def persist_failed_waveform(
    directory: Path, case: PreviewCase, generated: GeneratedAudio, session_id: str
) -> FileArtifact:
    path = (
        directory
        / "failed_audio"
        / session_id
        / f"{case.case_id}_cap{generated.termination.max_new_tokens}.wav"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(
        path, generated.synthesis.waveform, generated.synthesis.sample_rate, subtype="FLOAT"
    )
    return artifact(path, path.relative_to(directory))


def record_generated(
    configuration: OmniAudioConfig, case: PreviewCase, generated: GeneratedAudio, session_id: str
) -> PreviewClip | None:
    directory = configuration.audio.output
    termination = generated.termination
    if termination.accepted:
        write_record(directory / "terminations" / f"{case.case_id}.json", termination)
        try:
            clip = persist_waveform(directory, case, generated)
        except ValueError:
            waveform = persist_failed_waveform(directory, case, generated, session_id)
            append_record(
                directory / "attempts.jsonl",
                OmniAttempt(
                    session_id=session_id,
                    termination=termination,
                    waveform=waveform,
                ),
            )
            raise
        waveform = clip.audio
    else:
        waveform = persist_failed_waveform(directory, case, generated, session_id)
        clip = None
    append_record(
        directory / "attempts.jsonl",
        OmniAttempt(
            session_id=session_id,
            termination=termination,
            waveform=waveform,
        ),
    )
    if clip is not None:
        write_record(directory / "clips" / f"{case.case_id}.json", clip)
    return clip


def verify_completed(configuration: EmotionalAudioConfig) -> tuple[PreviewClip, ...]:
    clips = completed_audio(configuration)
    for clip in clips:
        termination = CodecTermination.model_validate_json(
            (configuration.output / "terminations" / f"{clip.case.case_id}.json").read_bytes()
        )
        if (
            not termination.accepted
            or termination.case_id != clip.case.case_id
            or termination.codec_tokens != clip.codec_tokens
            or termination.max_new_tokens
            not in (configuration.plan.max_new_tokens, configuration.retry_max_new_tokens)
        ):
            raise ValueError(f"Completed clip has inconsistent codec evidence: {clip.case.case_id}")
    return clips


def initialize_records(configuration: OmniAudioConfig) -> tuple[PreviewClip, ...]:
    audio = configuration.audio
    audio.output.mkdir(parents=True, exist_ok=True)
    config_path = audio.output / "config.json"
    if config_path.exists():
        if EmotionalAudioConfig.model_validate_json(config_path.read_bytes()) != audio:
            raise ValueError("Omni audio resume configuration changed")
    else:
        write_record(config_path, audio)
        write_record(audio.output / "plan.json", audio.plan)
    deployment_path = audio.output / "deployment.yaml"
    deployment_bytes = configuration.deploy_config.read_bytes()
    if deployment_path.exists():
        if deployment_path.read_bytes() != deployment_bytes:
            raise ValueError("Omni deployment provenance changed")
    else:
        partial_deployment = deployment_path.with_suffix(".part")
        partial_deployment.write_bytes(deployment_bytes)
        partial_deployment.replace(deployment_path)
    provenance = OmniProvenance(
        configuration=configuration,
        deployment=artifact(deployment_path, Path(deployment_path.name)),
        model_revision=ModelRevision(
            model_name=audio.plan.model_name,
            snapshot_revisions=(audio.plan.revision,),
            main_revision=audio.plan.revision,
        ),
        module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch_version=version("torch"),
        transformers_version=version("transformers"),
        vllm_version=version("vllm"),
        omni_version=version("vllm-omni"),
    )
    provenance_path = audio.output / "provenance.json"
    if provenance_path.exists():
        if OmniProvenance.model_validate_json(provenance_path.read_bytes()) != provenance:
            raise ValueError("Omni resume provenance, source, deployment or runtime changed")
    else:
        write_record(provenance_path, provenance)
    (audio.output / "synthesis_protocol.md").write_text(
        "# Omni codec completion and randomness\n\n"
        "Stage-0 finish reason and codec decode-token counts come from SpeechOutputTokenCounter. "
        "Committed clips require stop below the actual cap. Codec token counts are pipeline "
        "decode-token counts, not a claim that raw EOS was retained in audio payloads. "
        "Batches share their first pending case's seed; individual capped retries use that case's "
        "seed. Replay logged batch membership and seeds; resume can change pending membership. "
        "The pinned deploy YAML controls subtalker sampling; its exact hash is in provenance.\n\n"
        "Summary peak_vram_gb is the largest observed post-batch nvidia-smi GPU-used sample "
        "(decimal GB), including worker processes; it is not a measured allocation peak. "
        "Failed capped waveforms and every successful/length attempt are retained. Committed "
        "clips are hash-verified and never overwritten; orphaned WAVs are archived "
        "before resume.\n",
        encoding="utf-8",
    )
    archive_uncommitted_audio(audio)
    return verify_completed(audio)


def save_summary(
    configuration: EmotionalAudioConfig,
    existing: tuple[PreviewClip, ...],
    completed: list[PreviewClip],
    failed: list[str],
    started: float,
    observed_memory: float,
) -> AudioGenerationSummary:
    summary = AudioGenerationSummary(
        planned=len(configuration.plan.cases),
        completed=len(existing) + len(completed),
        failed_case_ids=tuple(failed),
        session_runtime_seconds=perf_counter() - started,
        session_audio_seconds=sum(clip.duration_seconds for clip in completed),
        session_completed=len(completed),
        peak_vram_gb=observed_memory,
    )
    write_record(configuration.output / "summary.json", summary)
    return summary


def generate_audio(configuration: OmniAudioConfig) -> AudioGenerationSummary:
    audio = configuration.audio
    existing = initialize_records(configuration)
    existing_ids = {clip.case.case_id for clip in existing}
    pending = tuple(case for case in audio.plan.cases if case.case_id not in existing_ids)
    started = perf_counter()
    completed: list[PreviewClip] = []
    failed: list[str] = []
    summary_path = audio.output / "summary.json"
    observed_memory = (
        AudioGenerationSummary.model_validate_json(summary_path.read_bytes()).peak_vram_gb
        if summary_path.exists()
        else 0.0
    )
    if not pending:
        return save_summary(audio, existing, completed, failed, started, observed_memory)
    prepared = prepare_model(audio.plan)
    session_id = str(uuid4())
    initialization_started = perf_counter()
    engine = create_engine(prepared, configuration)
    try:
        append_record(
            audio.output / "sessions.jsonl",
            OmniSession(
                session_id=session_id,
                started_at=datetime.now(timezone.utc),
                initialization_seconds=perf_counter() - initialization_started,
            ),
        )
        for offset in range(0, len(pending), audio.batch_size):
            cases = pending[offset : offset + audio.batch_size]
            batch_started = perf_counter()
            try:
                outcomes = generate_batch(engine, prepared, audio.plan, cases, cases[0].seed)
            except (ValueError, RuntimeError) as error:
                for case in cases:
                    write_record(
                        audio.output / "failures" / f"{case.case_id}.json",
                        PreviewSynthesisFailure(
                            case=case,
                            error=str(error),
                            runtime_seconds=perf_counter() - batch_started,
                        ),
                    )
                    failed.append(case.case_id)
                save_summary(audio, existing, completed, failed, started, observed_memory)
                raise
            for case, outcome in zip(cases, outcomes, strict=True):
                try:
                    clip = record_generated(configuration, case, outcome, session_id)
                    if clip is None:
                        retry_plan = audio.plan.model_copy(
                            update={"max_new_tokens": audio.retry_max_new_tokens}
                        )
                        retry = generate_batch(engine, prepared, retry_plan, (case,), case.seed)[0]
                        clip = record_generated(configuration, case, retry, session_id)
                    if clip is None:
                        raise ValueError(f"Codec generation failed after retry for {case.case_id}")
                    completed.append(clip)
                except (ValueError, RuntimeError) as error:
                    write_record(
                        audio.output / "failures" / f"{case.case_id}.json",
                        PreviewSynthesisFailure(
                            case=case,
                            error=str(error),
                            runtime_seconds=perf_counter() - batch_started,
                        ),
                    )
                    failed.append(case.case_id)
                    print(f"FAILED {case.case_id}: {error}", flush=True)
            used_memory = gpu_used_gb()
            observed_memory = max(observed_memory, used_memory)
            append_record(
                audio.output / "batches.jsonl",
                OmniBatchTiming(
                    session_id=session_id,
                    case_ids=tuple(case.case_id for case in cases),
                    actual_batch_seed=cases[0].seed,
                    max_new_tokens=audio.plan.max_new_tokens,
                    runtime_seconds=perf_counter() - batch_started,
                    gpu_used_gb_after=used_memory,
                ),
            )
            summary = save_summary(audio, existing, completed, failed, started, observed_memory)
            print(
                f"Omni audio {summary.completed}/{summary.planned}; batch "
                f"{perf_counter() - batch_started:.2f}s; failed {len(failed)}",
                flush=True,
            )
        return summary
    finally:
        engine.close()
