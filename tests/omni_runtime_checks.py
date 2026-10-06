"""Explicitly invoked CPU checks inside the configured isolated Omni environment."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pytest
import torch
from pytest import MonkeyPatch
from transformers import PreTrainedTokenizerBase
from vllm import SamplingParams
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
from vllm_omni.outputs import OmniRequestOutput

from scripts.inventory_results import write_record
from speech_projector import emotional_audio_omni as implementation
from speech_projector.emotion_preview import Delivery, PreviewCase, PreviewPlan
from speech_projector.emotional_audio import EmotionalAudioConfig
from speech_projector.emotional_audio_omni import (
    CodecFinish,
    CodecTermination,
    OmniAttempt,
    OmniAudioConfig,
    PreparedModel,
)
from speech_projector.journal import read_journal
from speech_projector.models import Record


class StageMetric(Record):
    finish_reason: str
    num_tokens_out: int


class PipelineMetrics(Record):
    stage_metrics: dict[str, StageMetric]


def generated(index: int, reason: str = "stop", tokens: int = 8) -> OmniRequestOutput:
    return OmniRequestOutput(
        request_id=f"{index}_test",
        metrics=PipelineMetrics(
            stage_metrics={"0": StageMetric(finish_reason=reason, num_tokens_out=tokens)}
        ).model_dump(),
        _multimodal_output={"audio": torch.ones(240), "sr": torch.tensor(24000)},
    )


class FakeEngine(Omni):
    def __init__(self, responses: Sequence[Sequence[OmniRequestOutput]]) -> None:
        self.responses = list(responses)
        self.seeds: list[int | None] = []
        self.caps: list[int] = []
        self.closed = False

    def generate(
        self,
        prompts: Sequence[OmniTokensPrompt],
        sampling_params_list: Sequence[SamplingParams],
        *,
        py_generator: Literal[False] = False,
        use_tqdm: bool = False,
    ) -> list[OmniRequestOutput]:
        self.seeds.append(sampling_params_list[0].seed)
        self.caps.append(sampling_params_list[0].max_tokens)
        return list(self.responses.pop(0))

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def configuration(tmp_path: Path) -> OmniAudioConfig:
    cases = tuple(
        PreviewCase(
            case_id=f"case_{index}",
            text="I could use your help today.",
            delivery=Delivery.NEUTRAL,
            instruct="Speak calmly.",
            seed=42 + index,
        )
        for index in range(2)
    )
    deployment = tmp_path / "deploy.yaml"
    deployment.write_text("stages: []\n", encoding="utf-8")
    return OmniAudioConfig(
        audio=EmotionalAudioConfig(
            output=tmp_path / "output",
            source_commit="a" * 40,
            plan=PreviewPlan(cases=cases, max_new_tokens=16),
            batch_size=2,
            retry_max_new_tokens=32,
        ),
        deploy_config=deployment,
    )


def install_fake(
    monkeypatch: MonkeyPatch, configuration: OmniAudioConfig, engine: FakeEngine
) -> None:
    prepared = PreparedModel(
        snapshot=configuration.audio.output,
        tokenizer=PreTrainedTokenizerBase(),
        model_config=Qwen3TTSConfig(),
    )

    def prepare(plan: PreviewPlan) -> PreparedModel:
        return prepared

    def create(prepared_model: PreparedModel, supplied: OmniAudioConfig) -> Omni:
        return engine

    def prompt(case: PreviewCase, plan: PreviewPlan, model: PreparedModel) -> OmniTokensPrompt:
        return OmniTokensPrompt(prompt_token_ids=[1])

    monkeypatch.setattr(implementation, "prepare_model", prepare)
    monkeypatch.setattr(implementation, "create_engine", create)
    monkeypatch.setattr(implementation, "make_prompt", prompt)
    monkeypatch.setattr(implementation, "gpu_used_gb", lambda: 10.25)


def test_sequential_calls_and_cap_retry(
    monkeypatch: MonkeyPatch, configuration: OmniAudioConfig
) -> None:
    engine = FakeEngine(((generated(1), generated(0, "length", 16)), (generated(0),)))
    install_fake(monkeypatch, configuration, engine)
    summary = implementation.generate_audio(configuration)
    assert summary.completed == 2 and not summary.failed_case_ids
    assert engine.seeds == [42, 42] and engine.caps == [16, 32] and engine.closed
    attempts = read_journal(configuration.audio.output / "attempts.jsonl", OmniAttempt)
    assert len(attempts) == 3
    assert attempts[0].termination.finish_reason == CodecFinish.LENGTH
    assert all(
        (configuration.audio.output / attempt.waveform.path).is_file() for attempt in attempts
    )
    assert implementation.verify_completed(configuration.audio)[0].case.case_id == "case_0"


def test_resume_skips_committed_and_rejects_changed_waveform(
    monkeypatch: MonkeyPatch, configuration: OmniAudioConfig
) -> None:
    engine = FakeEngine(((generated(0), generated(1)),))
    install_fake(monkeypatch, configuration, engine)
    implementation.generate_audio(configuration)
    resumed = implementation.generate_audio(configuration)
    assert resumed.session_completed == 0 and engine.caps == [16]
    path = configuration.audio.output / "audio/case_0.wav"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        implementation.generate_audio(configuration)


def test_provenance_rejects_deployment_change(configuration: OmniAudioConfig) -> None:
    implementation.initialize_records(configuration)
    configuration.deploy_config.write_text("stages: changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        implementation.initialize_records(configuration)


def test_orphan_waveform_is_archived(configuration: OmniAudioConfig) -> None:
    implementation.initialize_records(configuration)
    case = configuration.audio.plan.cases[0]
    outcome = implementation.decode_output(generated(0), case, configuration.audio.plan, 42, 0.1)
    implementation.persist_waveform(configuration.audio.output, case, outcome)
    assert not (configuration.audio.output / "clips/case_0.json").exists()
    assert implementation.initialize_records(configuration) == ()
    assert not (configuration.audio.output / "audio/case_0.wav").exists()
    assert len(tuple((configuration.audio.output / "orphaned_audio").glob("*.wav"))) == 1


def test_resume_rejects_unconfigured_cap(
    monkeypatch: MonkeyPatch, configuration: OmniAudioConfig
) -> None:
    engine = FakeEngine(((generated(0), generated(1)),))
    install_fake(monkeypatch, configuration, engine)
    implementation.generate_audio(configuration)
    path = configuration.audio.output / "terminations/case_0.json"
    termination = CodecTermination.model_validate_json(path.read_bytes())
    write_record(path, termination.model_copy(update={"max_new_tokens": 1000}))
    with pytest.raises(ValueError, match="inconsistent codec"):
        implementation.verify_completed(configuration.audio)


def test_retry_cap_failure_is_retained(
    monkeypatch: MonkeyPatch, configuration: OmniAudioConfig
) -> None:
    engine = FakeEngine(((generated(0, "length", 16), generated(1)), (generated(0, "length", 32),)))
    install_fake(monkeypatch, configuration, engine)
    summary = implementation.generate_audio(configuration)
    assert summary.completed == 1 and summary.failed_case_ids == ("case_0",) and engine.closed
    assert len(tuple((configuration.audio.output / "failed_audio").rglob("*.wav"))) == 2
    assert (configuration.audio.output / "failures/case_0.json").is_file()


def test_missing_codec_evidence_closes_engine(
    monkeypatch: MonkeyPatch, configuration: OmniAudioConfig
) -> None:
    engine = FakeEngine(((OmniRequestOutput(request_id="0_test"), generated(1)),))
    install_fake(monkeypatch, configuration, engine)
    with pytest.raises(ValueError, match="Missing codec"):
        implementation.generate_audio(configuration)
    assert engine.closed
    assert (configuration.audio.output / "summary.json").is_file()


def test_stop_at_cap_minus_one_is_valid(configuration: OmniAudioConfig) -> None:
    implementation.initialize_records(configuration)
    case = configuration.audio.plan.cases[0]
    outcome = implementation.decode_output(
        generated(0, "stop", 15), case, configuration.audio.plan, 42, 0.1
    )
    clip = implementation.record_generated(configuration, case, outcome, "test-session")
    assert clip is not None and clip.codec_tokens == 15
    assert len(read_journal(configuration.audio.output / "attempts.jsonl", OmniAttempt)) == 1
