from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import Tensor
from torch.nn import functional as functional
from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from speech_projector.evaluation import evaluate, original_dialogue_text
from speech_projector.inputs import SpeechInput, TranscriptInput
from speech_projector.llm import FrozenQwen
from speech_projector.models import (
    EvaluationCondition,
    Example,
    ExperimentStage,
    LinearProjectorConfig,
    RunConfig,
    Split,
)
from speech_projector.projectors import Projector
from speech_projector.training import (
    TrainingResourceRecord,
    ValidationCheckpointRecord,
    gradient_sanity,
    train_run,
    validation_loss,
    weights_digest,
)


class CountingTokenizer(PreTrainedTokenizerFast):
    def decode(self, token_ids: Tensor, skip_special_tokens: bool = False) -> str:
        return str(token_ids.numel())


class EosCheckingQwen(Qwen3_5ForCausalLM):
    def generate(
        self,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        max_new_tokens: int,
        do_sample: bool,
        use_cache: bool,
        pad_token_id: int | None,
        eos_token_id: int | None,
    ) -> Tensor:
        assert eos_token_id == 0
        return torch.tensor([[eos_token_id]], dtype=torch.long)


@pytest.fixture
def wrapper() -> FrozenQwen:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "yes": 1, "no": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    result = FrozenQwen.__new__(FrozenQwen)
    result.config = RunConfig(
        name="tiny",
        stage=ExperimentStage.V0,
        train_examples=2,
        epochs=1,
        learning_rate=0.001,
        gradient_accumulation=2,
        gradient_checkpointing=True,
        projector=LinearProjectorConfig(
            compression_factor=2,
            encoder_dimension=8,
            embedding_dimension=32,
        ),
    )
    result.device = torch.device("cpu")
    result.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    result.model = Qwen3_5ForCausalLM(
        Qwen3_5TextConfig(
            vocab_size=8,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            layer_types=["full_attention", "full_attention"],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [1, 1, 0],
            },
        )
    )
    result.model.requires_grad_(False)
    result.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    result.model.eval()
    return result


@pytest.fixture
def example(tmp_path: Path) -> Example:
    path = tmp_path / "features.pt"
    torch.save(torch.randn(9, 8), path)
    return Example(
        example_id="one",
        dialogue_id="dialogue",
        split=Split.TRAIN,
        history=(),
        user_text="no",
        target_text="yes",
        audio_path=tmp_path / "unused.wav",
        duration=0.18,
        domain="test",
        emotion="neutral",
        feature_path=path,
    )


def test_target_only_logits_match_full_masked_cross_entropy(
    wrapper: FrozenQwen, example: Example
) -> None:
    speech = torch.randn(4, 32, requires_grad=True)
    prepared = wrapper.prepare(example, SpeechInput(speech))
    assert torch.all(prepared.labels[:, : prepared.target_start] == -100)
    assert torch.all(
        prepared.labels[:, prepared.target_start : prepared.target_start + prepared.target_tokens]
        >= 0
    )
    assert torch.all(prepared.labels[:, prepared.target_start + prepared.target_tokens :] == -100)
    assert prepared.embeddings.shape[1] % wrapper.config.sequence_length_multiple == 0
    full = wrapper.model(
        inputs_embeds=prepared.embeddings, attention_mask=prepared.attention_mask, use_cache=False
    )
    expected = functional.cross_entropy(
        full.logits[:, :-1].float().reshape(-1, full.logits.shape[-1]),
        prepared.labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    actual = wrapper.loss(example, SpeechInput(speech))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert speech.grad is not None
    assert speech.grad.abs().sum() > 0


def test_frozen_weights_and_projector_gradient(wrapper: FrozenQwen, example: Example) -> None:
    projector = Projector(wrapper.config.projector)
    check = gradient_sanity(wrapper, projector, example)
    assert check.llm_weights_unchanged
    assert check.projector_changed
    assert not check.llm_has_gradients


def test_speech_input_does_not_expose_current_user_transcript(
    wrapper: FrozenQwen, example: Example
) -> None:
    speech = torch.randn(3, 32)
    original = wrapper.prepare(example, SpeechInput(speech))
    changed = wrapper.prepare(
        example.model_copy(update={"user_text": "yes yes yes"}), SpeechInput(speech)
    )
    torch.testing.assert_close(original.embeddings, changed.embeddings)
    torch.testing.assert_close(original.labels, changed.labels)


class TextRecordingQwen(FrozenQwen):
    def __init__(self, wrapper: FrozenQwen) -> None:
        self.config = wrapper.config
        self.device = wrapper.device
        self.model = wrapper.model
        self.tokenizer = wrapper.tokenizer
        self.loss_inputs: list[TranscriptInput] = []
        self.generation_inputs: list[TranscriptInput] = []

    def loss(self, example: Example, utterance: TranscriptInput | SpeechInput) -> Tensor:
        assert isinstance(utterance, TranscriptInput)
        self.loss_inputs.append(utterance)
        return super().loss(example, utterance)

    def generate(self, example: Example, utterance: TranscriptInput | SpeechInput) -> str:
        assert isinstance(utterance, TranscriptInput)
        self.generation_inputs.append(utterance)
        return "yes"


def test_selected_text_reaches_loss_and_generation_without_mutating_example(
    wrapper: FrozenQwen, example: Example
) -> None:
    recording = TextRecordingQwen(wrapper)
    original = example.model_dump_json()

    def synthesis_text(selected: Example) -> TranscriptInput:
        assert selected is example
        return TranscriptInput("yes yes")

    outcome = evaluate(
        recording,
        None,
        [example],
        wrapper.config,
        EvaluationCondition.TEXT,
        text_input=synthesis_text,
    )
    assert recording.loss_inputs == [TranscriptInput("yes yes")]
    assert recording.generation_inputs == [TranscriptInput("yes yes")]
    assert example.model_dump_json() == original
    assert outcome.samples[0].user_transcript == example.user_text
    assert outcome.metrics.cross_entropy == pytest.approx(
        wrapper.loss(example, TranscriptInput("yes yes")).item()
    )


def test_default_text_factory_preserves_original_dialogue_input(
    wrapper: FrozenQwen, example: Example
) -> None:
    default = TextRecordingQwen(wrapper)
    explicit = TextRecordingQwen(wrapper)
    default_outcome = evaluate(default, None, [example], wrapper.config, EvaluationCondition.TEXT)
    explicit_outcome = evaluate(
        explicit,
        None,
        [example],
        wrapper.config,
        EvaluationCondition.TEXT,
        text_input=original_dialogue_text,
    )
    expected = [TranscriptInput(example.user_text)]
    assert default.loss_inputs == explicit.loss_inputs == expected
    assert default.generation_inputs == explicit.generation_inputs == expected
    assert default_outcome.example_losses == explicit_outcome.example_losses
    assert default_outcome.samples == explicit_outcome.samples


def test_generation_stops_at_tokenizer_eos_when_model_config_disagrees(
    wrapper: FrozenQwen, example: Example
) -> None:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "yes": 1, "no": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    wrapper.tokenizer = CountingTokenizer(
        tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[UNK]", pad_token="no"
    )
    wrapper.model = EosCheckingQwen(wrapper.model.config)
    wrapper.model.config.eos_token_id = 7
    assert wrapper.generate(example, TranscriptInput(example.user_text)) == "1"


def test_finished_resume_preserves_final_and_best_checkpoint_weights(
    wrapper: FrozenQwen, example: Example, tmp_path: Path
) -> None:
    projector = Projector(wrapper.config.projector)
    examples = [example, example.model_copy(update={"example_id": "two"})]
    first = train_run(wrapper.config, examples, [example], tmp_path / "run", wrapper, projector)
    digest = weights_digest(projector)
    record_path = tmp_path / "run" / "best_validation.json"
    record = ValidationCheckpointRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
    best_weights = record.checkpoint_path.read_bytes()
    best_projector = Projector(wrapper.config.projector)
    best_projector.load_state_dict(load_file(str(record.checkpoint_path)))
    assert record.step == first.steps
    assert validation_loss(wrapper, best_projector, [example]) == pytest.approx(
        record.cross_entropy
    )
    resources_path = tmp_path / "run" / "training_resources.json"
    resources_path.write_text(
        TrainingResourceRecord(peak_vram_gb=4.15).model_dump_json(), encoding="utf-8"
    )
    second = train_run(wrapper.config, examples, [example], tmp_path / "run", wrapper, projector)
    assert first.steps == second.steps == 1
    assert weights_digest(projector) == digest
    assert record.checkpoint_path.read_bytes() == best_weights
    assert (
        ValidationCheckpointRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
        == record
    )
    assert second.peak_vram_gb == 4.15
    assert (
        TrainingResourceRecord.model_validate_json(
            resources_path.read_text(encoding="utf-8")
        ).peak_vram_gb
        == 4.15
    )


@pytest.mark.parametrize("padding", [1, 5])
def test_right_padding_does_not_change_real_logits(
    wrapper: FrozenQwen, example: Example, padding: int
) -> None:
    prepared = wrapper.prepare(example, SpeechInput(torch.randn(3, 32)))
    original = wrapper.model(
        inputs_embeds=prepared.embeddings, attention_mask=prepared.attention_mask, use_cache=False
    )
    padded: Tensor = functional.pad(prepared.embeddings, (0, 0, 0, padding))
    mask = functional.pad(prepared.attention_mask, (0, padding))
    output = wrapper.model(inputs_embeds=padded, attention_mask=mask, use_cache=False)
    torch.testing.assert_close(original.logits, output.logits[:, : prepared.embeddings.shape[1]])
