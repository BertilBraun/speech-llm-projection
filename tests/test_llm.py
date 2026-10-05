from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import Tensor
from torch.nn import functional as functional
from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from speech_projector.llm import FrozenQwen
from speech_projector.models import Architecture, Example, ProjectorConfig, RunConfig, Split
from speech_projector.projectors import Projector
from speech_projector.training import gradient_sanity, train_run, weights_digest


@pytest.fixture
def wrapper() -> FrozenQwen:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "yes": 1, "no": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    result = FrozenQwen.__new__(FrozenQwen)
    result.config = RunConfig(
        name="tiny",
        stage="test",
        train_examples=2,
        epochs=1,
        learning_rate=0.001,
        gradient_accumulation=2,
        gradient_checkpointing=True,
        projector=ProjectorConfig(
            architecture=Architecture.LINEAR,
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
    prepared = wrapper.prepare(example, speech_embeddings=speech)
    assert torch.all(prepared.labels[:, : prepared.target_start] == -100)
    assert torch.all(prepared.labels[:, prepared.target_start :] >= 0)
    full = wrapper.model(
        inputs_embeds=prepared.embeddings, attention_mask=prepared.attention_mask, use_cache=False
    )
    expected = functional.cross_entropy(
        full.logits[:, :-1].float().reshape(-1, full.logits.shape[-1]),
        prepared.labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    actual = wrapper.loss(example, speech_embeddings=speech)
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


def test_finished_run_resume_does_not_repeat_updates(
    wrapper: FrozenQwen, example: Example, tmp_path: Path
) -> None:
    projector = Projector(wrapper.config.projector)
    examples = [example, example.model_copy(update={"example_id": "two"})]
    first = train_run(wrapper.config, examples, [example], tmp_path / "run", wrapper, projector)
    digest = weights_digest(projector)
    second = train_run(wrapper.config, examples, [example], tmp_path / "run", wrapper, projector)
    assert first.steps == second.steps == 1
    assert weights_digest(projector) == digest


@pytest.mark.parametrize("padding", [1, 5])
def test_right_padding_does_not_change_real_logits(
    wrapper: FrozenQwen, example: Example, padding: int
) -> None:
    prepared = wrapper.prepare(example, speech_embeddings=torch.randn(3, 32))
    original = wrapper.model(
        inputs_embeds=prepared.embeddings, attention_mask=prepared.attention_mask, use_cache=False
    )
    padded: Tensor = functional.pad(prepared.embeddings, (0, 0, 0, padding))
    mask = functional.pad(prepared.attention_mask, (0, padding))
    output = wrapper.model(inputs_embeds=padded, attention_mask=mask, use_cache=False)
    torch.testing.assert_close(original.logits, output.logits[:, : prepared.embeddings.shape[1]])
