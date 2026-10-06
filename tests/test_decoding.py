import warnings

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch import Tensor
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LogitsProcessorList,
    PreTrainedTokenizerFast,
)

from speech_projector.decoding import (
    PresencePenaltyLogitsProcessor,
    generation_parameters,
    generation_processors,
)
from speech_projector.models import GreedyDecodingConfig, SamplingDecodingConfig


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "speech": 3}
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="[UNK]")),
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_presence_penalty_is_once_per_generated_token_and_independent_per_row(
    dtype: torch.dtype,
) -> None:
    processor = PresencePenaltyLogitsProcessor(2.0)
    scores = torch.tensor([[4.0, -3.0, 0.0, 1.0], [4.0, -3.0, 0.0, 1.0]], dtype=dtype)
    generated = torch.tensor([[0, 0, 2], [1, 3, 1]], dtype=torch.long)
    expected = torch.tensor([[2.0, -3.0, -2.0, 1.0], [4.0, -5.0, 0.0, -1.0]], dtype=dtype)
    original = scores.clone()
    assert torch.equal(processor(generated, scores), expected)
    assert torch.equal(scores, original)
    assert processor(generated, scores).dtype == dtype


def test_empty_generated_prefix_preserves_first_step_logits() -> None:
    scores = torch.tensor([[1.0, float("-inf"), 4.0], [2.0, 0.0, -3.0]])
    input_ids = torch.empty((2, 0), dtype=torch.long)
    assert torch.equal(PresencePenaltyLogitsProcessor(2.0)(input_ids, scores), scores)


@pytest.mark.parametrize("penalty", (float("nan"), float("inf"), float("-inf")))
def test_nonfinite_penalty_is_rejected(penalty: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        PresencePenaltyLogitsProcessor(penalty)


def test_greedy_generation_uses_no_unused_sampling_options(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    with warnings.catch_warnings(record=True) as captured:
        parameters = generation_parameters(GreedyDecodingConfig(), 16, tokenizer)
        parameters.validate()
    assert not captured
    assert parameters.do_sample is False
    assert parameters.max_new_tokens == 16
    assert parameters.pad_token_id == 1
    assert parameters.eos_token_id == 2
    assert parameters.use_cache is True
    assert not generation_processors(GreedyDecodingConfig())


def test_qwen_sampling_parameters_and_presence_processor(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    config = SamplingDecodingConfig()
    parameters = generation_parameters(config, 2048, tokenizer)
    assert parameters.do_sample is True
    assert parameters.temperature == 1.0
    assert parameters.top_p == 1.0
    assert parameters.top_k == 20
    assert parameters.min_p == 0.0
    assert parameters.repetition_penalty == 1.0
    processed = generation_processors(config)(
        torch.tensor([[3, 3]], dtype=torch.long), torch.zeros((1, 4))
    )
    assert torch.equal(processed, torch.tensor([[0.0, 0.0, 0.0, -2.0]]))


def test_invalid_generation_cap_and_missing_eos_fail_clearly(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        generation_parameters(GreedyDecodingConfig(), 0, tokenizer)
    tokenizer.eos_token = None
    with pytest.raises(ValueError, match="padding and EOS"):
        generation_parameters(GreedyDecodingConfig(), 16, tokenizer)


class RecordingPresencePenalty(PresencePenaltyLogitsProcessor):
    def __init__(self) -> None:
        super().__init__(2.0)
        self.generated_prefixes: list[Tensor] = []

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        self.generated_prefixes.append(input_ids.clone())
        return super().__call__(input_ids, scores)


def test_hf_inputs_embeds_processor_receives_only_generated_tokens(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=4,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=0,
            eos_token_id=2,
            pad_token_id=1,
        )
    ).eval()
    processor = RecordingPresencePenalty()
    parameters = generation_parameters(GreedyDecodingConfig(), 3, tokenizer)
    parameters.min_new_tokens = 3
    with torch.no_grad():
        generated: Tensor = model.generate(
            inputs_embeds=model.get_input_embeddings()(torch.tensor([[3, 3, 3, 3, 3]])),
            attention_mask=torch.ones((1, 5), dtype=torch.long),
            generation_config=parameters,
            logits_processor=LogitsProcessorList([processor]),
        )
    assert tuple(prefix.shape for prefix in processor.generated_prefixes) == (
        torch.Size((1, 0)),
        torch.Size((1, 1)),
        torch.Size((1, 2)),
    )
    assert torch.equal(processor.generated_prefixes[1], generated[:, :1])
    assert torch.equal(processor.generated_prefixes[2], generated[:, :2])
