from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import Tensor
from torch.nn import functional as functional
from transformers import (
    GenerationConfig,
    LogitsProcessorList,
    PreTrainedTokenizerFast,
    Qwen3_5ForCausalLM,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from speech_projector.evaluation import evaluate, original_dialogue_text
from speech_projector.generation import CompletedGeneration, TokenLimitedGeneration
from speech_projector.inputs import SpeechInput, TranscriptInput
from speech_projector.llm import FrozenQwen, example_prompt, generation_seed
from speech_projector.models import (
    ChatPromptConfig,
    EvaluationCondition,
    Example,
    ExperimentStage,
    GenerationKind,
    LinearProjectorConfig,
    RunConfig,
    RunPromptConfig,
    SamplingDecodingConfig,
    Split,
    SystemPromptConfig,
)
from speech_projector.overnight_continuation import prepare_continuation
from speech_projector.projectors import Projector
from speech_projector.training import (
    BatchedParityPolicy,
    TrainingResourceRecord,
    TrainingState,
    ValidationCheckpointRecord,
    batched_gradient_sanity,
    gradient_sanity,
    train_run,
    validate_batched_parity,
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
        generation_config: GenerationConfig,
        logits_processor: LogitsProcessorList,
    ) -> Tensor:
        assert generation_config.eos_token_id == 0
        assert not generation_config.do_sample
        assert generation_config.use_cache
        assert not logits_processor
        return torch.tensor([[generation_config.eos_token_id]], dtype=torch.long)


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


def test_custom_system_prompt_is_shared_by_transcript_and_speech(
    wrapper: FrozenQwen, example: Example
) -> None:
    original_text = wrapper.prepare(example, TranscriptInput(example.user_text))
    original_speech = wrapper.prepare(example, SpeechInput(torch.zeros(3, 32)))
    wrapper.config = wrapper.config.model_copy(
        update={"prompt": SystemPromptConfig(system_text="yes yes")}
    )
    changed_text = wrapper.prepare(example, TranscriptInput(example.user_text))
    changed_speech = wrapper.prepare(example, SpeechInput(torch.zeros(3, 32)))
    difference = original_text.target_start - changed_text.target_start
    assert difference != 0
    assert original_speech.target_start - changed_speech.target_start == difference
    torch.testing.assert_close(changed_text.embeddings[0, :2], changed_speech.embeddings[0, :2])


def test_standard_chat_omits_system_message_for_both_input_branches(
    wrapper: FrozenQwen, example: Example
) -> None:
    previous_text = wrapper.prepare(example, TranscriptInput(example.user_text))
    speech = torch.zeros(3, 32)
    previous_speech = wrapper.prepare(example, SpeechInput(speech))
    wrapper.config = wrapper.config.model_copy(update={"prompt": ChatPromptConfig()})
    chat_text = wrapper.prepare(example, TranscriptInput(example.user_text))
    chat_speech = wrapper.prepare(example, SpeechInput(speech))
    expected = wrapper._embed(wrapper._encode("<|im_start|>user\n"))
    torch.testing.assert_close(chat_text.embeddings[0, : expected.shape[0]], expected)
    torch.testing.assert_close(chat_speech.embeddings[0, : expected.shape[0]], expected)
    removed = previous_text.target_start - chat_text.target_start
    assert removed > 0
    assert previous_speech.target_start - chat_speech.target_start == removed


def test_target_scores_match_loss_and_preserve_target_ids(
    wrapper: FrozenQwen, example: Example
) -> None:
    utterance = TranscriptInput(example.user_text)
    scores = wrapper.score_target(example, utterance)
    assert scores.logits.shape[0] == scores.target_token_ids.shape[0]
    expected = functional.cross_entropy(scores.logits.float(), scores.target_token_ids)
    torch.testing.assert_close(wrapper.loss(example, utterance), expected)


def test_left_padding_batch_preserves_each_prompt_and_real_next_logits(
    wrapper: FrozenQwen, example: Example
) -> None:
    examples = (example, example.model_copy(update={"user_text": "yes yes yes"}))
    utterances = tuple(TranscriptInput(item.user_text) for item in examples)
    batch = wrapper.prepare_generation_batch(examples, utterances)
    assert batch.embeddings.shape[1] % wrapper.config.sequence_length_multiple == 0
    positions = (batch.attention_mask.cumsum(-1) - 1).clamp_min(0)
    batched = wrapper.model(
        inputs_embeds=batch.embeddings,
        attention_mask=batch.attention_mask,
        position_ids=positions,
        use_cache=False,
        logits_to_keep=1,
    ).logits
    for index, (item, utterance) in enumerate(zip(examples, utterances, strict=True)):
        prompt = wrapper._prompt(item, utterance)
        torch.testing.assert_close(batch.embeddings[index, -prompt.shape[0] :], prompt)
        assert int(batch.attention_mask[index].sum()) == prompt.shape[0]
        original = wrapper.model(
            inputs_embeds=prompt.unsqueeze(0), use_cache=False, logits_to_keep=1
        ).logits
        torch.testing.assert_close(batched[index], original[0])


class BatchEosQwen(Qwen3_5ForCausalLM):
    def generate(
        self,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        generation_config: GenerationConfig,
        logits_processor: LogitsProcessorList,
    ) -> Tensor:
        assert generation_config.eos_token_id == 0
        assert generation_config.max_new_tokens == 3
        assert not generation_config.do_sample
        assert generation_config.use_cache
        assert not logits_processor
        assert inputs_embeds.shape[0] == 2
        assert torch.all(attention_mask[:, -1] == 1)
        assert torch.any(attention_mask[:, 0] == 0)
        return torch.tensor([[1, 0, 2], [2, 2, 2]], dtype=torch.long)


def test_batch_generation_distinguishes_completed_eos_and_cap_hit(
    wrapper: FrozenQwen, example: Example
) -> None:
    wrapper.tokenizer.eos_token = "[UNK]"
    wrapper.tokenizer.pad_token = "no"
    wrapper.model = BatchEosQwen(wrapper.model.config)
    examples = (example, example.model_copy(update={"user_text": "yes yes yes"}))
    results = wrapper.generate_batch(
        examples, tuple(TranscriptInput(item.user_text) for item in examples), 3
    )
    assert results[0] == CompletedGeneration(text="yes", token_ids=(1, 0))
    assert results[1] == TokenLimitedGeneration(partial_text="no no no", token_ids=(2, 2, 2))


def test_batched_speech_evaluation_saves_completion_and_original_order(
    wrapper: FrozenQwen, example: Example
) -> None:
    wrapper.tokenizer.eos_token = "[UNK]"
    wrapper.tokenizer.pad_token = "no"
    wrapper.model = BatchEosQwen(wrapper.model.config)
    wrapper.config = wrapper.config.model_copy(
        update={
            "generation_batch_size": 2,
            "max_new_tokens": 3,
            "semantic_examples": 2,
            "qualitative_examples": 2,
        }
    )
    second = example.model_copy(update={"example_id": "two", "user_text": "unused transcript"})
    outcome = evaluate(
        wrapper,
        Projector(wrapper.config.projector),
        (example, second),
        wrapper.config,
        EvaluationCondition.SPEECH,
        diagnostics=False,
    )
    assert tuple(sample.example_id for sample in outcome.samples) == ("one", "two")
    assert tuple(sample.generated_response for sample in outcome.samples) == ("yes", "no no no")
    assert outcome.metrics.completed_generations == 1
    assert outcome.metrics.token_limited_generations == 1
    assert outcome.metrics.generated_tokens == 5
    assert outcome.samples[0].generation is not None
    assert outcome.samples[0].generation.kind == GenerationKind.COMPLETED
    assert outcome.samples[1].generation is not None
    assert outcome.samples[1].generation.kind == GenerationKind.TOKEN_LIMIT


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
    assert outcome.samples[0].user_transcript == "yes yes"
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


def test_best_checkpoint_compares_the_same_subset_while_final_score_uses_full_validation(
    wrapper: FrozenQwen,
    example: Example,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = wrapper.config.model_copy(
        update={"epochs": 2, "evaluation_interval": 1, "training_validation_examples": 1}
    )
    wrapper.config = config
    training_examples = [example, example.model_copy(update={"example_id": "train-two"})]
    validation_examples = [
        example.model_copy(update={"example_id": "validation-one", "split": Split.VALIDATION}),
        example.model_copy(update={"example_id": "validation-two", "split": Split.VALIDATION}),
    ]
    subset_scores = iter((2.0, 1.0, 1.5, 1.6))
    training_scores = iter((5.0, 2.0))
    populations: list[tuple[Split, int]] = []

    def deterministic_validation_loss(
        selected_wrapper: FrozenQwen,
        selected_projector: Projector,
        selected_examples: list[Example],
    ) -> float:
        assert selected_wrapper is wrapper
        assert selected_projector is projector
        split = selected_examples[0].split
        populations.append((split, len(selected_examples)))
        if split == Split.TRAIN:
            return next(training_scores)
        return next(subset_scores) if len(selected_examples) == 1 else 0.1

    monkeypatch.setattr("speech_projector.training.validation_loss", deterministic_validation_loss)
    projector = Projector(config.projector)
    output_directory = tmp_path / "selection"
    outcome = train_run(
        config, training_examples, validation_examples, output_directory, wrapper, projector
    )
    best = ValidationCheckpointRecord.model_validate_json(
        (output_directory / "best_validation.json").read_bytes()
    )
    assert best.step == 1
    assert best.cross_entropy == 1.0
    assert outcome.steps == 2
    assert populations == [
        (Split.VALIDATION, 1),
        (Split.TRAIN, 2),
        (Split.VALIDATION, 1),
        (Split.VALIDATION, 1),
        (Split.VALIDATION, 2),
        (Split.VALIDATION, 1),
        (Split.TRAIN, 2),
    ]
    final_state = TrainingState.model_validate_json(
        (output_directory / "checkpoint" / "state.json").read_bytes()
    )
    assert final_state.final_validation_loss == 0.1
    best_projector = Projector(config.projector)
    best_projector.load_state_dict(load_file(str(best.checkpoint_path)))
    assert weights_digest(best_projector) != weights_digest(projector)


class SamplingCheckingQwen(Qwen3_5ForCausalLM):
    def generate(
        self,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        generation_config: GenerationConfig,
        logits_processor: LogitsProcessorList,
    ) -> Tensor:
        assert generation_config.do_sample
        assert generation_config.temperature == 1
        assert generation_config.top_p == 1
        assert generation_config.top_k == 20
        assert generation_config.use_cache
        assert generation_config.eos_token_id == 0
        assert len(logits_processor) == 1
        tokens = torch.randint(1, 3, (inputs_embeds.shape[0], 8))
        return torch.cat(
            (tokens, torch.zeros((inputs_embeds.shape[0], 1), dtype=torch.long)), dim=1
        )


def test_sampling_is_batch_deterministic_and_restores_external_rng(
    wrapper: FrozenQwen, example: Example
) -> None:
    wrapper.config = wrapper.config.model_copy(update={"decoding": SamplingDecodingConfig()})
    wrapper.tokenizer.eos_token = "[UNK]"
    wrapper.tokenizer.pad_token = "no"
    wrapper.model = SamplingCheckingQwen(wrapper.model.config)
    original_rng = torch.random.get_rng_state().clone()
    speech = SpeechInput(torch.zeros(3, 32))
    first = wrapper.generate_batch((example,), (speech,), wrapper.config.max_new_tokens)
    assert torch.equal(torch.random.get_rng_state(), original_rng)
    torch.manual_seed(987654)
    changed_rng = torch.random.get_rng_state().clone()
    repeated = wrapper.generate_batch((example,), (speech,), wrapper.config.max_new_tokens)
    assert repeated == first
    assert torch.equal(torch.random.get_rng_state(), changed_rng)
    transcript = wrapper.generate_batch(
        (example,), (TranscriptInput(example.user_text),), wrapper.config.max_new_tokens
    )
    assert transcript == first
    assert isinstance(first[0], CompletedGeneration)
    assert wrapper.generate(example, speech) == first[0].text
    assert torch.equal(torch.random.get_rng_state(), changed_rng)


def test_generation_seed_depends_on_seed_ordered_identifiers_and_token_cap(
    wrapper: FrozenQwen, example: Example
) -> None:
    other = example.model_copy(update={"example_id": "two"})
    baseline = generation_seed(wrapper.config, (example, other), 100)
    assert baseline == generation_seed(wrapper.config, (example, other), 100)
    assert baseline != generation_seed(wrapper.config, (other, example), 100)
    assert baseline != generation_seed(wrapper.config, (example, other), 101)
    assert baseline != generation_seed(
        wrapper.config.model_copy(update={"seed": wrapper.config.seed + 1}), (example, other), 100
    )
    changed_text = example.model_copy(update={"user_text": "different", "target_text": "different"})
    assert baseline == generation_seed(wrapper.config, (changed_text, other), 100)


@pytest.mark.parametrize("prompt", (ChatPromptConfig(), SystemPromptConfig(system_text="no yes")))
def test_explicit_example_prompt_overrides_run_prompt_for_both_inputs(
    wrapper: FrozenQwen, example: Example, prompt: ChatPromptConfig | SystemPromptConfig
) -> None:
    copied = example.model_copy(update={"prompt": prompt})
    assert example_prompt(copied, wrapper.config) == prompt
    assert example_prompt(example, wrapper.config) == wrapper.config.prompt
    assert isinstance(example.prompt, RunPromptConfig)
    reference = FrozenQwen.__new__(FrozenQwen)
    reference.config = wrapper.config.model_copy(update={"prompt": prompt})
    reference.device = wrapper.device
    reference.tokenizer = wrapper.tokenizer
    reference.model = wrapper.model
    for utterance in (TranscriptInput(example.user_text), SpeechInput(torch.randn(3, 32))):
        assert torch.equal(
            wrapper._prompt(copied, utterance), reference._prompt(example, utterance)
        )
    changed_label = copied.model_copy(update={"emotion": "unspoken privileged delivery"})
    speech = SpeechInput(torch.randn(3, 32))
    assert torch.equal(wrapper._prompt(copied, speech), wrapper._prompt(changed_label, speech))


def test_batched_target_chunk_loss_and_gradient_match_individual_example_average(
    wrapper: FrozenQwen, example: Example
) -> None:
    other = example.model_copy(update={"example_id": "two", "target_text": "yes no yes"})
    first = torch.randn(3, 32, requires_grad=True)
    second = torch.randn(7, 32, requires_grad=True)
    separate = (
        wrapper.loss(example, SpeechInput(first)) + wrapper.loss(other, SpeechInput(second))
    ) / 2
    separate.backward()
    expected_first = first.grad.clone()
    expected_second = second.grad.clone()
    first.grad = None
    second.grad = None
    batched = wrapper.loss_batch((example, other), (SpeechInput(first), SpeechInput(second)))
    assert torch.allclose(batched, separate, atol=1e-6)
    batched.backward()
    assert torch.allclose(first.grad, expected_first, atol=1e-6)
    assert torch.allclose(second.grad, expected_second, atol=1e-6)
    assert all(parameter.grad is None for parameter in wrapper.model.parameters())


def test_fixed_update_budget_preserves_mid_epoch_offset_and_finished_resume(
    wrapper: FrozenQwen, example: Example, tmp_path: Path
) -> None:
    config = wrapper.config.model_copy(
        update={
            "train_examples": 6,
            "epochs": 3,
            "max_optimizer_updates": 1,
            "microbatch_size": 2,
            "gradient_accumulation": 2,
        }
    )
    wrapper.config = config
    examples = [example.model_copy(update={"example_id": str(index)}) for index in range(6)]
    projector = Projector(config.projector)
    directory = tmp_path / "fixed"
    outcome = train_run(config, examples, [example], directory, wrapper, projector)
    state = TrainingState.model_validate_json(
        (directory / "checkpoint" / "state.json").read_bytes()
    )
    assert (state.step, state.epoch, state.offset, state.examples_seen) == (1, 0, 4, 4)
    before = weights_digest(projector)
    resumed = train_run(config, examples, [example], directory, wrapper, projector)
    assert resumed.steps == outcome.steps == 1
    assert weights_digest(projector) == before
    assert len((directory / "train.jsonl").read_text().splitlines()) == 1


def test_continuation_preserves_optimizer_cursor_and_exact_training_trajectory(
    wrapper: FrozenQwen, example: Example, tmp_path: Path
) -> None:
    config = wrapper.config.model_copy(
        update={"train_examples": 6, "epochs": 3, "max_optimizer_updates": 2}
    )
    examples = [example.model_copy(update={"example_id": str(index)}) for index in range(6)]
    wrapper.config = config
    torch.manual_seed(42)
    uninterrupted = Projector(config.projector)
    train_run(config, examples, [example], tmp_path / "full", wrapper, uninterrupted)
    short = config.model_copy(update={"name": "short", "max_optimizer_updates": 1})
    wrapper.config = short
    torch.manual_seed(42)
    projector = Projector(config.projector)
    source = tmp_path / "source"
    train_run(short, examples, [example], source, wrapper, projector)
    original_state = (source / "checkpoint" / "state.json").read_bytes()
    destination = tmp_path / "continued"
    continuation = config.model_copy(update={"name": "continued"})
    receipt = prepare_continuation(source, destination, continuation)
    assert prepare_continuation(source, destination, continuation) == receipt
    copied = TrainingState.model_validate_json(
        (destination / "checkpoint" / "state.json").read_bytes()
    )
    assert copied.offset == 2 and copied.step == 1 and copied.final_validation_loss is None
    wrapper.config = continuation
    train_run(continuation, examples, [example], destination, wrapper, projector)
    assert weights_digest(projector) == weights_digest(uninterrupted)
    assert (source / "checkpoint" / "state.json").read_bytes() == original_state


def test_batched_gradient_gate_compares_same_weights_and_rejects_misalignment(
    wrapper: FrozenQwen, example: Example
) -> None:
    projector = Projector(wrapper.config.projector)
    other = example.model_copy(update={"example_id": "two", "target_text": "yes no yes"})
    check = batched_gradient_sanity(wrapper, projector, [example, other])
    assert check.absolute_loss_difference < 1e-6
    assert check.gradient_cosine > 0.99999
    assert check.gradient_norm_ratio == pytest.approx(1, abs=1e-5)
    assert check.llm_weights_unchanged and check.projector_weights_unchanged
    assert not check.llm_has_gradients
    validate_batched_parity(check, BatchedParityPolicy())
    with pytest.raises(ValueError, match="parity failed"):
        validate_batched_parity(
            check.model_copy(update={"gradient_cosine": 0.5}), BatchedParityPolicy()
        )
