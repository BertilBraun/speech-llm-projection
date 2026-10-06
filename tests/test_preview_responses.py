from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import Tensor
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessorList,
    PreTrainedTokenizerFast,
    Qwen3_5ForCausalLM,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from speech_projector.emotion_preview import Delivery, default_preview_plan
from speech_projector.generation import CompletedGeneration, TokenLimitedGeneration
from speech_projector.models import SamplingDecodingConfig
from speech_projector.preview_responses import (
    PREVIEW_SYSTEM,
    DeliveryPreviewRequest,
    PreviewRequest,
    PreviewResponse,
    PreviewResponseAttempt,
    PreviewResponseConfig,
    PreviewResponseProvenance,
    PreviewTeacher,
    TranscriptPreviewRequest,
    decode_preview_tokens,
    generate_preview_attempts,
    preview_messages,
    preview_requests,
    render_preview_responses,
    request_id,
    run_preview_requests,
)


def configuration(directory: Path) -> PreviewResponseConfig:
    return PreviewResponseConfig(
        plan_path=directory / "plan.json",
        output_directory=directory / "responses",
        revision="a" * 40,
        source_git_commit="b" * 40,
    )


def fixture_response(request: PreviewRequest) -> PreviewResponse:
    return PreviewResponse(
        request=request,
        messages=preview_messages(request, PREVIEW_SYSTEM),
        prompt_text="test-local formatted chat",
        prompt_token_ids=(1, 2),
        attempts=(
            PreviewResponseAttempt(
                token_cap=256,
                seed=42,
                runtime_seconds=0.1,
                generation=CompletedGeneration(
                    text="How are you feeling about it?", token_ids=(3, 4)
                ),
            ),
        ),
    )


def test_fixed_cases_keep_literal_text_and_add_only_two_transcript_controls() -> None:
    plan = default_preview_plan()
    requests = preview_requests(plan)
    assert len(requests) == 12
    assert len({request_id(item) for item in requests}) == 12
    for case, request in zip(plan.cases, requests[:10], strict=True):
        assert request == DeliveryPreviewRequest(case=case)
        assert case.text in ("I'm good.", "That went really well.")
    assert requests[-2:] == (
        TranscriptPreviewRequest(text_id="transcript_1", text="I'm good."),
        TranscriptPreviewRequest(text_id="transcript_2", text="That went really well."),
    )


def test_controls_share_system_prompt_without_delivery_metadata_or_training_fields() -> None:
    plan = default_preview_plan()
    case = next(item for item in plan.cases if item.delivery == Delivery.SARCASTIC)
    aware = preview_messages(DeliveryPreviewRequest(case=case), PREVIEW_SYSTEM)
    control = preview_messages(
        TranscriptPreviewRequest(text_id="control", text=case.text), PREVIEW_SYSTEM
    )
    assert aware[0] == control[0]
    assert '<current_message>\n"I\'m good."\n</current_message>' in aware[1].content
    assert "<intended_delivery>" in aware[1].content
    assert case.instruct in aware[1].content
    assert "<intended_delivery>" not in control[1].content
    assert "do not assume sarcasm means sadness" in aware[0].content
    assert plan == default_preview_plan()


@pytest.mark.parametrize("completed", (True, False))
def test_retry_keeps_capped_attempt_and_uses_larger_budget(tmp_path: Path, completed: bool) -> None:
    config = configuration(tmp_path)
    calls: list[int] = []

    def generate(cap: int) -> PreviewResponseAttempt:
        calls.append(cap)
        response = (
            CompletedGeneration(text="Reply.", token_ids=(1, 2))
            if completed or cap == 512
            else TokenLimitedGeneration(partial_text="Partial", token_ids=(1,))
        )
        return PreviewResponseAttempt(
            token_cap=cap, seed=42, runtime_seconds=1, generation=response
        )

    attempts = generate_preview_attempts(config, generate)
    assert calls == ([256] if completed else [256, 512])
    assert isinstance(attempts[-1].generation, CompletedGeneration)
    if not completed:
        assert isinstance(attempts[0].generation, TokenLimitedGeneration)


def test_retry_does_not_hide_a_second_capped_generation(tmp_path: Path) -> None:
    def capped(cap: int) -> PreviewResponseAttempt:
        return PreviewResponseAttempt(
            token_cap=cap,
            seed=42,
            runtime_seconds=1,
            generation=TokenLimitedGeneration(partial_text="Still incomplete", token_ids=(1,)),
        )

    attempts = generate_preview_attempts(configuration(tmp_path), capped)
    assert len(attempts) == 2
    assert isinstance(attempts[-1].generation, TokenLimitedGeneration)


@pytest.mark.parametrize("tokens,completed", (((1, 2, 0), True), ((1,), False)))
def test_eos_and_caps_use_actual_ids_instead_of_response_text(
    tokens: tuple[int, ...], completed: bool
) -> None:
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[UNK]": 0, "reply": 1, "[EOS]": 2}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
        eos_token="[EOS]",
    )
    result = decode_preview_tokens(tokens, tokenizer)
    assert isinstance(result, CompletedGeneration) == completed
    assert result.token_ids == ((1, 2) if completed else (1,))


def test_journal_resume_reuses_exact_requests_and_rejects_changed_provenance(
    tmp_path: Path,
) -> None:
    config = configuration(tmp_path)
    provenance = PreviewResponseProvenance(
        configuration=config, plan=default_preview_plan(), plan_sha256="plan-hash"
    )
    calls: list[str] = []

    def respond(request: PreviewRequest) -> PreviewResponse:
        calls.append(request_id(request))
        return fixture_response(request)

    records = run_preview_requests(provenance, respond)
    assert len(calls) == 12
    assert run_preview_requests(provenance, respond) == records
    assert len(calls) == 12
    changed = provenance.model_copy(update={"plan_sha256": "changed-plan"})
    with pytest.raises(ValueError, match="provenance differs"):
        run_preview_requests(changed, respond)
    assert len(calls) == 12


def test_journal_rejects_wrong_request_even_with_original_provenance(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    provenance = PreviewResponseProvenance(
        configuration=config, plan=default_preview_plan(), plan_sha256="plan-hash"
    )
    records = run_preview_requests(provenance, fixture_response)
    wrong = records[0].model_copy(update={"request": records[1].request})
    (config.output_directory / "responses.jsonl").write_text(
        wrong.model_dump_json() + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="input requests"):
        run_preview_requests(provenance, fixture_response)


def test_invalid_budget_fails_before_generation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exceed"):
        PreviewResponseConfig(
            plan_path=tmp_path / "plan",
            output_directory=tmp_path,
            revision="a" * 40,
            source_git_commit="b" * 40,
            initial_token_cap=512,
            retry_token_cap=256,
        )
    assert configuration(tmp_path).decoding == SamplingDecodingConfig()


def test_readable_comparison_preserves_metadata_only_limit_and_controls() -> None:
    records = tuple(fixture_response(item) for item in preview_requests(default_preview_plan()))
    readable = render_preview_responses(records)
    assert "Qwen did not hear the audio" in readable
    assert "not emotion-recognition measurements" in readable
    assert readable.count("## ") == 12
    assert readable.count("Transcript-only control") == 2


class CheckingPreviewModel(Qwen3_5ForCausalLM):
    def generate(
        self,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        generation_config: GenerationConfig,
        logits_processor: LogitsProcessorList,
    ) -> Tensor:
        assert not torch.is_grad_enabled()
        assert not self.training
        assert inputs_embeds.shape[:2] == attention_mask.shape
        assert attention_mask.eq(1).all()
        assert generation_config.do_sample
        assert generation_config.temperature == 1
        assert generation_config.top_p == 1
        assert generation_config.top_k == 20
        assert generation_config.eos_token_id == 2
        assert len(logits_processor) == 1
        scores = torch.zeros((1, self.config.vocab_size))
        assert torch.equal(logits_processor(torch.empty((1, 0), dtype=torch.long), scores), scores)
        torch.rand(1)
        return torch.tensor([[1, 2]], dtype=torch.long)


def test_pinned_inference_freezes_weights_uses_native_chat_and_restores_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = configuration(tmp_path)
    vocabulary = Tokenizer(WordLevel({"[UNK]": 0, "reply": 1, "[EOS]": 2}, unk_token="[UNK]"))
    vocabulary.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=vocabulary, unk_token="[UNK]", eos_token="[EOS]", pad_token="[UNK]"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{message.role}}:{{message.content}}\n{% endfor %}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
        "{% if not enable_thinking %}<think></think>{% endif %}"
    )
    text_config = Qwen3_5TextConfig(
        vocab_size=8,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000, "mrope_section": [1, 1, 0]},
    )
    model = CheckingPreviewModel(text_config)

    def tokenizer_provider(model_name: str, revision: str) -> PreTrainedTokenizerFast:
        assert (model_name, revision) == (config.model_name, config.revision)
        return tokenizer

    def config_provider(model_name: str, revision: str) -> Qwen3_5TextConfig:
        assert (model_name, revision) == (config.model_name, config.revision)
        return text_config

    def model_provider(
        model_name: str,
        *,
        revision: str,
        config: Qwen3_5TextConfig,
        dtype: torch.dtype,
        attn_implementation: str,
    ) -> CheckingPreviewModel:
        assert model_name == "Qwen/Qwen3.5-2B"
        assert revision == "a" * 40
        assert config is text_config
        assert dtype == torch.float32
        assert attn_implementation == "sdpa"
        return model

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer_provider)
    monkeypatch.setattr(Qwen3_5TextConfig, "from_pretrained", config_provider)
    monkeypatch.setattr(Qwen3_5ForCausalLM, "from_pretrained", model_provider)
    teacher = PreviewTeacher(config, torch.device("cpu"))
    assert all(not parameter.requires_grad for parameter in teacher.model.parameters())
    before = tuple(parameter.detach().clone() for parameter in model.parameters())
    random_state = torch.random.get_rng_state().clone()
    response = teacher.respond(preview_requests(default_preview_plan())[0])
    assert "<think></think>" in response.prompt_text
    assert response.prompt_token_ids == tuple(
        tokenizer.encode(response.prompt_text, add_special_tokens=False)
    )
    assert isinstance(response.attempts[0].generation, CompletedGeneration)
    assert response.attempts[0].generation.token_ids == (1, 2)
    assert torch.equal(random_state, torch.random.get_rng_state())
    assert all(
        torch.equal(left, right) for left, right in zip(before, model.parameters(), strict=True)
    )
