"""Small, resumable text-and-delivery teacher preview without training inputs."""

import hashlib
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

import torch
from pydantic import Field, model_validator
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Qwen3_5ForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from speech_projector.decoding import generation_parameters, generation_processors
from speech_projector.emotion_preview import PreviewCase, PreviewPlan
from speech_projector.generation import (
    CompletedGeneration,
    GenerationResult,
    TokenLimitedGeneration,
)
from speech_projector.journal import append_record, read_journal
from speech_projector.models import Record, SamplingDecodingConfig

PREVIEW_SYSTEM = (
    "Respond to the user's message as a conversational assistant in one or two concise sentences. "
    "Stay grounded in what the user actually said; do not invent events, personal "
    "experiences, or reasons for their feelings. Do not discuss your own voice, "
    "tone-production abilities, or roleplay. Delivery metadata describes the user's "
    "tone and may be ambiguous. Adapt gently, acknowledge "
    "uncertainty when useful, and do not assume sarcasm means sadness."
)


class DeliveryPreviewRequest(Record):
    kind: Literal["delivery"] = "delivery"
    case: PreviewCase


class TranscriptPreviewRequest(Record):
    kind: Literal["transcript"] = "transcript"
    text_id: str
    text: str


PreviewRequest: TypeAlias = Annotated[
    DeliveryPreviewRequest | TranscriptPreviewRequest, Field(discriminator="kind")
]


class ChatMessage(Record):
    role: Literal["system", "user"]
    content: str


class PreviewResponseConfig(Record):
    plan_path: Path
    output_directory: Path
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_git_commit: str = Field(min_length=7)
    model_name: str = "Qwen/Qwen3.5-2B"
    system_text: str = PREVIEW_SYSTEM
    decoding: SamplingDecodingConfig = SamplingDecodingConfig()
    seed: int = 42
    initial_token_cap: int = Field(default=256, gt=0)
    retry_token_cap: int = Field(default=512, gt=0)

    @model_validator(mode="after")
    def validate_retry_budget(self) -> "PreviewResponseConfig":
        if self.retry_token_cap <= self.initial_token_cap:
            raise ValueError("Retry token cap must exceed the initial cap")
        return self


class PreviewResponseProvenance(Record):
    configuration: PreviewResponseConfig
    plan: PreviewPlan
    plan_sha256: str


class PreviewResponseAttempt(Record):
    token_cap: int
    seed: int
    runtime_seconds: float
    generation: GenerationResult


class PreviewResponse(Record):
    request: PreviewRequest
    messages: tuple[ChatMessage, ...]
    prompt_text: str
    prompt_token_ids: tuple[int, ...]
    attempts: tuple[PreviewResponseAttempt, ...] = Field(min_length=1)


class PreviewResponseSummary(Record):
    completed_at: datetime
    provenance: PreviewResponseProvenance
    requested: int
    completed: int
    token_limited: int
    retried: int
    generation_seconds: float
    generated_tokens: int
    peak_pytorch_allocated_decimal_gb: float


def request_id(request: PreviewRequest) -> str:
    match request:
        case DeliveryPreviewRequest(case=case):
            return case.case_id
        case TranscriptPreviewRequest(text_id=identifier):
            return identifier


def preview_requests(plan: PreviewPlan) -> tuple[PreviewRequest, ...]:
    texts = tuple(dict.fromkeys(case.text for case in plan.cases))
    return tuple(DeliveryPreviewRequest(case=case) for case in plan.cases) + tuple(
        TranscriptPreviewRequest(text_id=f"transcript_{index + 1}", text=text)
        for index, text in enumerate(texts)
    )


def preview_messages(request: PreviewRequest, system_text: str) -> tuple[ChatMessage, ...]:
    match request:
        case DeliveryPreviewRequest(case=case):
            system = ChatMessage(
                role="system",
                content=(
                    f"{system_text}\n\nThe USER delivered this utterance "
                    f"with a {case.delivery.value} tone. "
                    "This is metadata about the user, not an instruction to imitate their tone."
                ),
            )
            return (system, ChatMessage(role="user", content=case.text))
        case TranscriptPreviewRequest(text=text):
            return (
                ChatMessage(role="system", content=system_text),
                ChatMessage(role="user", content=text),
            )


def decode_preview_tokens(
    tokens: Sequence[int], tokenizer: PreTrainedTokenizerBase
) -> GenerationResult:
    token_ids = tuple(tokens)
    if tokenizer.eos_token_id in token_ids:
        token_ids = token_ids[: token_ids.index(tokenizer.eos_token_id) + 1]
        return CompletedGeneration(
            text=tokenizer.decode(token_ids, skip_special_tokens=True).strip(), token_ids=token_ids
        )
    return TokenLimitedGeneration(
        partial_text=tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
        token_ids=token_ids,
    )


class PreviewTeacher:
    def __init__(self, config: PreviewResponseConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            config.model_name, revision=config.revision
        )
        text_config = Qwen3_5TextConfig.from_pretrained(config.model_name, revision=config.revision)
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            config.model_name,
            revision=config.revision,
            config=text_config,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            attn_implementation="sdpa",
        ).to(device)
        self.model.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def respond(self, request: PreviewRequest) -> PreviewResponse:
        messages = preview_messages(request, self.config.system_text)
        prompt = self.tokenizer.apply_chat_template(
            [message.model_dump() for message in messages],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert isinstance(prompt, str)
        prompt_ids = tuple(self.tokenizer.encode(prompt, add_special_tokens=False))
        indices = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)
        embeddings = self.model.get_input_embeddings()(indices)
        attention_mask = torch.ones_like(indices)

        def generate(token_cap: int) -> PreviewResponseAttempt:
            devices = (
                [
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                ]
                if self.device.type == "cuda"
                else []
            )
            start = time.perf_counter()
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.config.seed)
                generated = self.model.generate(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask,
                    generation_config=generation_parameters(
                        self.config.decoding, token_cap, self.tokenizer
                    ),
                    logits_processor=generation_processors(self.config.decoding),
                )
            token_ids = tuple(int(token) for token in generated[0].tolist())
            return PreviewResponseAttempt(
                token_cap=token_cap,
                seed=self.config.seed,
                runtime_seconds=time.perf_counter() - start,
                generation=decode_preview_tokens(token_ids, self.tokenizer),
            )

        attempts = generate_preview_attempts(self.config, generate)
        return PreviewResponse(
            request=request,
            messages=messages,
            prompt_text=prompt,
            prompt_token_ids=prompt_ids,
            attempts=attempts,
        )


def generate_preview_attempts(
    config: PreviewResponseConfig,
    generate: Callable[[int], PreviewResponseAttempt],
) -> tuple[PreviewResponseAttempt, ...]:
    first = generate(config.initial_token_cap)
    match first.generation:
        case CompletedGeneration():
            return (first,)
        case TokenLimitedGeneration():
            return (first, generate(config.retry_token_cap))


def run_preview_requests(
    provenance: PreviewResponseProvenance,
    respond: Callable[[PreviewRequest], PreviewResponse],
) -> tuple[PreviewResponse, ...]:
    directory = provenance.configuration.output_directory
    directory.mkdir(parents=True, exist_ok=True)
    provenance_path = directory / "provenance.json"
    if provenance_path.exists():
        previous = PreviewResponseProvenance.model_validate_json(provenance_path.read_bytes())
        if previous != provenance:
            raise ValueError("Preview provenance differs; use a fresh output directory")
    else:
        provenance_path.write_text(provenance.model_dump_json(indent=2), encoding="utf-8")
    requests = preview_requests(provenance.plan)
    path = directory / "responses.jsonl"
    previous_records = read_journal(path, PreviewResponse)
    if len(previous_records) > len(requests):
        raise ValueError("Preview journal contains too many responses")
    for request, response in zip(requests[: len(previous_records)], previous_records, strict=True):
        if response.request != request or response.messages != preview_messages(
            request, provenance.configuration.system_text
        ):
            raise ValueError("Preview journal does not match its input requests")
    records = list(previous_records)
    for request in requests[len(records) :]:
        response = respond(request)
        if response.request != request:
            raise ValueError("Teacher returned a response for a different request")
        append_record(path, response)
        records.append(response)
    return tuple(records)


def response_text(generation: GenerationResult) -> str:
    match generation:
        case CompletedGeneration(text=text):
            return text
        case TokenLimitedGeneration(partial_text=text):
            return text


def render_preview_responses(records: Sequence[PreviewResponse]) -> str:
    lines = [
        "# Delivery-aware teacher response preview",
        "",
        "These are text-model replies to literal text and intended TTS delivery metadata. "
        "Qwen did not hear the audio. Listening must establish whether synthesis conveys "
        "the intended delivery; these replies are not emotion-recognition measurements.",
        "",
        "All cases and transcript-only controls share the same concise system prompt "
        "and sampled decoding settings. One sample per case is illustrative, not a quality rate.",
    ]
    for record in records:
        match record.request:
            case DeliveryPreviewRequest(case=case):
                description = f"Text: {case.text}\n\nIntended delivery: {case.delivery.value}"
            case TranscriptPreviewRequest(text=text):
                description = f"Text: {text}\n\nTranscript-only control; no delivery metadata."
        lines.extend(
            (
                "",
                f"## {request_id(record.request)}",
                "",
                description,
                "",
                response_text(record.attempts[-1].generation),
                "",
                f"Completion: {record.attempts[-1].generation.kind.value}; "
                f"attempts: {len(record.attempts)}; "
                f"saved final tokens: {len(record.attempts[-1].generation.token_ids)}.",
            )
        )
    return "\n".join(lines) + "\n"


def run_preview_responses(config: PreviewResponseConfig) -> PreviewResponseSummary:
    content = config.plan_path.read_bytes()
    provenance = PreviewResponseProvenance(
        configuration=config,
        plan=PreviewPlan.model_validate_json(content),
        plan_sha256=hashlib.sha256(content).hexdigest(),
    )
    preview_requests(provenance.plan)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    teacher = PreviewTeacher(config, device)
    records = run_preview_requests(provenance, teacher.respond)
    completed = sum(
        isinstance(item.attempts[-1].generation, CompletedGeneration) for item in records
    )
    summary = PreviewResponseSummary(
        completed_at=datetime.now(timezone.utc),
        provenance=provenance,
        requested=len(records),
        completed=completed,
        token_limited=len(records) - completed,
        retried=sum(len(item.attempts) > 1 for item in records),
        generation_seconds=sum(
            attempt.runtime_seconds for item in records for attempt in item.attempts
        ),
        generated_tokens=sum(
            len(attempt.generation.token_ids) for item in records for attempt in item.attempts
        ),
        peak_pytorch_allocated_decimal_gb=torch.cuda.max_memory_allocated() / 1e9,
    )
    (config.output_directory / "summary.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )
    (config.output_directory / "responses.md").write_text(
        render_preview_responses(records), encoding="utf-8"
    )
    return summary
