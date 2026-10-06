"""Batched frozen-model drafting and paired teacher targets with durable traces."""

import time
from collections.abc import Callable, Sequence
from enum import Enum
from pathlib import Path

import torch
from pydantic import Field, ValidationError, model_validator
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Qwen3_5ForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from speech_projector.decoding import generation_parameters, generation_processors
from speech_projector.emotional_dataset import (
    DeliveryAnnotation,
    DraftRequest,
    EmotionalDatasetConfig,
    EmotionalUtterance,
    GeneratedDraftBatch,
    build_draft_requests,
    draft_prompt,
    run_draft_construction,
)
from speech_projector.generation import (
    CompletedGeneration,
    GenerationResult,
    TokenLimitedGeneration,
)
from speech_projector.journal import append_record, read_journal
from speech_projector.models import GreedyDecodingConfig, Record
from speech_projector.preview_responses import ChatMessage, decode_preview_tokens

DRAFT_SYSTEM = "Write the requested conversational utterances. Return only the requested JSON."
TEACHER_SYSTEM = (
    "Reply naturally in one or two short sentences. "
    "The supplied tone describes the user's emotional delivery. "
    "Respond appropriately to that delivery and what they said."
)


class GenerationBackend(str, Enum):
    HUGGING_FACE = "huggingface"
    VLLM = "vllm"


class EmotionalGenerationConfig(Record):
    output_directory: Path
    trace_subdirectory: Path = Path(".")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    model_name: str = "Qwen/Qwen3.5-2B"
    backend: GenerationBackend = GenerationBackend.HUGGING_FACE
    dataset: EmotionalDatasetConfig = EmotionalDatasetConfig()
    inference_batch_size: int = Field(default=8, gt=0)
    seed: int = Field(default=42, ge=0)
    max_model_tokens: int = Field(default=8192, gt=0)
    draft_token_cap: int = Field(default=2048, gt=0)
    draft_retry_token_cap: int = Field(default=4096, gt=0)
    teacher_token_cap: int = Field(default=256, gt=0)
    teacher_retry_token_cap: int = Field(default=512, gt=0)
    syntax_attempts: int = Field(default=3, gt=0)
    draft_system: str = DRAFT_SYSTEM
    teacher_system: str = TEACHER_SYSTEM
    decoding: GreedyDecodingConfig = GreedyDecodingConfig()

    @model_validator(mode="after")
    def validate_retry_caps(self) -> "EmotionalGenerationConfig":
        if self.trace_subdirectory.is_absolute() or ".." in self.trace_subdirectory.parts:
            raise ValueError("Trace subdirectory must remain within the dataset output directory")
        if self.draft_retry_token_cap <= self.draft_token_cap:
            raise ValueError("Draft retry budget must exceed the initial budget")
        if self.teacher_retry_token_cap <= self.teacher_token_cap:
            raise ValueError("Teacher retry budget must exceed the initial budget")
        return self


class TextGenerationRequest(Record):
    request_id: str
    messages: tuple[ChatMessage, ...] = Field(min_length=1)


class TextGenerationOutcome(Record):
    request: TextGenerationRequest
    prompt_text: str
    prompt_token_ids: tuple[int, ...]
    generation: GenerationResult


class TextGenerationBatch(Record):
    token_cap: int
    seed: int
    runtime_seconds: float
    outcomes: tuple[TextGenerationOutcome, ...]


class EmotionalGenerationProvenance(Record):
    configuration: EmotionalGenerationConfig


class DraftSyntaxFailure(Record):
    request: DraftRequest
    attempt: int
    response: CompletedGeneration
    validation_error: str


class EmotionalTeacherRequest(Record):
    utterance: EmotionalUtterance
    annotation: DeliveryAnnotation


class EmotionalTeacherTarget(Record):
    request: EmotionalTeacherRequest
    response: CompletedGeneration
    capped_attempts: tuple[TokenLimitedGeneration, ...]


class EmotionalGenerationSummary(Record):
    configuration: EmotionalGenerationConfig
    utterances: int
    completed_targets: int
    generation_batches: int
    generation_seconds: float
    generated_tokens: int


GenerateBatch = Callable[[Sequence[TextGenerationRequest], int], TextGenerationBatch]


def generation_from_finish_reason(
    text: str, token_ids: Sequence[int], finish_reason: str | None
) -> GenerationResult:
    match finish_reason:
        case "stop":
            return CompletedGeneration(text=text.strip(), token_ids=tuple(token_ids))
        case "length":
            return TokenLimitedGeneration(partial_text=text.strip(), token_ids=tuple(token_ids))
        case reason:
            raise ValueError(f"Unexpected provider termination: {reason}")


class FrozenTextGenerator:
    def __init__(self, config: EmotionalGenerationConfig, device: torch.device) -> None:
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
    def generate(
        self, requests: Sequence[TextGenerationRequest], token_cap: int
    ) -> TextGenerationBatch:
        if not requests:
            raise ValueError("Text generation requires a nonempty batch")
        started = time.perf_counter()
        prompts: list[str] = []
        token_ids: list[tuple[int, ...]] = []
        for request in requests:
            prompt = self.tokenizer.apply_chat_template(
                [message.model_dump() for message in request.messages],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            assert isinstance(prompt, str)
            prompts.append(prompt)
            token_ids.append(tuple(self.tokenizer.encode(prompt, add_special_tokens=False)))
        longest = max(len(tokens) for tokens in token_ids)
        if longest + token_cap > self.config.max_model_tokens:
            raise ValueError("Prompt plus output budget exceeds configured model token budget")
        padding_id = self.tokenizer.pad_token_id
        if padding_id is None:
            raise ValueError("Batch generation requires a tokenizer padding token")
        indices = torch.tensor(
            [[padding_id] * (longest - len(tokens)) + list(tokens) for tokens in token_ids],
            device=self.device,
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [[0] * (longest - len(tokens)) + [1] * len(tokens) for tokens in token_ids],
            device=self.device,
            dtype=torch.long,
        )
        embeddings = self.model.get_input_embeddings()(indices)
        devices = (
            [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            if self.device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.config.seed)
            generated: torch.Tensor = self.model.generate(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                generation_config=generation_parameters(
                    self.config.decoding, token_cap, self.tokenizer
                ),
                logits_processor=generation_processors(self.config.decoding),
            )
        outcomes = tuple(
            TextGenerationOutcome(
                request=request,
                prompt_text=prompt,
                prompt_token_ids=tokens,
                generation=decode_preview_tokens(
                    tuple(int(token) for token in row.tolist()), self.tokenizer
                ),
            )
            for request, prompt, tokens, row in zip(
                requests, prompts, token_ids, generated, strict=True
            )
        )
        return TextGenerationBatch(
            token_cap=token_cap,
            seed=self.config.seed,
            runtime_seconds=time.perf_counter() - started,
            outcomes=outcomes,
        )


class CachedGeneration:
    def __init__(self, config: EmotionalGenerationConfig, generate: GenerateBatch) -> None:
        self.configuration = config
        self.generate_batch = generate
        directory = config.output_directory / config.trace_subdirectory
        directory.mkdir(parents=True, exist_ok=True)
        provenance = EmotionalGenerationProvenance(configuration=config)
        path = directory / "generation_provenance.json"
        if path.exists():
            if EmotionalGenerationProvenance.model_validate_json(path.read_bytes()) != provenance:
                raise ValueError("Generation resume configuration/source/model provenance differs")
        else:
            path.write_text(provenance.model_dump_json(indent=2), encoding="utf-8")
        self.path = directory / "generation_batches.jsonl"
        self.outcomes: dict[tuple[str, int], TextGenerationOutcome] = {}
        for batch in read_journal(self.path, TextGenerationBatch):
            for outcome in batch.outcomes:
                key = (outcome.request.request_id, batch.token_cap)
                if key in self.outcomes:
                    raise ValueError("Generation journal repeats a request and token budget")
                self.outcomes[key] = outcome

    def is_cached(self, request: TextGenerationRequest, token_cap: int) -> bool:
        previous = self.outcomes.get((request.request_id, token_cap))
        if previous is not None and previous.request != request:
            raise ValueError("Cached prompt differs from the current request")
        return previous is not None

    def generate(
        self, requests: Sequence[TextGenerationRequest], token_cap: int
    ) -> tuple[TextGenerationOutcome, ...]:
        if len({request.request_id for request in requests}) != len(requests):
            raise ValueError("Generation batch repeats request identifiers")
        pending: list[TextGenerationRequest] = []
        for request in requests:
            previous = self.outcomes.get((request.request_id, token_cap))
            if previous is None:
                pending.append(request)
            elif previous.request != request:
                raise ValueError("Cached prompt differs from the current request")
        if pending:
            batch = self.generate_batch(pending, token_cap)
            if (
                batch.token_cap != token_cap
                or batch.seed != self.configuration.seed
                or tuple(item.request for item in batch.outcomes) != tuple(pending)
            ):
                raise ValueError("Backend outputs do not match the requested batch/order/budget")
            append_record(self.path, batch)
            for outcome in batch.outcomes:
                self.outcomes[(outcome.request.request_id, token_cap)] = outcome
        return tuple(self.outcomes[(request.request_id, token_cap)] for request in requests)


def draft_generation_request(
    request: DraftRequest, config: EmotionalGenerationConfig
) -> TextGenerationRequest:
    return TextGenerationRequest(
        request_id=f"draft:{request.batch_id}",
        messages=(
            ChatMessage(role="system", content=config.draft_system),
            ChatMessage(role="user", content=draft_prompt(request)),
        ),
    )


def teacher_generation_request(
    request: EmotionalTeacherRequest, config: EmotionalGenerationConfig
) -> TextGenerationRequest:
    metadata = (
        f"The USER delivered this utterance with a {request.annotation.delivery.value} tone.\n"
        "This is metadata about the user, not an instruction to imitate their tone."
    )
    return TextGenerationRequest(
        request_id=f"teacher:{request.utterance.base_id}:{request.annotation.delivery.value}",
        messages=(
            ChatMessage(role="system", content=config.teacher_system + "\n\n" + metadata),
            ChatMessage(role="user", content=request.utterance.text),
        ),
    )


def completed_outcomes(
    requests: Sequence[TextGenerationRequest],
    cached: CachedGeneration,
    initial_cap: int,
    retry_cap: int,
) -> tuple[tuple[CompletedGeneration, tuple[TokenLimitedGeneration, ...]], ...]:
    initial = cached.generate(requests, initial_cap)
    retry_requests = tuple(
        item.request for item in initial if isinstance(item.generation, TokenLimitedGeneration)
    )
    retries = cached.generate(retry_requests, retry_cap) if retry_requests else ()
    retry_by_id = {item.request.request_id: item.generation for item in retries}
    outputs: list[tuple[CompletedGeneration, tuple[TokenLimitedGeneration, ...]]] = []
    for item in initial:
        match item.generation:
            case CompletedGeneration() as response:
                outputs.append((response, ()))
            case TokenLimitedGeneration() as capped:
                match retry_by_id[item.request.request_id]:
                    case CompletedGeneration() as response:
                        outputs.append((response, (capped,)))
                    case TokenLimitedGeneration():
                        raise ValueError(
                            f"Generation remains token-limited: {item.request.request_id}; "
                            "both attempts saved"
                        )
    return tuple(outputs)


def generate_drafts(
    config: EmotionalGenerationConfig, cached: CachedGeneration
) -> tuple[EmotionalUtterance, ...]:
    requests = build_draft_requests(config.dataset)
    saved = read_journal(config.output_directory / "utterances.jsonl", EmotionalUtterance)
    completed_ids = {item.base_id for item in saved}
    pending = tuple(
        request
        for request in requests
        if any(item.base_id not in completed_ids for item in request.assignments)
    )
    pending_indices = {request.batch_id: index for index, request in enumerate(pending)}

    def generate(request: DraftRequest) -> GeneratedDraftBatch:
        original = draft_generation_request(request, config)
        if (
            not cached.is_cached(original, config.draft_token_cap)
            and request.batch_id in pending_indices
        ):
            offset = pending_indices[request.batch_id]
            batch = tuple(
                draft_generation_request(item, config)
                for item in pending[offset : offset + config.inference_batch_size]
            )
            completed_outcomes(batch, cached, config.draft_token_cap, config.draft_retry_token_cap)
        for attempt in range(config.syntax_attempts):
            response, _ = completed_outcomes(
                (original,), cached, config.draft_token_cap, config.draft_retry_token_cap
            )[0]
            try:
                parsed = GeneratedDraftBatch.model_validate_json(response.text)
                print(f"Draft batch ready for acceptance: {request.batch_id}", flush=True)
                return parsed
            except ValidationError as error:
                append_record(
                    config.output_directory / "draft_syntax_failures.jsonl",
                    DraftSyntaxFailure(
                        request=request,
                        attempt=attempt + 1,
                        response=response,
                        validation_error=str(error),
                    ),
                )
                original = TextGenerationRequest(
                    request_id=f"draft:{request.batch_id}:syntax:{attempt + 1}",
                    messages=(
                        ChatMessage(role="system", content=config.draft_system),
                        ChatMessage(
                            role="user",
                            content=draft_prompt(request)
                            + "\nReturn valid JSON only. Previous schema error:\n"
                            + str(error)[:1000],
                        ),
                    ),
                )
        raise ValueError(f"Draft JSON remained invalid for {request.batch_id}; outputs saved")

    return run_draft_construction(config.output_directory, config.dataset, generate)


def generate_teacher_targets(
    config: EmotionalGenerationConfig, cached: CachedGeneration
) -> tuple[EmotionalTeacherTarget, ...]:
    utterances = read_journal(config.output_directory / "utterances.jsonl", EmotionalUtterance)
    if len(utterances) != config.dataset.utterance_count:
        raise ValueError("Teacher generation requires the complete configured utterance manifest")
    requests = tuple(
        EmotionalTeacherRequest(utterance=item, annotation=annotation)
        for item in utterances
        for annotation in item.deliveries
    )
    path = config.output_directory / "teacher_targets.jsonl"
    previous = read_journal(path, EmotionalTeacherTarget)
    expected = {teacher_generation_request(item, config).request_id: item for item in requests}
    completed_ids: set[str] = set()
    for target in previous:
        identifier = teacher_generation_request(target.request, config).request_id
        if identifier in completed_ids or expected.get(identifier) != target.request:
            raise ValueError("Teacher target journal differs from the exact paired requests")
        completed_ids.add(identifier)
    pending = tuple(
        item
        for item in requests
        if teacher_generation_request(item, config).request_id not in completed_ids
    )
    targets = list(previous)
    for offset in range(0, len(pending), config.inference_batch_size):
        batch = pending[offset : offset + config.inference_batch_size]
        outputs = completed_outcomes(
            tuple(teacher_generation_request(item, config) for item in batch),
            cached,
            config.teacher_token_cap,
            config.teacher_retry_token_cap,
        )
        for request, (response, capped) in zip(batch, outputs, strict=True):
            if not response.text:
                raise ValueError("Teacher produced an empty completed response; trace saved")
            target = EmotionalTeacherTarget(
                request=request, response=response, capped_attempts=capped
            )
            append_record(path, target)
            targets.append(target)
        print(f"Completed paired targets: {len(targets)}/{len(requests)}", flush=True)
    return tuple(targets)


def summarize_generation(config: EmotionalGenerationConfig) -> EmotionalGenerationSummary:
    batches = read_journal(
        config.output_directory / config.trace_subdirectory / "generation_batches.jsonl",
        TextGenerationBatch,
    )
    summary = EmotionalGenerationSummary(
        configuration=config,
        utterances=len(
            read_journal(config.output_directory / "utterances.jsonl", EmotionalUtterance)
        ),
        completed_targets=len(
            read_journal(config.output_directory / "teacher_targets.jsonl", EmotionalTeacherTarget)
        ),
        generation_batches=len(batches),
        generation_seconds=sum(item.runtime_seconds for item in batches),
        generated_tokens=sum(
            len(outcome.generation.token_ids) for batch in batches for outcome in batch.outcomes
        ),
    )
    (config.output_directory / config.trace_subdirectory / "generation_summary.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )
    return summary
