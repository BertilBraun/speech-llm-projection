"""Offline vLLM supplier for the shared emotional generation pipeline."""

import time
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Literal

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.sampling_params import StructuredOutputsParams

from speech_projector.emotional_dataset import GeneratedDraftBatch
from speech_projector.emotional_generation import (
    EmotionalGenerationConfig,
    TextGenerationBatch,
    TextGenerationOutcome,
    TextGenerationRequest,
    generation_from_finish_reason,
)
from speech_projector.journal import append_record
from speech_projector.models import Record


class VllmTerminationEvidence(Record):
    captured_at: datetime
    request_id: str
    token_cap: int
    finish_reason: Literal["stop", "length"]
    stop_reason: int | str | None
    eos_present_in_returned_ids: bool


class VllmTextGenerator:
    def __init__(self, config: EmotionalGenerationConfig) -> None:
        self.configuration = config
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            config.model_name, revision=config.revision
        )
        self.model = LLM(
            model=config.model_name,
            revision=config.revision,
            tokenizer_revision=config.revision,
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_model_len=config.max_model_tokens,
            gpu_memory_utilization=0.85,
            language_model_only=True,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
            generation_config="vllm",
            seed=config.seed,
        )

    def generate(
        self, requests: Sequence[TextGenerationRequest], token_cap: int
    ) -> TextGenerationBatch:
        if not requests:
            raise ValueError("vLLM generation requires a nonempty batch")
        started = time.perf_counter()
        prompts: list[str] = []
        native_inputs: list[TokensPrompt] = []
        parameters: list[SamplingParams] = []
        for request in requests:
            prompt = self.tokenizer.apply_chat_template(
                [message.model_dump() for message in request.messages],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            assert isinstance(prompt, str)
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(tokens) + token_cap > self.configuration.max_model_tokens:
                raise ValueError("Prompt plus output budget exceeds configured model token budget")
            prompts.append(prompt)
            native_inputs.append(TokensPrompt(prompt_token_ids=tokens))
            parameters.append(
                SamplingParams(
                    temperature=0,
                    presence_penalty=0,
                    frequency_penalty=0,
                    repetition_penalty=1,
                    max_tokens=token_cap,
                    seed=self.configuration.seed,
                    structured_outputs=StructuredOutputsParams(
                        json=GeneratedDraftBatch.model_json_schema()
                    )
                    if request.request_id.startswith("draft:")
                    else None,
                )
            )
        outputs = self.model.generate(native_inputs, sampling_params=parameters, use_tqdm=False)
        outcomes: list[TextGenerationOutcome] = []
        evidence_path = (
            self.configuration.output_directory
            / self.configuration.trace_subdirectory
            / "vllm_termination.jsonl"
        )
        for request, prompt, native, output in zip(
            requests, prompts, native_inputs, outputs, strict=True
        ):
            if not output.finished or len(output.outputs) != 1:
                raise ValueError("vLLM must return one finished completion per request")
            if output.prompt_token_ids is None:
                raise ValueError("vLLM did not return the required prompt token identifiers")
            if tuple(output.prompt_token_ids) != tuple(native["prompt_token_ids"]):
                raise ValueError("vLLM prompt token IDs differ from the recorded request")
            completion = output.outputs[0]
            tokens = tuple(completion.token_ids)
            response = generation_from_finish_reason(
                completion.text, tokens, completion.finish_reason
            )
            append_record(
                evidence_path,
                VllmTerminationEvidence(
                    captured_at=datetime.now(timezone.utc),
                    request_id=request.request_id,
                    token_cap=token_cap,
                    finish_reason=completion.finish_reason,
                    stop_reason=completion.stop_reason,
                    eos_present_in_returned_ids=self.tokenizer.eos_token_id in tokens,
                ),
            )
            outcomes.append(
                TextGenerationOutcome(
                    request=request,
                    prompt_text=prompt,
                    prompt_token_ids=tuple(native["prompt_token_ids"]),
                    generation=response,
                )
            )
        return TextGenerationBatch(
            token_cap=token_cap,
            seed=self.configuration.seed,
            runtime_seconds=time.perf_counter() - started,
            outcomes=tuple(outcomes),
        )
