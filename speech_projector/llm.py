"""Frozen Qwen with differentiable speech embeddings and target-only logits."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as functional
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Qwen3_5ForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from speech_projector.decoding import generation_parameters, generation_processors
from speech_projector.generation import (
    CompletedGeneration,
    GenerationResult,
    TokenLimitedGeneration,
)
from speech_projector.inputs import SpeechInput, TranscriptInput, UtteranceInput
from speech_projector.models import (
    ChatPromptConfig,
    Example,
    PromptConfig,
    RunConfig,
    RunPromptConfig,
    SystemPromptConfig,
)
from speech_projector.prompts import ASSISTANT_SUFFIX, USER_PREFIX, system_prefix


@dataclass(frozen=True)
class EmbeddedSequence:
    embeddings: Tensor
    attention_mask: Tensor
    labels: Tensor
    target_start: int
    target_tokens: int


@dataclass(frozen=True)
class TargetScores:
    target_token_ids: Tensor
    logits: Tensor


@dataclass(frozen=True)
class GenerationBatch:
    embeddings: Tensor
    attention_mask: Tensor


@dataclass(frozen=True)
class TrainingBatch:
    embeddings: Tensor
    attention_mask: Tensor
    sequences: tuple[EmbeddedSequence, ...]


def generation_seed(config: RunConfig, examples: Sequence[Example], max_new_tokens: int) -> int:
    digest = hashlib.sha256(f"{config.seed}:{max_new_tokens}:".encode("ascii"))
    for example in examples:
        identifier = example.example_id.encode("utf-8")
        digest.update(len(identifier).to_bytes(8, "little"))
        digest.update(identifier)
    return int.from_bytes(digest.digest()[:8], "little") % (2**63 - 1)


def example_prompt(example: Example, config: RunConfig) -> PromptConfig:
    match example.prompt:
        case RunPromptConfig():
            return config.prompt
        case ChatPromptConfig() | SystemPromptConfig():
            return example.prompt


class FrozenQwen:
    def __init__(self, config: RunConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(config.model_name)
        text_config = Qwen3_5TextConfig.from_pretrained(config.model_name)
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            config.model_name,
            config=text_config,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            attn_implementation="sdpa",
        ).to(device)
        self.model.requires_grad_(False)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.model.eval()

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _embed(self, token_ids: list[int]) -> Tensor:
        indices = torch.tensor(token_ids, device=self.device, dtype=torch.long)
        return self.model.get_input_embeddings()(indices)

    def _history_ids(self, example: Example) -> list[int]:
        remaining = self.config.max_history_tokens
        encoded_turns: list[list[int]] = []
        turns = example.history[-self.config.history_turns :] if self.config.history_turns else ()
        for turn in reversed(turns):
            start = self._encode(f"<|im_start|>{turn.role.value}\n")
            end = self._encode("<|im_end|>\n")
            body = self._encode(turn.text)
            available = remaining - len(start) - len(end)
            if available <= 0:
                break
            encoded = start + body[-available:] + end
            encoded_turns.append(encoded)
            remaining -= len(encoded)
        return [token for turn in reversed(encoded_turns) for token in turn]

    def _prompt(self, example: Example, utterance: UtteranceInput) -> Tensor:
        start = self._encode(system_prefix(example_prompt(example, self.config)))
        prefix = self._embed(start + self._history_ids(example) + self._encode(USER_PREFIX))
        match utterance:
            case TranscriptInput(text=text):
                current = self._embed(self._encode(text))
            case SpeechInput(embeddings=embeddings):
                if embeddings.ndim != 2:
                    raise ValueError("Speech embeddings must have shape (tokens, dimension)")
                if embeddings.shape[1] != self.model.config.hidden_size:
                    raise ValueError("Speech embedding dimension differs from Qwen")
                current = embeddings.to(device=self.device, dtype=prefix.dtype)
        suffix = self._embed(self._encode(ASSISTANT_SUFFIX))
        return torch.cat((prefix, current, suffix), dim=0)

    def _target_ids(self, example: Example) -> list[int]:
        response = self._encode(example.target_text)[: self.config.max_target_tokens - 1]
        return response + self._encode("<|im_end|>")

    def target_token_count(self, example: Example) -> int:
        return len(self._target_ids(example))

    def prepare(self, example: Example, utterance: UtteranceInput) -> EmbeddedSequence:
        prompt = self._prompt(example, utterance)
        target_ids = self._target_ids(example)
        embeddings = torch.cat((prompt, self._embed(target_ids)), dim=0).unsqueeze(0)
        labels = torch.full(embeddings.shape[:2], -100, device=self.device, dtype=torch.long)
        labels[0, prompt.shape[0] :] = torch.tensor(target_ids, device=self.device)
        attention_mask = torch.ones(embeddings.shape[:2], device=self.device, dtype=torch.long)
        padding = (-embeddings.shape[1]) % self.config.sequence_length_multiple
        # Bucketing avoids recompiling Triton kernels for every distinct utterance length.
        embeddings = functional.pad(embeddings, (0, 0, 0, padding))
        attention_mask = functional.pad(attention_mask, (0, padding))
        labels = functional.pad(labels, (0, padding), value=-100)
        return EmbeddedSequence(
            embeddings=embeddings,
            attention_mask=attention_mask,
            labels=labels,
            target_start=prompt.shape[0],
            target_tokens=len(target_ids),
        )

    def score_target(self, example: Example, utterance: UtteranceInput) -> TargetScores:
        sequence = self.prepare(example, utterance)
        predictor_positions = torch.arange(
            sequence.target_start - 1,
            sequence.target_start + sequence.target_tokens - 1,
            device=self.device,
        )
        output = self.model(
            inputs_embeds=sequence.embeddings,
            attention_mask=sequence.attention_mask,
            use_cache=False,
            logits_to_keep=predictor_positions,
        )
        return TargetScores(
            logits=output.logits[0],
            target_token_ids=sequence.labels[
                0, sequence.target_start : sequence.target_start + sequence.target_tokens
            ],
        )

    def loss(self, example: Example, utterance: UtteranceInput) -> Tensor:
        scores = self.score_target(example, utterance)
        return functional.cross_entropy(scores.logits.float(), scores.target_token_ids)

    def prepare_training_batch(
        self, examples: Sequence[Example], utterances: Sequence[UtteranceInput]
    ) -> TrainingBatch:
        if not examples or len(examples) != len(utterances):
            raise ValueError("Training needs equal nonempty input batches")
        sequences = tuple(
            self.prepare(example, utterance)
            for example, utterance in zip(examples, utterances, strict=True)
        )
        longest = max(sequence.embeddings.shape[1] for sequence in sequences)
        embeddings = torch.cat(
            tuple(
                functional.pad(
                    sequence.embeddings, (0, 0, 0, longest - sequence.embeddings.shape[1])
                )
                for sequence in sequences
            )
        )
        attention_mask = torch.cat(
            tuple(
                functional.pad(
                    sequence.attention_mask, (0, longest - sequence.attention_mask.shape[1])
                )
                for sequence in sequences
            )
        )
        return TrainingBatch(
            embeddings=embeddings, attention_mask=attention_mask, sequences=sequences
        )

    def _target_cross_entropy_sum(self, hidden_states: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.model.lm_head(hidden_states)
        return functional.cross_entropy(logits.float(), target_ids, reduction="sum")

    def loss_batch(
        self, examples: Sequence[Example], utterances: Sequence[UtteranceInput]
    ) -> Tensor:
        """Per-example CE mean, avoiding a full-sequence vocabulary projection."""
        batch = self.prepare_training_batch(examples, utterances)
        output = self.model.model(
            inputs_embeds=batch.embeddings,
            attention_mask=batch.attention_mask,
            use_cache=False,
        )
        example_losses: list[Tensor] = []
        for row, sequence in enumerate(batch.sequences):
            hidden_states = output.last_hidden_state[
                row, sequence.target_start - 1 : sequence.target_start + sequence.target_tokens - 1
            ]
            target_ids = sequence.labels[
                0, sequence.target_start : sequence.target_start + sequence.target_tokens
            ]
            chunks = tuple(
                checkpoint(
                    self._target_cross_entropy_sum,
                    hidden_states[start : start + 256],
                    target_ids[start : start + 256],
                    use_reentrant=False,
                )
                for start in range(0, sequence.target_tokens, 256)
            )
            example_losses.append(torch.stack(chunks).sum() / sequence.target_tokens)
        return torch.stack(example_losses).mean()

    def prepare_generation_batch(
        self,
        examples: Sequence[Example],
        utterances: Sequence[UtteranceInput],
    ) -> GenerationBatch:
        if not examples or len(examples) != len(utterances):
            raise ValueError("Generation needs equal nonempty input batches")
        prompts = tuple(
            self._prompt(example, utterance)
            for example, utterance in zip(examples, utterances, strict=True)
        )
        longest = max(prompt.shape[0] for prompt in prompts)
        longest += (-longest) % self.config.sequence_length_multiple
        embeddings = torch.stack(
            tuple(
                functional.pad(prompt, (0, 0, longest - prompt.shape[0], 0)) for prompt in prompts
            )
        )
        attention_mask = torch.stack(
            tuple(
                functional.pad(
                    torch.ones(prompt.shape[0], device=self.device, dtype=torch.long),
                    (longest - prompt.shape[0], 0),
                )
                for prompt in prompts
            )
        )
        return GenerationBatch(embeddings=embeddings, attention_mask=attention_mask)

    @torch.no_grad()
    def _generate(
        self,
        examples: Sequence[Example],
        embeddings: Tensor,
        attention_mask: Tensor,
        max_new_tokens: int,
    ) -> Tensor:
        devices = (
            [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            if self.device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(generation_seed(self.config, examples, max_new_tokens))
            generated: Tensor = self.model.generate(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                generation_config=generation_parameters(
                    self.config.decoding, max_new_tokens, self.tokenizer
                ),
                logits_processor=generation_processors(self.config.decoding),
            )
        return generated

    @torch.no_grad()
    def generate_batch(
        self,
        examples: Sequence[Example],
        utterances: Sequence[UtteranceInput],
        max_new_tokens: int,
    ) -> tuple[GenerationResult, ...]:
        if max_new_tokens <= 0:
            raise ValueError("Generation token cap must be positive")
        self.model.eval()
        batch = self.prepare_generation_batch(examples, utterances)
        generated = self._generate(examples, batch.embeddings, batch.attention_mask, max_new_tokens)
        results: list[GenerationResult] = []
        for row in generated:
            token_ids = tuple(int(token) for token in row.tolist())
            if self.tokenizer.eos_token_id in token_ids:
                end = token_ids.index(self.tokenizer.eos_token_id) + 1
                completed = token_ids[:end]
                results.append(
                    CompletedGeneration(
                        text=self.tokenizer.decode(completed, skip_special_tokens=True).strip(),
                        token_ids=completed,
                    )
                )
            else:
                results.append(
                    TokenLimitedGeneration(
                        partial_text=self.tokenizer.decode(
                            token_ids, skip_special_tokens=True
                        ).strip(),
                        token_ids=token_ids,
                    )
                )
        return tuple(results)

    @torch.no_grad()
    def generate(self, example: Example, utterance: UtteranceInput) -> str:
        self.model.eval()
        prompt = self._prompt(example, utterance).unsqueeze(0)
        generated = self._generate(
            (example,),
            prompt,
            torch.ones(prompt.shape[:2], device=self.device, dtype=torch.long),
            self.config.max_new_tokens,
        )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
