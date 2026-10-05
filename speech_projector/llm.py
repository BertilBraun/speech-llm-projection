"""Frozen Qwen with differentiable speech embeddings and target-only logits."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as functional
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Qwen3_5ForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from speech_projector.models import Example, RunConfig


@dataclass(frozen=True)
class EmbeddedSequence:
    embeddings: Tensor
    attention_mask: Tensor
    labels: Tensor
    target_start: int
    target_tokens: int


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

    def _prompt(
        self, example: Example, speech_embeddings: Tensor | None, transcript: str | None
    ) -> Tensor:
        if (speech_embeddings is None) == (transcript is None):
            raise ValueError("Provide exactly one of speech embeddings or transcript")
        start = self._encode(
            "<|im_start|>system\nYou are a helpful conversational assistant. "
            "Reply naturally to the user's utterance.<|im_end|>\n"
        )
        prefix = self._embed(
            start + self._history_ids(example) + self._encode("<|im_start|>user\n")
        )
        match speech_embeddings:
            case None:
                assert transcript is not None
                current = self._embed(self._encode(transcript))
            case _:
                if speech_embeddings.ndim != 2:
                    raise ValueError("Speech embeddings must have shape (tokens, dimension)")
                if speech_embeddings.shape[1] != self.model.config.hidden_size:
                    raise ValueError("Speech embedding dimension differs from Qwen")
                current = speech_embeddings.to(device=self.device, dtype=prefix.dtype)
        suffix = self._embed(
            self._encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        )
        return torch.cat((prefix, current, suffix), dim=0)

    def _target_ids(self, example: Example) -> list[int]:
        response = self._encode(example.target_text)[: self.config.max_target_tokens - 1]
        return response + self._encode("<|im_end|>")

    def target_token_count(self, example: Example) -> int:
        return len(self._target_ids(example))

    def prepare(
        self,
        example: Example,
        speech_embeddings: Tensor | None = None,
        transcript: str | None = None,
    ) -> EmbeddedSequence:
        prompt = self._prompt(example, speech_embeddings, transcript)
        target_ids = self._target_ids(example)
        embeddings = torch.cat((prompt, self._embed(target_ids)), dim=0).unsqueeze(0)
        labels = torch.full(embeddings.shape[:2], -100, device=self.device, dtype=torch.long)
        labels[0, prompt.shape[0] :] = torch.tensor(target_ids, device=self.device)
        return EmbeddedSequence(
            embeddings=embeddings,
            attention_mask=torch.ones(embeddings.shape[:2], device=self.device, dtype=torch.long),
            labels=labels,
            target_start=prompt.shape[0],
            target_tokens=len(target_ids),
        )

    def loss(
        self,
        example: Example,
        speech_embeddings: Tensor | None = None,
        transcript: str | None = None,
    ) -> Tensor:
        sequence = self.prepare(example, speech_embeddings, transcript)
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
        return functional.cross_entropy(
            output.logits[0].float(), sequence.labels[0, sequence.target_start :]
        )

    @torch.no_grad()
    def generate(
        self,
        example: Example,
        speech_embeddings: Tensor | None = None,
        transcript: str | None = None,
    ) -> str:
        self.model.eval()
        prompt = self._prompt(example, speech_embeddings, transcript).unsqueeze(0)
        generated = self.model.generate(
            inputs_embeds=prompt,
            attention_mask=torch.ones(prompt.shape[:2], device=self.device, dtype=torch.long),
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.model.config.eos_token_id,
        )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
