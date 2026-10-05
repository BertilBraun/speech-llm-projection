"""Profile real cached examples and verify the frozen-model gradient boundary."""

import json
import time
from pathlib import Path

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5

from speech_projector.configuration import feasibility_run
from speech_projector.data import load_examples
from speech_projector.llm import FrozenQwen
from speech_projector.models import Record, Split
from speech_projector.projectors import Projector
from speech_projector.training import gradient_sanity, load_features


class StepProfile(Record):
    example_id: str
    duration: float
    speech_tokens: int
    target_tokens: int
    seconds: float
    loss: float
    peak_vram_gb: float


def main() -> None:
    directory = Path("results/model_profile")
    directory.mkdir(parents=True, exist_ok=True)
    config = feasibility_run()
    examples = load_examples(Path("data/examples.jsonl"), Split.TRAIN)
    wrapper = FrozenQwen(config, torch.device("cuda"))
    projector = Projector(config.projector).to(wrapper.device)
    print(f"all_fast_kernels={modeling_qwen3_5.is_fast_path_available}", flush=True)
    print(f"delta_kernel={modeling_qwen3_5.chunk_gated_delta_rule}", flush=True)
    check = gradient_sanity(wrapper, projector, examples[0])
    (directory / "gradient_check.json").write_text(
        check.model_dump_json(indent=2), encoding="utf-8"
    )
    print(check.model_dump_json(), flush=True)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=config.learning_rate)
    profiles: list[StepProfile] = []
    for example in examples[:5]:
        wrapper.model.train(config.gradient_checkpointing)
        projector.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.monotonic()
        speech = projector(load_features(example, wrapper.device))
        loss = wrapper.loss(example, speech_embeddings=speech)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        profile = StepProfile(
            example_id=example.example_id,
            duration=example.duration,
            speech_tokens=speech.shape[0],
            target_tokens=wrapper.target_token_count(example),
            seconds=time.monotonic() - start,
            loss=loss.item(),
            peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
        )
        print(profile.model_dump_json(), flush=True)
        profiles.append(profile)
    (directory / "steps.json").write_text(
        json.dumps([profile.model_dump() for profile in profiles], indent=2), encoding="utf-8"
    )
    with torch.no_grad():
        speech = projector(load_features(examples[0], wrapper.device))
        generated = wrapper.generate(examples[0], speech_embeddings=speech)
        textual = wrapper.generate(examples[0], transcript=examples[0].user_text)
    content = (
        f"User: {examples[0].user_text}\nGold: {examples[0].target_text}"
        f"\nSpeech: {generated}\nText: {textual}"
    )
    (directory / "generations.txt").write_text(content, encoding="utf-8")
    print(content, flush=True)


if __name__ == "__main__":
    main()
