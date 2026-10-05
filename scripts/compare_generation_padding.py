"""Verify greedy Qwen generation is invariant to masked left padding."""

import time
from pathlib import Path

import torch
from torch.nn import functional as functional

from speech_projector.configuration import feasibility_run
from speech_projector.data import load_examples
from speech_projector.inputs import TranscriptInput
from speech_projector.llm import FrozenQwen
from speech_projector.models import Split


def main() -> None:
    config = feasibility_run()
    wrapper = FrozenQwen(config, torch.device("cuda"))
    examples = load_examples(Path("data/examples.jsonl"), Split.VALIDATION)[:2]
    with torch.no_grad():
        for example in examples:
            prompt = wrapper._prompt(example, TranscriptInput(example.user_text)).unsqueeze(0)
            mask = torch.ones(prompt.shape[:2], device=wrapper.device, dtype=torch.long)
            padding = (-prompt.shape[1]) % config.sequence_length_multiple
            padded = functional.pad(prompt, (0, 0, padding, 0))
            padded_mask = functional.pad(mask, (padding, 0))
            outputs: list[torch.Tensor] = []
            for embeddings, attention_mask in ((prompt, mask), (padded, padded_mask)):
                torch.cuda.synchronize()
                start = time.monotonic()
                generated = wrapper.model.generate(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask,
                    max_new_tokens=64,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=wrapper.tokenizer.pad_token_id,
                    eos_token_id=wrapper.tokenizer.eos_token_id,
                )
                torch.cuda.synchronize()
                print(
                    f"length={embeddings.shape[1]} seconds={time.monotonic() - start:.3f}",
                    flush=True,
                )
                outputs.append(generated)
            assert torch.equal(outputs[0], outputs[1]), "Masked left padding changed greedy tokens"
            print(wrapper.tokenizer.decode(outputs[1][0], skip_special_tokens=True), flush=True)


if __name__ == "__main__":
    main()
