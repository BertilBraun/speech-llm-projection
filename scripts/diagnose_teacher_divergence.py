"""Measure actual greedy branch margins before the first single/batch divergence."""

import argparse
from pathlib import Path

import torch
from transformers.generation.utils import GenerateDecoderOnlyOutput

from scripts.profile_teacher_warm import WarmTeacherProfile
from speech_projector.inputs import TranscriptInput
from speech_projector.llm import FrozenQwen, GenerationBatch
from speech_projector.models import Record
from speech_projector.teacher_configuration import teacher_compression_runs


class BranchComparison(Record):
    example_id: str
    common_prefix_tokens: int
    exact_tokens: bool
    single_token: int
    batch_token: int
    single_top_two_margin: float
    batch_top_two_margin: float
    single_margin_over_batch_choice: float
    batch_margin_over_single_choice: float
    maximum_score_difference: float
    mean_score_difference: float


class GenerationBranchReport(Record):
    comparisons: tuple[BranchComparison, ...]


def generate_scores(wrapper: FrozenQwen, batch: GenerationBatch) -> GenerateDecoderOnlyOutput:
    return wrapper.model.generate(
        inputs_embeds=batch.embeddings,
        attention_mask=batch.attention_mask,
        max_new_tokens=128,
        do_sample=False,
        use_cache=True,
        pad_token_id=wrapper.tokenizer.pad_token_id,
        eos_token_id=wrapper.tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )


@torch.no_grad()
def diagnose(profile_path: Path, output: Path) -> GenerationBranchReport:
    profile = WarmTeacherProfile.model_validate_json(profile_path.read_bytes())
    examples = profile.examples
    wrapper = FrozenQwen(teacher_compression_runs()[0], torch.device("cuda"))
    utterances = tuple(TranscriptInput(example.user_text) for example in examples)
    batched = generate_scores(wrapper, wrapper.prepare_generation_batch(examples, utterances))
    comparisons: list[BranchComparison] = []
    for index, example in enumerate(examples[:16]):
        prompt = wrapper._prompt(example, utterances[index]).unsqueeze(0)
        original = GenerationBatch(
            prompt, torch.ones(prompt.shape[:2], device=wrapper.device, dtype=torch.long)
        )
        single = generate_scores(wrapper, original)
        single_ids = tuple(int(token) for token in single.sequences[0].tolist())
        batch_ids = tuple(int(token) for token in batched.sequences[index].tolist())
        eos = (
            batch_ids.index(wrapper.tokenizer.eos_token_id) + 1
            if wrapper.tokenizer.eos_token_id in batch_ids
            else len(batch_ids)
        )
        batch_ids = batch_ids[:eos]
        common = 0
        for left, right in zip(single_ids, batch_ids, strict=False):
            if left != right:
                break
            common += 1
        exact = single_ids == batch_ids
        position = min(common, len(single.scores) - 1, len(batched.scores) - 1)
        single_score = single.scores[position][0].float()
        batch_score = batched.scores[position][index].float()
        single_top = single_score.topk(2)
        batch_top = batch_score.topk(2)
        error = (single_score - batch_score).abs()
        record = BranchComparison(
            example_id=example.example_id,
            common_prefix_tokens=common,
            exact_tokens=exact,
            single_token=single_ids[position],
            batch_token=batch_ids[position],
            single_top_two_margin=float(single_top.values[0] - single_top.values[1]),
            batch_top_two_margin=float(batch_top.values[0] - batch_top.values[1]),
            single_margin_over_batch_choice=float(
                single_score[single_ids[position]] - single_score[batch_ids[position]]
            ),
            batch_margin_over_single_choice=float(
                batch_score[batch_ids[position]] - batch_score[single_ids[position]]
            ),
            maximum_score_difference=float(error.max()),
            mean_score_difference=float(error.mean()),
        )
        comparisons.append(record)
        print(record.model_dump_json(), flush=True)
    report = GenerationBranchReport(comparisons=tuple(comparisons))
    output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    diagnose(arguments.profile, arguments.output)


if __name__ == "__main__":
    main()
