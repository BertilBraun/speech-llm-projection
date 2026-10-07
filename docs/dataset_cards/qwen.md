---
language:
- en
license: other
license_name: collection-and-source-output-terms
license_link: LICENSE.md
task_categories:
- audio-classification
- text-generation
tags:
- synthetic
- paired-speech
- emotional-delivery
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: validation
    path: data/validation-*.parquet
  - split: test
    path: data/test-*.parquet
dataset_info:
  features:
  - name: case_id
    dtype: string
  - name: utterance_id
    dtype: string
  - name: family_id
    dtype: string
  - name: split
    dtype: string
  - name: domain
    dtype: string
  - name: intent
    dtype: string
  - name: text
    dtype: string
  - name: delivery
    dtype: string
  - name: teacher_response
    dtype: string
  - name: audio_sha256
    dtype: string
  - name: sample_rate
    dtype: int64
  - name: duration_seconds
    dtype: float64
  - name: audio
    dtype: audio
---

# Qwen paired emotional speech

**Local release preparation; not yet uploaded.** Add the final source-repository
URL and select the publisher's collection license before public publication.
See [collection and source terms](LICENSE.md).

This older collection contains **5,000 distinct English utterances**, each
spoken with two requested deliveries and paired with two Qwen teacher responses.
The spoken words stay identical within each pair. There are **10,000
audio/response examples**; intended delivery labels are not human annotations.

## Construction

Qwen3.5-4B authored short everyday utterances for two plausible deliveries chosen
from neutral, happy, sad, frustrated and sarcastic. Qwen3.5-2B generated concise
responses from the text plus intended-tone metadata. Qwen CustomVoice synthesized
each delivery with written voice instructions, using its Ryan voice and the
vLLM-Omni production backend. Those instructions are not part of the spoken text.
The original mono 24 kHz WAV bytes are preserved; no audio is regenerated or
normalized by export. Actual generation configs retain prompts and model pins.

## Splits and format

| Split | Utterances | Examples |
|---|---:|---:|
| Train | 4,550 | 9,100 |
| Validation | 230 | 460 |
| Test | 220 | 440 |

Whole design families determine the splits, with both deliveries together.
Parquet embeds the original WAV bytes and records their SHA256. Group on
`utterance_id` for paired comparisons, and use `family_id` for clustered
evaluation. `teacher_response` is generated supervision, not factual gold.
The small `examples.jsonl` index and export receipt preserve coverage and hashes.
Three overlong training pairs were excluded from the later mixed projector
experiment; this source release retains them and the remaining weak contrasts.

## Intended use and limitations

Useful for synthetic same-text speech conditioning and comparisons with the Neu
collection. Some sad/sarcastic deliveries were judged weak or unnatural during
informal listening, and teacher replies can differ little in emotional content.
Only one synthetic voice is represented. These labels do not measure natural
emotion recognition; the source wording can also suggest an interpretation.
Replies can be generic or invent details. No new perceptual rejection was
applied for release. Do not treat this collection and Neu as identically
distributed: their prompts, delivery labels and synthesis backends differ.

## Source terms

The Qwen text and CustomVoice model cards list Apache 2 for those models. This
does not automatically select a license for the authored collection; that grant
remains for the publisher to document. This package contains no Neu-generated
audio, DeepDialogue-derived ordinary examples or model weights.

Sources: [Qwen teacher](https://huggingface.co/Qwen/Qwen3.5-2B),
[Qwen CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice).
