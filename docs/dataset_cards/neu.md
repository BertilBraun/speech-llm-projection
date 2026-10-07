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

# Neu paired emotional speech

**Local release preparation; not yet uploaded.** Add the final source-repository
URL and select the publisher's collection license before public publication.
See [collection and output terms](LICENSE.md) and the accompanying
[NeuTTS Open License](NeuTTS-Open-License-1.0.txt).

This collection contains **5,000 distinct English utterances**, each spoken with
two intended deliveries, and a corresponding Qwen teacher response per delivery.
The spoken words are identical within a pair. Labels describe synthesis controls,
not human-verified emotion. There are **10,000 audio/response examples**.

## Construction

Qwen3.5-4B authored short, concrete utterances across ten everyday domains.
Each text supports one pair: happy/sad, happy/fearful, happy/angry or sad/angry.
Each pair type has 1,250 utterances. Literal texts do not explicitly name the
assigned emotion; structural constraints and exact duplicate rules were applied,
without rejecting weak perceived deliveries or teacher contrasts.

Qwen3.5-2B generated concise responses to each utterance plus intended-tone
metadata. That metadata is not spoken. NeuTTS-2E generated audio using its Paul
voice and explicit emotion argument; NeuCodec decoded the waveform. Existing
watermarks and original float32 mono 24 kHz WAV bytes are preserved unchanged.
Generation configs retain exact prompts, model revisions and parameters.

## Splits and format

| Split | Utterances | Examples |
|---|---:|---:|
| Train | 4,550 | 9,100 |
| Validation | 230 | 460 |
| Test | 220 | 440 |

Whole design families determine splits. Group on `utterance_id` for the two
deliveries, and use `family_id` for clustered evaluation. The portable Parquet
files embed original WAV bytes, with SHA256 in `audio_sha256`. `teacher_response`
is synthetic tone-conditioned supervision, not a transcript or factual gold.
`examples.jsonl` is the lightweight metadata index; the export receipt records
coverage, counts and shard hashes.

Audio totals **14.8 hours**; mean duration **5.32 seconds**. Text averages
**13.7 words**, teacher responses **23.1 words**. The 93 exact-identical response
pairs (**1.86%**) remain in the dataset. No new editorial filtering occurred.

## Intended use and limitations

Useful for paired speech-conditioning studies, matched/swapped target tests and
synthetic response adaptation. It is not a benchmark of natural emotion
recognition or a validated production tone classifier. It uses one synthetic
voice, four intended deliveries and constrained everyday contexts. Some audio
contrasts are weak, and teacher responses may be generic or invent details.
Different responses alone do not prove appropriate emotional understanding.

## Source/output terms

NeuTTS-2E uses a custom license whose commercial-use condition explicitly covers
generated outputs. Its annual-revenue threshold is US$5 million. The full pinned
license is included; do not relabel the audio as unrestricted Apache-licensed
data. NeuCodec and Qwen model cards list Apache 2 for their models. The
publisher's additional collection license has not yet been selected. This
collection does not contain DeepDialogue-derived examples or model weights.

Sources: [NeuTTS-2E](https://huggingface.co/neuphonic/neutts-2e),
[NeuCodec](https://huggingface.co/neuphonic/neucodec),
[Qwen teacher](https://huggingface.co/Qwen/Qwen3.5-2B).
