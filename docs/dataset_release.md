# Paired emotional speech: generation and release

**Publishing the generation pipeline and the paired corpora is worthwhile.** The
useful research unit is the same literal utterance spoken in two deliveries,
with two tone-conditioned teacher responses. It supports paired assignment
tests, acoustic ablations and response conditioning without changing the words.
The labels describe requested synthesis controls; they are not human-verified
emotion annotations. The source audio and existing supervision remain unchanged.

## Collections

| Collection | Utterances | Audio/teacher examples | Voice | Requested deliveries |
|---|---:|---:|---|---|
| Older Qwen | 5,000 | 10,000 | Ryan | Neutral, happy, sad, frustrated, sarcastic; two per utterance |
| Neu | 5,000 | 10,000 | Paul | Happy/sad, happy/fearful, happy/angry, sad/angry |

Each source collection has 9,100 train, 460 validation and 440 test examples.
Whole design families determine the splits; both deliveries stay together.
The mixed projector training excluded three overlong older pairs, leaving 9,094
older training examples. **The release contains the full original collection**,
including those clips and weak contrasts; it does not apply new filtering.

The Neu texts span ten everyday domains. Each pair type has 1,250 utterances.
Texts average 13.7 words and teacher replies 23.1 words. Audio totals **14.8 hours**,
averaging **5.32 seconds**; original float32 WAV payload is **5.11 GB**. The Neu
collection has 93 identical teacher-reply pairs (1.86%); these are retained.
Older Qwen sad/sarcastic deliveries and weak teacher contrasts were a motivating
quality issue, not a hidden exclusion criterion.

## Generation pipeline

1. Assign a domain, situation, intent, concrete fact, split family and plausible
   pair of deliveries. Generate a user utterance of at least seven words whose
   literal content supports both deliveries. A structural check enforces IDs,
   coverage, word limits and exact duplicate rules. The Neu stage also excludes
   explicit emotion-label words. There is no automatic perceptual quality filter.
2. Generate two replies with frozen **Qwen3.5-2B**, given the utterance and metadata
   explaining how the user delivered it. The metadata is supplied to the teacher
   and never spoken. The prompt requests a concise response and discourages
   naming the tone or inventing events. It does not guarantee factual correctness.
3. Synthesize the identical text twice. Qwen CustomVoice uses written delivery
   instructions. NeuTTS-2E uses its explicit emotion argument and fixed Paul
   speaker. Those controls belong to synthesis, not to the audio transcript.
4. Preserve accepted audio, actual termination evidence, WAV hashes, pinned model
   revisions, configurations and failed attempts. Resume only missing work.

Authoring used Qwen3.5-4B and teacher replies used Qwen3.5-2B. Text generation used
vLLM; older production audio used vLLM-Omni, while Neu audio used batched official
PyTorch generation. Their isolated environments have different dependencies:
installing all backends together into the training environment is not the
documented reproduction path.

| Task | Entry point | Supporting implementation |
|---|---|---|
| Older texts/teacher targets, vLLM | [`generate_emotional_vllm.py`](../scripts/generate_emotional_vllm.py) | [`emotional_generation.py`](../speech_projector/emotional_generation.py), [`emotional_dataset.py`](../speech_projector/emotional_dataset.py) |
| Older Qwen audio, vLLM-Omni | [`generate_emotional_audio_omni.py`](../scripts/generate_emotional_audio_omni.py) | [`emotional_audio_omni.py`](../speech_projector/emotional_audio_omni.py) |
| Fresh Neu texts/teacher targets | [`generate_neu_dataset.py`](../scripts/generate_neu_dataset.py) | [`neu_dataset.py`](../speech_projector/neu_dataset.py), [`neu_generation.py`](../speech_projector/neu_generation.py) |
| Neu audio | [`generate_neutts_corpus.py`](../scripts/generate_neutts_corpus.py) | [`benchmark_neutts_batch.py`](../scripts/benchmark_neutts_batch.py), [`neutts_corpus.py`](../speech_projector/neutts_corpus.py) |
| Portable dataset export | [`export_emotional_dataset.py`](../scripts/export_emotional_dataset.py) | [`dataset_release.py`](../speech_projector/dataset_release.py) |

Every generation entry point requires its typed `--config`. Exported
`generation/` files preserve the actual prompts, parameters and model revisions.
Those historical files contain original node paths: for a new generation run,
set fresh output/model paths and record the executing source revision. Preserve
the original files as evidence rather than silently changing their attribution.
Some stages resumed after source fixes, so a config's declared source commit is
not always the sole execution commit. The backed-up source handoffs document
that history; byte-identical regeneration is not promised across runtimes.

For example, after preparing new configuration files in the appropriate GPU
environment:

```powershell
python -m scripts.generate_neu_dataset --config .\neu_drafts.json --stage drafts
python -m scripts.generate_neu_dataset --config .\neu_targets.json --stage targets
python -m scripts.generate_neutts_corpus --config .\neu_audio.json
```

The Neu drafting configuration refers to the older utterance manifest to avoid
exact overlap. The export keeps normalized public examples for both collections;
the Neu source manifest and generation schemas are separately preserved in the
local replay. Generating fresh data requires those source manifests and accepted
upstream model access, not just a teacher config.

## Local export and validation

From this repository, with the already verified local source backups present:

```powershell
uv run python -m scripts.export_emotional_dataset --config .\configs\release\neu_export.json
uv run python -m scripts.export_emotional_dataset --config .\configs\release\qwen_export.json
uv run python -m scripts.export_emotional_dataset --config .\configs\release\neu_export.json --verify
uv run python -m scripts.export_emotional_dataset --config .\configs\release\qwen_export.json --verify
```

The output is under ignored `results_preview/publication/`. The exporter requires
a new or empty destination and leaves both sealed source trees untouched.
It checks exact pair coverage, identical spoken text, distinct deliveries,
whole-family split separation, teacher/audio identity, true completion evidence,
WAV hashes, finite mono 24 kHz content and recorded durations.

Each release has split-specific Parquet shards, with **the original WAV bytes
embedded unchanged**, a small `examples.jsonl` index, JSON schema, generation
configs and source manifests, dataset card and export receipt. Group by `utterance_id` to compare the
two deliveries; retain `family_id` for clustered evaluation. Parquet audio uses
the Hub's documented `bytes`/`path` structure. [Hugging Face audio format](https://huggingface.co/docs/hub/en/datasets-audio)

| Public field | Meaning |
|---|---|
| `case_id`, `utterance_id`, `family_id` | Delivery example, same-text pair, split/evaluation family |
| `split`, `domain`, `intent` | Original assignment metadata |
| `text`, `delivery` | Exact spoken words and intended synthesis delivery |
| `teacher_response` | Existing Qwen reply conditioned on text plus intended delivery |
| `audio`, `audio_sha256` | Embedded original WAV and its byte hash |
| `sample_rate`, `duration_seconds` | Original audio measurements |

## Source terms and release scope

Keep the two emotional collections separate from DeepDialogue-derived examples:

- **DeepDialogue-xtts** declares **CC BY-NC 4**. Its derived ordinary collection
  is excluded from these exports. Cite the source and retain its restrictions if
  publishing that collection separately. [Dataset card](https://huggingface.co/datasets/SALT-Research/DeepDialogue-xtts)
- **NeuTTS-2E** uses **NeuTTS Open License 1.0**, not Apache 2.0. Section 5(a)
  explicitly includes generated outputs in its commercial-use conditions. Its
  revenue threshold is **US$5 million**; preserve the supplied terms rather than
  claiming unrestricted commercial use. The pinned license is included unchanged
  in the Neu release. [Model card](https://huggingface.co/neuphonic/neutts-2e),
  [pinned license](licenses/NeuTTS-Open-License-1.0.txt)
- **NeuCodec, Qwen3.5-2B and Qwen CustomVoice** list Apache 2 for their models.
  Those model licenses do not automatically select a license for this project's
  code or authored dataset collection. [NeuCodec](https://huggingface.co/neuphonic/neucodec),
  [Qwen text model](https://huggingface.co/Qwen/Qwen3.5-2B),
  [Qwen TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)

The collection contributions use **CC BY 4.0**. Neu-generated audio is excluded
from that grant and remains governed by the included NeuTTS output terms; the
Neu card uses `license: other` to make that distinction visible. The Qwen card
uses `license: cc-by-4.0`. See the [collection grant](dataset_cards/LICENSE.md).
No model
weights, cloned reference recordings, frozen feature caches or checkpoints are
included. Neu's existing watermark is preserved.

## Preparation status

Both full collections were exported locally and all **20,000 embedded original
WAVs** were independently read back from their Parquet shards and hash-checked.
Each collection has 41 shards and exact 9,100/460/440 example split coverage.
The local `export_receipt.json`, `export_verification.json` and reproducibility
inventory preserve exact counts and byte evidence. Original source archives
remain unchanged. Public release staging is separate, under
`results_preview/publication/public_20261007/`; unchanged Parquet files are linked
to the existing verified exports to avoid duplicating their storage. New cards,
license files and inventories belong to the public release, leaving the older
sealed preparation snapshots intact.

Public destinations are
[Neu paired emotional speech](https://huggingface.co/datasets/BertilBraun/neu-paired-emotional-speech)
and [Qwen paired emotional speech](https://huggingface.co/datasets/BertilBraun/qwen-paired-emotional-speech).
Publication receipts and status are recorded in the [publication guide](publication.md).

Useful applications are paired conditioning experiments and evaluation on
synthetic speech. These collections are not a benchmark of natural emotional
understanding: each uses one voice, delivery strength varies, labels can be
perceptually wrong, teacher contrasts can be weak and replies can invent facts.
The two collections also use different prompts and backends, so a pooled result
must account for those differences.
