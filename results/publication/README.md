# Public paired emotional speech releases

Published on 7 October 2026 with original audio and teacher targets unchanged.

| Collection | Examples | Original WAV Parquet shards | Verified files | Release size |
|---|---:|---:|---:|---:|
| [Neu](https://huggingface.co/datasets/BertilBraun/neu-paired-emotional-speech) | 10,000 | 41 | 62 | 4.81 GB |
| [Qwen](https://huggingface.co/datasets/BertilBraun/qwen-paired-emotional-speech) | 10,000 | 41 | 59 | 5.92 GB |

Each collection has 5,000 utterances, two deliveries per utterance and
9,100/460/440 examples in train/validation/test. Anonymous API checks verified
every release file against its Git blob or LFS SHA256 at the pinned revisions:
[Neu receipt](neu_hub_verification.json) and [Qwen receipt](qwen_hub_verification.json).
The server-created `.gitattributes` is excluded from these payload file counts.

Neu's viewer exposes all three splits, paired test rows and reachable WAV
preview assets. The size aggregation and Qwen viewer were still indexing at the
check; pending service jobs must not be mistaken for zero data rows. Both full
public releases are independently available and byte-verified.

Collection contributions use CC BY 4.0. Neu-generated audio is excluded from that
grant and retains its supplied NeuTTS output terms. DeepDialogue-derived ordinary
training data and model weights are excluded from both releases.

See the [publication report](../../docs/publication.md) for source links,
verification commands, service status and resolved publication failures.
