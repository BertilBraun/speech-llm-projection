# Publication

The source repository is
[BertilBraun/speech-llm-projection](https://github.com/BertilBraun/speech-llm-projection).
The user prepared and pushed the GitHub repository; the Hugging Face datasets
are published from this checkout at the user's request. Publication uses
the saved Hub login and requires no new GPU work.

## GitHub

The main README is the concise project report. The full report, generation guide
and serving handoff live in `docs/`; reviewed measurements and response examples
live in `results/followup_20261007/`. Large data, checkpoints and operational
archives remain ignored.

Choose a code license before describing the repository as open source. The
checkout currently has no project-wide license grant. Third-party terms in
`docs/licenses/` are attribution evidence, not a license for our code.

The configured GitHub remote receives committed source and documentation:

```powershell
git push -u origin master
```

This publishes the existing Git history, including historical experiment notes
and source attribution. Do not add ignored backup trees, saved authentication
material or model weights. The complete historical scientific evidence remains
available locally; the release does not rewrite it.

## Hugging Face

The Neu and older Qwen collections have separate public dataset destinations.
The [generation/release guide](dataset_release.md) explains the source terms and
limitations. Complete local packages are prepared under:

```text
results_preview/publication/neu-paired-emotional-speech/
results_preview/publication/qwen-paired-emotional-speech/
```

Each package includes a card, source/output terms, original-audio Parquet shards,
example index, generation configurations and verification receipts. The original
prepared exports above remain sealed. Public staging with final cards, source
links and new inventories is under `results_preview/publication/public_20261007/`.
Collection contributions use CC BY 4.0; Neu audio is expressly excluded and
retains the NeuTTS output terms. Its card therefore uses `license: other`.

| Verified local package | Examples | Parquet shards | Package size |
|---|---:|---:|---:|
| Neu | 10,000 | 41 | 4.81 GB |
| Older Qwen | 10,000 | 41 | 5.92 GB |

All embedded WAV hashes were checked against the original source receipts, and
each complete package inventory was independently verified. These sizes describe
the compressed release packages, rather than the larger raw WAV payloads.

The existing project environment includes the Hub CLI. The current saved login
was used without printing or copying credentials into the repository. To upload
or resume the same prepared public release:

```powershell
uv run hf upload BertilBraun/neu-paired-emotional-speech .\results_preview\publication\public_20261007\neu-paired-emotional-speech . --repo-type dataset --no-private
uv run hf upload BertilBraun/qwen-paired-emotional-speech .\results_preview\publication\public_20261007\qwen-paired-emotional-speech . --repo-type dataset --no-private
```

The first upload is several GB. Upload the prepared folders, not the much larger
research replay. Split paths are explicitly declared in each card so additional
JSON/config files are not interpreted as data splits. Exact hashes and split
counts stay in receipts even though prose measurements are rounded.

The CLI's supported upload mechanism and authentication are documented in the
[official Hub upload guide](https://huggingface.co/docs/huggingface_hub/en/guides/upload).
After upload, verify anonymous public access, file coverage and every Git/LFS
content hash at an immutable Hub revision:

```powershell
uv run python -m scripts.verify_hub_dataset --directory .\results_preview\publication\public_20261007\neu-paired-emotional-speech --repository BertilBraun/neu-paired-emotional-speech --output .\results\publication\neu_hub_verification.json
uv run python -m scripts.verify_hub_dataset --directory .\results_preview\publication\public_20261007\qwen-paired-emotional-speech --repository BertilBraun/qwen-paired-emotional-speech --output .\results\publication\qwen_hub_verification.json
```

Also check the viewer's split counts and first rows after its asynchronous
indexing completes. Audio bytes and split coverage were independently verified
before upload; viewer availability is a separate service check.

## Completed publication, 7 October 2026

| Public collection | Examples | Shards | Matching public files | Size |
|---|---:|---:|---:|---:|
| [Neu](https://huggingface.co/datasets/BertilBraun/neu-paired-emotional-speech) | 10,000 | 41 | 62 | 4.81 GB |
| [Qwen](https://huggingface.co/datasets/BertilBraun/qwen-paired-emotional-speech) | 10,000 | 41 | 59 | 5.92 GB |

Both anonymous API checks confirmed public access. Every local file matched the
remote file's exact size and Git blob or LFS SHA256 content hash. Each collection
retains its original 9,100/460/440 examples across train/validation/test. The
independent pre-upload checks read back all 20,000 embedded WAVs; final public
verification compares the containing shards and all supporting files.

Pinned verification receipts:

- [Neu](../results/publication/neu_hub_verification.json), revision
  `c973637893abe730314df06e0514be2bcc0cf802`.
- [Qwen](../results/publication/qwen_hub_verification.json), revision
  `40c0538dec46602a8f751998c69988ae9a3f5642`.

Neu's viewer recognizes all three splits and serves paired test preview rows
with WAV audio. Qwen's viewer and the dataset-size aggregation jobs were still
indexing at the final check; the service returned busy/not-ready responses.
This does not prevent access to the verified public Parquet files. Viewer
readiness is separate from the completed upload and file-integrity checks.

The initial local staging copy exhausted available disk space. The user removed
only that newly created duplicate directory; replacement staging links the
unchanged Parquet shards and copies the small metadata. Original sealed releases
were preserved. Two Qwen CLI metadata retries encountered TLS EOF errors while
calling the redundant repository-creation endpoint. Uploading to the existing
repository through the same official SDK succeeded with normal TLS verification.
No audio was retranscribed, regenerated, filtered or recompressed.

The source changes passed `uv run pytest -m "not integration"` (**618 passed,
two infrastructure tests deselected**), `uv run ruff format`,
`uv run ruff check --fix`, link checks, dataset-card parsing and figure inspection.
Generated SVG whitespace initially failed the diff check; vector export now
normalizes it, both SVGs parse successfully, and the final diff check passes.
