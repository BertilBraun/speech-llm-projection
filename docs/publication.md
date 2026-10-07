# Manual publication

The user will create the repositories and upload manually. No GitHub remote or
Hugging Face publication destination has been configured, and nothing has been
uploaded. The local code and dataset preparation do not require a new GPU run.

## GitHub

The main README is the concise project report. The full report, generation guide
and serving handoff live in `docs/`; reviewed measurements and response examples
live in `results/followup_20261007/`. Large data, checkpoints and operational
archives remain ignored.

Choose a code license before describing the repository as open source. The
checkout currently has no project-wide license grant. Third-party terms in
`docs/licenses/` are attribution evidence, not a license for our code.

After creating the desired GitHub repository:

```powershell
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin master
```

This publishes the existing Git history, including historical experiment notes
and source attribution. Do not add ignored backup trees, saved authentication
material or model weights. The complete historical scientific evidence remains
available locally; the release does not rewrite it.

## Hugging Face

Publish the Neu and older Qwen collections as separate dataset repositories.
The [generation/release guide](dataset_release.md) explains the source terms and
limitations. Complete local packages are prepared under:

```text
results_preview/publication/neu-paired-emotional-speech/
results_preview/publication/qwen-paired-emotional-speech/
```

Each package includes a card, source/output terms, original-audio Parquet shards,
example index, generation configurations and verification receipts. The card's
`license: other` describes those supplied terms; select and document the
publisher's additional collection license before public upload. The Neu output
conditions must remain visible. Add the final GitHub source URL to each card so
readers can find the generation scripts.

| Verified local package | Examples | Parquet shards | Package size |
|---|---:|---:|---:|
| Neu | 10,000 | 41 | 4.81 GB |
| Older Qwen | 10,000 | 41 | 5.92 GB |

All embedded WAV hashes were checked against the original source receipts, and
each complete package inventory was independently verified. These sizes describe
the compressed release packages, rather than the larger raw WAV payloads.

Install the Hub CLI in a separate publication environment if it is not already
available, then authenticate interactively. Do not paste a token into chat or
save it in this repository. After creating each dataset repository on the Hub:

```powershell
hf auth login
hf upload YOUR_ACCOUNT/neu-paired-emotional-speech .\results_preview\publication\neu-paired-emotional-speech --repo-type dataset
hf upload YOUR_ACCOUNT/qwen-paired-emotional-speech .\results_preview\publication\qwen-paired-emotional-speech --repo-type dataset
```

The first upload is several GB. Upload the prepared folders, not the much larger
research replay. Split paths are explicitly declared in each card so additional
JSON/config files are not interpreted as data splits. Exact hashes and split
counts stay in receipts even though prose measurements are rounded.

The CLI's supported upload mechanism and authentication are documented in the
[official Hub upload guide](https://huggingface.co/docs/huggingface_hub/en/guides/upload).
After upload, inspect the dataset viewer, play both deliveries of a few pairs,
and confirm the split counts against the local receipt. Remote upload and viewer
verification remain the publisher's next step.
