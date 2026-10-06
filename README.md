# Speech projector overnight research

Frozen Whisper Small encoder → trainable temporal compressor/projector → frozen
Qwen3.5-2B text decoder. Previous dialogue turns use text, the current utterance
uses only cached speech features, and teacher-forced CE trains only the projector.

The experiment covers feasibility, nested data scaling, temporal compression and
linear/MLP/learned temporal architectures. DeepDialogue XTTS is synthetic spoken
dialogue, so conclusions concern this dataset rather than natural conversations.

Code uses typed Pydantic configuration/manifests and local JSON/JSONL artifacts.
The rented node is `/workspace/speech-projector`; checkpoints and results must
also be copied locally because the node has no persistent volume.

See [EXPERIMENT_CHECKLIST.md](EXPERIMENT_CHECKLIST.md) for live progress. Exact
commands and results will be added as the pipeline is validated.

## Reproducing the node environment

The validated environment is Python 3.12, PyTorch 2.6.0+cu124,
Transformers 5.13.0, Triton 3.2.0, flash-linear-attention 0.3.2 and
causal-conv1d 1.5.2. The causal convolution extension required a source build
with PyTorch's matching C++ ABI; the default downloaded wheel did not import.
The complete package inventory is saved in `results/environment.txt`.
Install the project itself with `uv pip install --no-deps -e .` so helper scripts
can import the package.

On this node use `/venv/main/bin/python` and `/venv/main/bin/ruff`. After
downloading the three model checkpoints, batch jobs enable offline model loading.

```text
python -m scripts.download_models
python -m speech_projector.data --root data --download-train 1000 --workers 4
python -m speech_projector.cache --root data --train-examples 1000
python -m speech_projector.launcher --manifest data/examples.jsonl --output results --smoke
python -m speech_projector.launcher --manifest data/examples.jsonl --output results
python -m speech_projector.launcher --manifest data/examples.jsonl --output results --suite
python -m speech_projector.report --root results --data-report data/dataset_report.json
```

Supply absolute paths for data/artifacts when working outside the project root.
The actual supervisor commands in `scripts/v0.conf` and `scripts/suite.conf`
use absolute remote paths. Runs resume their last optimizer checkpoint and skip
completed result files. Dataset manifests have a deterministic nested order.

Teacher-forced sequences are right padded to multiples of 64 to reduce kernel
recompilation. Labels mask all padding/history/speech positions. Generation
uses the actual unpadded prompt. A left-padding optimization was deferred after
an initial parity test diverged; that test preceded the chat stop-token repair.

## Reading the experiment artifacts

Start with `results/analysis/research_report.md`. It distinguishes measured runs
from pending work and combines losses, semantic response comparisons, audio
controls, fixed examples, resource accounting and unresolved limitations.
`results/analysis/scientific_comparison.png` shows data scaling and compression
with complementary metrics; `results/summary.csv` contains controlled final
checkpoint comparisons.

Each run directory contains its exact `config.json` and ordered `subset.jsonl`,
`train.jsonl`, final `result.json`, rolling optimizer checkpoint, and retained
best-validation projector. Its `validation` and `test` directories contain
metrics, per-example losses, paired audio controls and readable generations.
Supplementary best-checkpoint and matched-budget evaluations live in separate
subdirectories and leave the main final-checkpoint result unchanged.

`results/dataset` preserves the original metadata and quality/leakage audits.
`results/audio` and `results/features` contain matching representative waveforms
and raw encoder states. The reproducibility directory records immutable model
and dataset revisions, exact manifest, environment, asset hashes and full result
inventory. The full training audio/cache remains on the node; the result package
preserves the subset definition and representative audit assets.

After stopping all result writers, create the full inventory on the node and
verify it again after copying the result directory:

```text
python -m scripts.inventory_results write-inventory --results results --writers-stopped
python -m scripts.inventory_results verify --results results
```

The verification checks exact file membership, sizes and SHA256 hashes. Training
source and the final validated source are archived separately when they differ;
run metadata records the source used at completion/evaluation. Any one-time
linear-config cleanup preserves the exact original JSON under `original_records`.
