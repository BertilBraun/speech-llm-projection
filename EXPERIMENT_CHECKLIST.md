# Overnight speech projector experiment

Started 2026-10-05 23:56 Europe/Berlin. Updated 2026-10-06 08:28 Europe/Berlin.
Node: one RTX3090, 45.1GiB cgroup RAM limit, 200GB disk; no persistent volume.

- [x] SSH access, node guide and single-GPU resource constraints verified.
- [x] Modular typed dataset, cache, projector, frozen LLM, training, evaluation and launcher implemented.
- [x] Actual DeepDialogue schema, composite dialogue splits, filters and distributions inspected.
- [x] Full 30k-plus-heldout waveform hash audit: no cross-split overlap.
- [x] Whisper Small dimension/rate/padding documented; reusable uncompressed BF16 cache extracted through 20k.
- [x] Target-only loss, projector gradients and unchanged frozen model weights verified on GPU.
- [x] 87 canonical tests plus two analysis tests passed locally and in isolated node staging; Ruff clean.
- [x] V0: raw 256-pair feasibility run and training/heldout conditioning probes completed.
- [x] V1: nested 1k/3k/10k/20k runs completed with fixed heldout splits.
- [x] V2: 25/10/5/2.5 pseudo-tokens per second evaluated at 10k examples.
- [x] V3: linear, MLP and strided convolution compared at 25 tokens/sec.
- [x] Original dialogue-text and Whisper-ASR baselines completed.
- [x] Late synthesis-alignment audit and aligned heldout sensitivity completed.
- [x] Separate clean V0 with 256 retained training pairs and 125/125 aligned heldout pairs completed.
- [x] Strict GPU source parity passed for all three architectures; initial LINEAR failure retained.
- [x] All completed training checkpoints copied off node; 20k first-epoch snapshot SHA verified locally.
- [x] Matched-update 20k, actual synthesis-text and best-checkpoint supplementary evaluations.
- [x] Final aggregate report, plots, qualitative comparisons and resource/protocol records.
- [x] Source archives, package inventory and exact off-node SHA256 verification.

## Current state

All eleven training runs completed successfully. Clean V0 and its probe exited 0 at
07:57 Berlin. All three supplementary evaluations finished around 08:07 Berlin;
the supervisor exited and the GPU process inventory is empty. Reports, source
archives and the first full backup are complete. At 08:27:40 Berlin, local
verification passed for all 561 files / 714,058,539 bytes, matching the sealed node
inventory exactly. The final documentation seal and its verification are recorded
separately in results/reproducibility/results_inventory_verification.json.
No further training, 30k experiment, emotion objective or LoRA is queued.

Raw V1/V2/V3 and the 20k extension ran from archived da16b58. Strict post-training
GPU parity passed for MLP, LINEAR fresh retry and convolution; all tested CE values
and greedy generations matched exactly. The initial LINEAR parity process failed
and its cause remains unresolved; its original reports, replay and diagnostics
are preserved without relaxing tolerances. Validated a083c71 was deployed only
after the raw matrix completed. Clean V0 trained from that immutable revision.

Clean V0 fixed training CE fell 2.8834 to 1.0797. Its paired training audio margin
is +1.1299 (SE 0.0740), 32/32 wins. Heldout paired margins are +0.1345 (SE 0.0478)
on validation and +0.0920 (SE 0.0425) on test; generalization remains modest.
The strongest raw setting is 20k/10Hz MLP: final validation CE 1.7411 and selected
step 4800 independently evaluated CE 1.7358. At matched 2,500 updates, the 20k
checkpoint gives CE 1.7973 versus 10k's 1.8061; the larger final gain therefore
includes additional optimization. Supplementary evaluations remain separate from
the main final-run records.

At 10k, 25Hz MLP gives the lowest controlled validation CE 1.7704. At 2.5Hz,
CE 1.8807 and positive audio controls show that 20-fold compression remains
functional, although topic recovery is inconsistent. Linear nearly matches MLP with 45.4%
fewer parameters; convolution performs worse under the fixed recipe.

The late alignment audit found 389 wrong-turn audios among the first 20k training
pairs, 357 with synthesis text equal to the assistant target. Six original V0
training pairs and three examples in each heldout split are affected. Raw results
are preserved; clean V0 and aligned sensitivity are separate controls. These
findings prevent claiming a clean-data requirement or reliable general spoken
conversation. Follow-up work must repair alignment before new training.

Earlier repaired failures include the causal-conv wheel ABI mismatch, HTTP worker
failures, chat EOS mismatch and probe import setup. Invalid pre-EOS generations
remain archived. Original V0 launch source was not captured and its early best
checkpoint was not retained; both limitations are documented.

## Inspection and finalization

Read results/analysis/morning_summary.md for the concise findings,
results/analysis/research_report.md for complete tables and interpretations,
results/analysis/core_qualitative_comparison.md for fixed examples, and each run's
training_protocol.md for exact source/configuration evidence. Raw results remain
unchanged by supplementary evaluations. Full training audio/cache stays on node;
the portable package includes checkpoints, all metrics/generations, pinned source
metadata and representative audio/features. The exact inventory and verification
receipts identify the final delivered file set. Full code validation ran 87
canonical tests plus two analysis tests, ruff format and ruff check --fix. Strict
GPU tests verified projector gradients, frozen weights and architecture parity.
The rented node remains running with no active experiment GPU job.
