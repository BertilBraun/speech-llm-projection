# Overnight speech projector experiment

Started 2026-10-05 23:56 Europe/Berlin. Node: 1 RTX3090, 46GiB RAM, 200GB disk.

- [x] SSH access and node operating guide checked.
- [x] Empty repository initialized.
- [x] Canonical typed data/config/result models defined.
- [x] DeepDialogue actual metadata inspected (243,295 utterance rows).
- [ ] Dialogue-disjoint manifests, filtering and audio quality inspection.
- [ ] Selective audio acquisition and reusable final Whisper state cache.
- [ ] Frozen Qwen wrapper, target masking, gradient and weight invariance verification.
- [ ] CPU sanity tests, formatting and lint checks.
- [ ] GPU smoke test, memory and throughput profile, initial generations.
- [ ] V0 small subset training, evaluation and audio-conditioning diagnostics.
- [ ] Text and ASR baselines on fixed validation/test examples.
- [ ] V1 nested data sizes.
- [ ] V2 compression curve.
- [ ] V3 linear/MLP/learned temporal comparison.
- [ ] Aggregate metrics, plots, readable qualitative examples and failures.
- [ ] Copy irreplaceable results/checkpoints off the nonpersistent node.

## Experiment state

Completed: none. Running: environment installation. Failed: none.
Planned: initial V0 256–1,000 examples at factor5 (10 tokens/s); matrix adapts to measured speed.
No long run starts until V0 correctness gates pass.
