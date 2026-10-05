# Overnight speech projector experiment

Started 2026-10-05 23:56 Europe/Berlin. Node: 1 RTX3090, 46GiB RAM, 200GB disk.

- [x] SSH access and node operating guide checked.
- [x] Empty repository initialized.
- [x] Canonical typed data/config/result models defined.
- [x] DeepDialogue actual metadata inspected (243,295 utterance rows).
- [x] Dialogue-disjoint manifests, filtering and audio quality inspection.
- [x] Selective audio acquisition and reusable final Whisper state cache (first1k+256heldout).
- [x] Frozen Qwen wrapper, target masking, gradient and weight invariance verification.
- [x] CPU sanity tests, formatting and lint checks (34 tests passed; continuing checks).
- [x] GPU smoke test, memory and throughput profile, initial generations.
- [x] V0 small subset training, evaluation and audio-conditioning diagnostics.
- [x] Text and ASR baselines on fixed validation/test examples.
- [ ] V1 nested data sizes.
- [ ] V2 compression curve.
- [ ] V3 linear/MLP/learned temporal comparison.
- [ ] Aggregate metrics, plots, readable qualitative examples and failures.
- [ ] Copy irreplaceable results/checkpoints off the nonpersistent node.

## Experiment state

Updated 2026-10-06 00:49 Europe/Berlin.
Completed: V0, all four heldout baselines, training/untrained conditioning probes,
first cache and ASR, model gradient/memory profile, 10k audio download.
Running: V1 nested1k/3k/10k supervised queue; 30k audio expansion download-only.
Queued: adaptiveV2 andV3; possible30k scaling run after assessing10k.
Prepared: result packaging with exact subset manifest, 16 audio clips and model revisions;
typed input API cleanup validated in isolated staging, pending post-suite merge.
Failed/repaired: causal-conv wheel C++ ABI mismatch; rebuilt compatible source successfully.
GPU gradient check:1,881,825,088 frozen parameters fullSHA256 unchanged; projector updated.
Peak allocated4.10GB. Warm example backward0.13–0.61s; sequence-shape recompilation
being reduced by teacher-sequence bucketing. ASR heldout normalized WER4.215%.
Repaired: chatEOS mismatch (tokenizer248046 vs textconfig248044); invalid generations archived.
V0 fixedtrainCE2.8865→0.7580; heldoutvalCE2.9639→2.4122; testCE2.3370.
Training32probe correct-vs-shuffled margin1.6228±SE0.0911,32/32 correctaudio wins.
Heldoutgrounding remains modest; controls justify proceeding to data scaling.
Text/ASR validation CE2.566/2.576, semantic similarity0.512/0.521;
V0 semantic similarity0.354. CE alone does not measure conversational grounding.
V1_1000 step200 validation CE2.1385; final evaluation pending.
Planned: initial V0 256–1,000 examples at factor5 (10 tokens/s); matrix adapts to measured speed.
No long run starts until V0 correctness gates pass.
