# Overnight speech projector experiment

Started 2026-10-05 23:56 Europe/Berlin. Node: 1 RTX3090, 45.1GiB cgroup RAM limit, 200GB disk.

- [x] SSH access and node operating guide checked.
- [x] Empty repository initialized.
- [x] Canonical typed data/config/result models defined.
- [x] DeepDialogue actual metadata inspected (243,295 utterance rows).
- [x] Dialogue-disjoint manifests, filtering and audio quality inspection.
- [x] Selective audio acquisition and reusable final Whisper state cache (first1k+256heldout).
- [x] Frozen Qwen wrapper, target masking, gradient and weight invariance verification.
- [x] CPU sanity tests, formatting and lint checks (56 tests passed locally and in isolated node staging).
- [x] GPU smoke test, memory and throughput profile, initial generations.
- [x] V0 small subset training, evaluation and audio-conditioning diagnostics.
- [x] Text and ASR baselines on fixed validation/test examples.
- [x] V1 nested 1k/3k/10k data sizes.
- [ ] V2 compression curve.
- [ ] V3 linear/MLP/learned temporal comparison.
- [ ] Aggregate metrics, plots, readable qualitative examples and failures.
- [ ] Copy irreplaceable results/checkpoints off the nonpersistent node.

## Experiment state

Updated 2026-10-06 03:04 Europe/Berlin.
Completed: V0, all four heldout baselines, training/untrained conditioning probes,
first cache and ASR, model gradient/memory profile, 10k audio download.
Completed V1:1k,3k and10k runs, including full validation/test and conditioning diagnostics.
Completed V2:25 tokens/s with10k examples, validation/test CE1.7704/1.6912.
Running: V2 5 tokens/s with10k examples; no other GPU workload.
Completed:30k audio download,20k new clips/6.889GB in84 minutes, zero failures.
Full30k train plus128val/128test waveform SHA256 audit found zero cross-split overlaps.
Queued: V2 2.5 tokens/s and V3 linear/convolutional comparison.
Registered but stopped:20k two-epoch V1 extension at10 tokens/s, to start after the core
suite if timing remains reasonable. The30k configuration remains available but unqueued.
Prepared: first-epoch checkpoint preservation for a2500-update/20k-exposure comparison
against the10k two-epoch model, with supplementary evaluation kept separate.
Prepared: verified result package with original80MB Parquet, exact subset manifest,
16 audio clips,16 raw feature tensors, model revisions and pinned dataset provenance.
Dataset revision0495356d589c06f08253ab29ad1a1482e05d90f8 matches the original Parquet SHA256.
Typed input API, canonical stage/condition enums and discriminated projector configs
validated in separate node staging, pending post-training GPU parity for all architectures.
Local candidate commitf81c483. Seeded projector weights/forward outputs are byte-identical
to the running source for linear,MLP andconvolutional architectures.
An external CPU watcher will pause the suite briefly after V2 to review compression
quality before V3; the live source and training process remain preserved.
Canonical remote source/Git remains da16b58 until every queued training run finishes.
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
V1_1000 final validation/test CE2.0871/2.0057; semantic0.3719/0.3714.
V1_3000 final validation/test CE1.9554/1.8756; semantic0.3912/0.3763.
3k paired shuffled-audio CE margins:validation+0.1323(SE0.0298),test+0.1458(SE0.0216).
Decoded topic grounding remains weak despite measurable loss-based conditioning.
V1_10000 final validation/test CE1.8061/1.7171; semantic0.4143/0.3830.
10k paired shuffled-audio CE margins:validation+0.4521(SE0.0844),test+0.3216(SE0.0486).
The10k checkpoint is selected for V2/V3; saturation is not established. Data size and
optimization exposure are confounded by equal epochs, and semantic improvements are noisy.
Best10k validation checkpoint is step2400, CE1.7957, retained separately from final2500.
10k+256heldout cache:5.687GB, latest7k extraction46.79GPU-synchronized seconds,
98.94s pipeline wall time. V0/1k/3k/10k checkpoints and baselines backed up locally.
Package hashes verified for46 copied files/118.60MB, including original metadata and audit assets.
All-result SHA256 inventory/relocation verification is ready for the final stopped-writer backup.
Remaining: finish V2/V3, assess optional larger-data run, evaluate promising best checkpoints,
validate the typed API on GPU, finalize the research report, verify and back up all artifacts.
Planned: initial V0 256–1,000 examples at factor5 (10 tokens/s); matrix adapts to measured speed.
No long run starts until V0 correctness gates pass.
