# Overnight speech projector experiment

Started 2026-10-05 23:56 Europe/Berlin. Node: 1 RTX3090, 45.1GiB cgroup RAM limit, 200GB disk.

- [x] SSH access and node operating guide checked.
- [x] Empty repository initialized.
- [x] Canonical typed data/config/result models defined.
- [x] DeepDialogue actual metadata inspected (243,295 utterance rows).
- [x] Dialogue-disjoint manifests, filtering and audio quality inspection.
- [x] Selective audio acquisition and reusable final Whisper state cache (first1k+256heldout).
- [x] Frozen Qwen wrapper, target masking, gradient and weight invariance verification.
- [x] CPU sanity tests, formatting and lint checks (87 local tests; 79 passed isolated node staging).
- [x] GPU smoke test, memory and throughput profile, initial generations.
- [x] V0 small subset training, evaluation and audio-conditioning diagnostics.
- [x] Text and ASR baselines on fixed validation/test examples.
- [x] V1 nested 1k/3k/10k data sizes.
- [x] V2 compression curve.
- [x] V3 linear/MLP/learned temporal comparison.
- [ ] Aggregate metrics, plots, readable qualitative examples and failures.
- [ ] Copy irreplaceable results/checkpoints off the nonpersistent node.

## Experiment state

Updated 2026-10-06 06:07 Europe/Berlin.
Completed: V0, all four heldout baselines, training/untrained conditioning probes,
first cache and ASR, model gradient/memory profile, 10k audio download.
Completed V1:1k,3k and10k runs, including full validation/test and conditioning diagnostics.
Completed V2:25 tokens/s with10k examples, validation/test CE1.7704/1.6912.
Completed V2:5 tokens/s, validation/test CE1.8558/1.7528 andcosine0.4415/0.3407.
Its validation shuffle margin is+0.3184(SE0.0667), below25Hz+0.5025(SE0.0807).
Five-Hz validation semantic gains do not replicate ontest; lower-rate quality remains mixed.
Completed V2:2.5 tokens/s, validation/test CE1.8807/1.7678 andcosine0.4053/0.3836.
Mean18.203pseudo-tokens; validation shuffled margin+0.3613(SE0.0724).
Twentyfold compression remains functional but topic recovery is inconsistent.
Completed V3 pooledlinear at25 tokens/s: validation/test CE1.78585/1.69272,
semantic0.40872/0.41229,1,576,448 parameters; locally backed up with raw records preserved.
Validation shuffled margin+0.55467(SE0.09239), no-history margin+0.72533(SE0.09743).
Completed V3 convolutional at25 tokens/s: validation/test CE1.90110/1.80423,
semantic0.35038/0.34481,4,068,608 parameters; locally backed up.
Fullcore suite EXITED0 at05:50:36Berlin, no failures or activecore run.
Running:20k two-epoch V1 extension at10 tokens/s, started05:51:22Berlin after verified core success.
Completed:30k audio download,20k new clips/6.889GB in84 minutes, zero failures.
Full30k train plus128val/128test waveform SHA256 audit found zero cross-split overlaps.
V3 reuses the completed MLP comparison at25 tokens/s. Linear nearly matches MLP with
fewer parameters; convolution has worse losses, semantic similarity and audio sensitivity.
Reviewed all4 rates using validation CE, validation controls and fixed validation topics;
selected25Hz and resumed the suite after a70-second pause. Decision saved inresults.
Twenty-k extension uses the unchanged nested collected prefix for controlled comparisons;
source/manifest remain frozen and the first-epoch snapshot watcher will preserve2500updates.
The30k configuration remains available but unqueued.
Prepared: first-epoch checkpoint preservation for a2500-update/20k-exposure comparison
against the10k two-epoch model, with supplementary evaluation kept separate.
Prepared: verified result package with original80MB Parquet, exact subset manifest,
16 audio clips,16 raw feature tensors, model revisions and pinned dataset provenance.
Dataset revision0495356d589c06f08253ab29ad1a1482e05d90f8 matches the original Parquet SHA256.
Typed input API, canonical stage/condition enums and discriminated projector configs
validated in separate node staging, pending post-training GPU parity for all architectures.
Local code candidate commit75c1c04. Seeded projector weights/forward outputs are byte-identical
to the running source for linear,MLP andconvolutional architectures.
The external CPU watcher paused the suite after V2 for the completed review and exited.
The live source and training process remained preserved; no completed run was repeated.
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
98.94s pipeline wall time. V0/1k/3k/10k/allV2 checkpoints and baselines backed up locally.
Exact selected-subset distributions and manifest-prefix agreement are audited and saved.
Package hashes verified for46 copied files/118.60MB, including original metadata and audit assets.
All-result SHA256 inventory/relocation verification is ready for the final stopped-writer backup.
Remaining: finish the authorized20k run, evaluate matched-budget and a promising best checkpoint,
validate the typed API on GPU, run the clean V0 control and synthesis-text baseline,
finalize the research report, verify and back up all artifacts.
Late audit: the initial random quality check missed audio/text alignment defects.
Training-prefix256/1k/3k/10k/20k mismatches:6/21/55/187/389;20k audio-target matches357.
Heldout3validation and3test mismatches:five audio originals equal the assistant target.
Original text baseline is an original-dialogue-text reference, not a perfect waveform transcript.
Aligned125-case per-split sensitivity preserves positive audio margins and the25Hz validation choice.
No fixed-case topic audit examples are misaligned. Raw128-case comparisons and manifests preserved.
Twenty-k expansion reauthorized on the same collected data to preserve controlled comparisons;
known target-audio contamination is prominent in the report and must be cleaned before follow-up.
Canonical synthesis audit and actual-synthesis-text baseline helper tested; baseline GPU run pending.
Prepared separate clean V0 control: first256 retained training pairs after excluding known
turn/audio mismatches,125 aligned validation and125 aligned test examples. All existing
cached features are required; original matrix inputs stay unchanged. Same5-epoch V0
configuration, followed by clean training/untrained-heldout audio-conditioning probes.
This late control addresses accidental target-audio supervision in six original V0 samples.
