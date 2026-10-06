# Qwen teacher-target rerun

Started 2026-10-06. One RTX3090; completed original experiments remain in results/.

- [x] User authorized 20k clean pairs, Qwen-generated targets and fresh compression runs.
- [x] Independent dataset, teacher-generation and evaluation tasks delegated.
- [x] Verified spoken-text manifest: 20k train / 512 validation / 512 test, complete dialogue splits.
- [ ] New selected audio downloaded and missing Whisper features cached; existing features reused.
- [x] Shared teacher/student prompt confirmed and teacher quality inspected.
- [x] Resumable generation validated; 320 bootstrap replies complete, larger journal pending.
- [ ] 20k training targets and fixed heldout teacher targets generated.
- [ ] Teacher-forced accuracy and response-quality judge validated.
- [ ] New gradient/frozen-weight smoke and tiny teacher-target feasibility run passed.
- [ ] Fresh 20k MLP at 10 pseudo-tokens/sec launched and completed.
- [ ] Controlled 25 / 10 / 5 / 2.5 pseudo-tokens/sec comparison completed.
- [ ] Linear versus MLP comparison at selected compression completed; convolution skipped.
- [ ] Transcript and ASR baselines evaluated on the fixed larger heldout set.
- [ ] Results, uncertainty estimates, qualitative comparisons and resource report saved.
- [ ] Checkpoints and self-contained result package backed up off node and verified.

The current implementation phase repairs the training target, spoken-text alignment
and evaluation sample size. Full-vocabulary hidden states will not be stored for
evaluation. Teacher-forced comparisons share the saved response prefix, which
preserves token alignment; generated-response quality is assessed separately.

## Current execution state

Remote source is frozen at 8da1b6c while the shared teacher journal is populated.
The supervisor is running the gradient smoke after completing 256 training and
32/32 heldout bootstrap targets. Generation took 190.6 seconds, produced 73,031
tokens, and peaked at 7.51 GB allocated VRAM. Full 20k/512/512 generation follows
the small training/audio-conditioning gate and missing-feature extraction.

Teacher and student use native chat formatting, two previous turns, and no custom
system or brevity instruction. Non-thinking generation starts with 2,048 tokens
and retries capped replies at 4,096; incomplete targets are rejected. The failed
256/512-budget attempt is archived separately, including its seven completed
targets and capped attempts. Missing cached checkpoint metadata was repaired
without changing model weights. The first weak judge calibration failed and is
preserved; a candidate-only rubric and a gated 4B fallback are being validated.

Local batched speech evaluation and explicit completion metadata passed 55
focused tests; these changes are queued for deployment after the teacher journal
finishes so resume provenance remains consistent. Final generation coverage will
be 512 validation and 512 test per main run; the small feasibility run is labeled
separately. Native Qwen target verbosity increases runtime relative to the first
program, and the experiment budget will be checked against measured throughput.
