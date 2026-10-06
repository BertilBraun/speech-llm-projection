# Qwen teacher-target rerun

Started 2026-10-06. One RTX3090; completed original experiments remain in results/.

- [x] User authorized 20k clean pairs, Qwen-generated targets and fresh compression runs.
- [x] Independent dataset, teacher-generation and evaluation tasks delegated.
- [ ] Verified spoken-text manifest: 20k train / 512 validation / 512 test, complete dialogue splits.
- [ ] New selected audio downloaded and missing Whisper features cached; existing features reused.
- [ ] Shared teacher/student prompt confirmed and teacher quality inspected.
- [ ] Resumable teacher generation validated, targets complete and lengths measured.
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
