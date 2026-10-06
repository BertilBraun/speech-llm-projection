# Qwen teacher-target rerun

Updated 2026-10-06. One RTX 3090; the original completed program remains sealed in results/.

- [x] User authorized 20k clean pairs, native Qwen replies as targets, fresh compression comparisons, and no convolution.
- [x] Independent dataset, model, and evaluation tasks delegated and merged.
- [x] Source verified: 20,000 training, 512 validation, and 512 test examples, with whole-dialogue splits and audio-cleaned transcripts.
- [x] All selected audio, 21,024 Whisper feature files, and held-out ASR transcripts ready; cached features reused.
- [x] Native chat uses the configured previous history and transcript/audio current turn; no custom system or style instruction.
- [x] Qwen-recommended non-thinking text sampling implemented with reproducible batch seeds and exact completion metadata.
- [x] Fresh sampled bootstrap: 320/320 EOS-complete; no retries or audit issues. Greedy pilot and noncompletion archived separately.
- [x] New smoke passed: projector gradients and updates, frozen Qwen unchanged, and no LLM parameter gradients.
- [x] Sampled V0 completed: 256 pairs over five epochs; fixed training CE 1.9065 to 1.5848; validation/test CE 1.7707/1.7422.
- [x] Training audio-conditioning gate passed: correct audio beats shuffled audio on all 32 cases; paired CE gap +0.23785 (SE 0.04134). Held-out benefits are modest, and wrong-topic outputs are retained.
- [x] Instruction-only 4B judge passed the original 13 and independent 30 cases. One generic nonanswer false positive is documented; earlier judge failures are preserved.
- [ ] Full 21,024 sampled teacher targets complete. Currently running; 896 completed at the latest observation, with node source frozen at 31c26b5.
- [ ] Full target, split, and ASR audits complete. Preliminary ASR WER is 2.14% on validation and 1.84% on test; one uncertain waveform mismatch is flagged, and canonical inputs remain unchanged.
- [ ] Fresh 20k MLP training and evaluation at 10 Hz complete.
- [ ] Controlled 25/10/5/2.5 Hz comparison complete.
- [ ] Linear versus MLP comparison at the selected compression complete.
- [ ] Nested 1k/3k/10k scaling reruns complete, with the 20k reference and fixed 512/512 held-out sets.
- [ ] Transcript and ASR baselines, teacher-prefix fidelity, calibrated main judgments, and small paired audio-control judgments complete.
- [ ] Scientific report, curves, fixed 16-case comparisons, and resource accounting complete.
- [ ] Final package sealed after writers stop, backed up locally, and hash-verified.

## Execution and scientific notes

The supervised pipeline is running full_teacher_targets. Cache work is marked reused so its actual extraction statistics are preserved. Node source remains 31c26b5 during teacher generation; later report and control-analysis improvements are validated locally and await a safe handoff after generation.

Incomplete teacher replies are saved as failures and never used as targets. Greedy generation failed on an infinite playlist even at 4,096 tokens; the same example completed under production sampling in 289 tokens. The former greedy bootstrap, V0, and full-generation pilot are under results_teacher/pilot_greedy/ and excluded from fresh results. Native teacher replies can still hallucinate or contain awkward or mixed-language text; they are teacher-policy references, not verified factual gold.

Teacher-prefix scoring feeds the identical saved response prefix to the transcript and speech paths, preserving token alignment. Exact KL, top1, and early-token metrics compare raw model distributions, while free-running quality is judged separately. Main generations use batched decoding with completion metadata; scalar V0 completion status remains explicitly unknown. History strata and correct-versus-wrong-audio controls distinguish speech use from generic fluent replies.

All 228 tests passed locally and on the node for the deployed source; Ruff format/check passed. Validated local report updates explicitly disclose historical failures. No emotional training, LoRA, speech output, streaming, or serving work is included.
