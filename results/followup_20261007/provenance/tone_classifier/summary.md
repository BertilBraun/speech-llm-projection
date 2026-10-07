# Frozen Whisper intended-tone classifier — Neu-only baseline

Single fixed CPU fit, with StandardScaler and balanced LogisticRegression fit only on TRAIN. C=1, LBFGS, maximum 1000 iterations, seed42; no held-out tuning or rejection.
Mean/std plus four relative temporal-bin means of raw frozen encoder states: 4608 dimensions. No literal text, explicit duration, waveform statistics or intended label is an input feature. Predictions contain no oracle label.

| Split | Clips | Families | Balanced accuracy | Macro F1 | Both deliveries correct / pairs |
|---|---:|---:|---:|---:|---:|
| train | 9100 | 455 | 1.000000 | 1.000000 | 4550/4550 |
| validation | 460 | 23 | 0.995690 | 0.996744 | 229/230 |
| test | 440 | 22 | 1.000000 | 1.000000 | 220/220 |

Only validation error: one intended fearful clip classified sad. All 440 fixed TEST clips, including the fixed64 emotional response panel, have correct intended labels. This means predicted/reference-tone cue inputs coincide on that panel; the two baseline methods retain separate provenance.
Feature load and hash 14.913937s; fit 3.720707s; total function wall 22.717704s. Interpreter/import startup and the separate preparation receipt are outside that timer.
Feature matrix: 10000×4608 float64 = 368,640,000 bytes. The temporary stack can briefly double this matrix allocation. One process observation during execution reported RSS 1,876,056 KiB = 1.921 decimal GB; this is not a measured peak.
CUDA was hidden; preparation reported CUDA unavailable. Classifier PID180765 was absent from the compute-process query; only the existing training PID was present.
Classifier feature source bd90c52d9e9a14f2f8152baea6441e5283a54fd9; node base remained 46ed24257606a72ace17c54064e56344d989088c. Exact code/script/config/Supervisor hashes are in deployed_files.jsonl. sklearn 1.9.1.
All family, normalized literal-text and cached-feature/audio paths were checked for cross-split overlap. All pairs retain exact words, split, family and two distinct intended labels. Report.json pins every one of the 10000 feature files.

## Interpretation limits

The classifier predicts intended labels on single-Paul synthetic Neu audio. High scores establish available synthetic acoustic information, which can include vocoder/prosody signatures; they do not verify perceived emotion, natural-speaker generalization, or conversational response quality. Old Qwen labels are not aliased into these four classes. Same-family sharing of broad contexts across explicit families remains a limitation of the original dataset, not new classifier data.

Local model/prediction copies and all four deployed files matched their canonical size/SHA256 records. No source audio/features or previous sealed package changed.
