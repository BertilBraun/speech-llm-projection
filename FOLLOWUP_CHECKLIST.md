# 10 Hz and lexical alignment follow-up — 7 October 2026

Authorized start: 08:51 UTC / 10:51 Berlin. Target report 13:51 UTC / 15:51 Berlin; maximum requested window 14:51 UTC / 16:51 Berlin. Root owns the single GPU queue and repository README. Model, data and evaluation subagents have separate implementation ownership.

Previous sealed results, older audio and supervision remain unchanged. Existing speech features are reused. No TTS, extra dataset generation, LoRA, encoder tuning or convolution search is part of this follow-up.

## Implementation and validation

- [x] Read complete node guide; GPU initially empty; node 42 GB and local approximately 14 GB free.
- [x] Snapshot and hash original 10 Hz optimizer/checkpoint; immutable continuation accepts only run name and cumulative update ceiling.
- [x] Validate continuation launcher on actual manifest and fixed validation panel; Ruff passes.
- [ ] Typed optional transcript-reconstruction mixture, objective journal and exact resume tests.
- [ ] Ordinary-only response-distribution KL; emotional examples retain original response CE; frozen-gradient GPU verification.
- [ ] Train-only CPU tone classifier and predictions, disjoint fixed validation/test families.
- [ ] ASR plus predicted tone and reference-tone ceiling under otherwise identical prompting.
- [ ] Fully matched speech/TEXT/ASR multi-turn evaluation.
- [ ] Understandable factual diagnostics and direct blinded emotional comparison.

## GPU queue and controlled runs

Only root may launch GPU work. Proposed objective branches share the full 10 Hz parent, seed, sample order, learning rate 0.0002, effective batch eight, and 2,000 additional optimizer updates. Original full-pass continuation preserves learning rate 0.001.

1. **RUNNING:** `followup_mean_10hz_epoch1_ce`, continue original 2,000→4,775 updates, exactly 38,193 unique examples/one pass. Supervisor `followup10hz-fullpass`, launched 08:58 UTC; frozen base source `46ed242` plus captured launcher SHA.
2. **PLANNED:** full 10 Hz checkpoint held-out evaluation; fixed panels identical to prior study.
3. **IMPLEMENTING:** response-only CE continuation control, 2,000 additional updates.
4. **IMPLEMENTING:** 30% transcript reconstruction / 70% original assistant-response CE, matched updates.
5. **IMPLEMENTING:** ordinary response KL / emotional original CE, matched updates. Runtime/memory smoke determines feasible quota before expensive execution.
6. **IMPLEMENTING CPU:** tone classifier; GPU text baselines queue after training release.
7. **PLANNED:** final selected checkpoint and fully matched multi-turn evaluation.

Runtime choices and any failed operations will be recorded here. Test responses do not select a checkpoint; validation precedes final test comparison. Additional training beyond these controlled branches requires scientific justification and available time within the requested window.

## Outputs and completion

- Node operations/logs: `/workspace/followup10hz_20261007`.
- Node new results: `/workspace/speech-projector/results_followup_20261007`.
- Local backup: `results_preview/followup10hz_20261007`; older sealed replay remains immutable.
- [ ] All actual checkpoints, configs, source snapshots, optimizer states, metrics, outputs and failures backed up and hash-verified.
- [ ] Root reads representative outputs; independent case-level scoring retained.
- [ ] Main README contains final results, scale explanations, tables, diagrams, graphs, limitations and recommended configuration.
- [ ] Required tests, Ruff format/lint and coherent feature commits complete.
- [ ] Jobs quiescent, result inventories verified and final report delivered.
