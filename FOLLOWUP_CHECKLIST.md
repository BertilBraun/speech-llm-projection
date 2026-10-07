# 10 Hz and lexical alignment follow-up — 7 October 2026

Authorized start: 08:51 UTC / 10:51 Berlin. Target report 13:51 UTC / 15:51 Berlin; maximum requested window 14:51 UTC / 16:51 Berlin. Root owns the single GPU queue and repository README. Model, data and evaluation subagents have separate implementation ownership.

Previous sealed results, older audio and supervision remain unchanged. Existing speech features are reused. No TTS, extra dataset generation, LoRA, encoder tuning or convolution search is part of this follow-up.

## Implementation and validation

- [x] Read complete node guide; GPU initially empty; node 42 GB and local approximately 14 GB free.
- [x] Snapshot and hash original 10 Hz optimizer/checkpoint; immutable continuation accepts only run name and cumulative update ceiling.
- [x] Validate continuation launcher on actual manifest and fixed validation panel; Ruff passes.
- [x] Typed optional transcript-reconstruction mixture, objective journal and exact resume tests (commits `ca7f8f4`, `9207a45`).
- [x] Ordinary-only response-distribution KL implemented; emotional examples retain original response CE. Frozen-gradient GPU verification is pending GPU release.
- [x] Train-only CPU tone classifier and predictions, disjoint fixed validation/test families (commit `bd90c52`). CPU job completed in 22.72 seconds; validation balanced accuracy 99.57%, test 100%. These are synthetic intended-delivery labels, not natural emotion accuracy.
- [x] ASR plus predicted tone and reference-tone ceiling implemented under otherwise identical prompting; inference pending GPU release. Classifier prediction/report hashes are bound and oracle reuse proves identical inputs before avoiding duplicate inference.
- [x] Fully matched speech/TEXT/ASR multi-turn evaluation implemented; generation pending GPU release.
- [x] Understandable factual diagnostics, blinded emotional cards/locked ratings and pre-results validation selection safeguards implemented; scoring pending outputs.

## GPU queue and controlled runs

Only root may launch GPU work. Proposed objective branches share the full 10 Hz parent, seed, sample order, learning rate 0.0002, effective batch eight, and 2,000 additional optimizer updates. Original full-pass continuation preserves learning rate 0.001.

1. **COMPLETE:** `followup_mean_10hz_epoch1_ce`, continued original 2,000→4,775 updates, exactly 38,193 unique examples/one pass. Supervisor `followup10hz-fullpass`, launched 08:58 UTC, exited successfully 09:53:39 UTC; frozen base source `46ed242` plus captured launcher SHA. Independent immutable capture of all 22 run files succeeded; off-node transfer underway.
2. **COMPLETE:** full 10 Hz checkpoint held-out evaluation, fixed TEST256 / 136 generations. Pooled CE 1.482761, 121 completed / 15 capped replies. All 16 output files copied and verified against node hashes.
3. **RUNNING:** response-only CE continuation control, 2,000 additional updates; launched 10:05 UTC after every preceding GPU job exited successfully.
4. **IMPLEMENTED:** 30% transcript reconstruction / 70% original assistant-response CE, matched updates; deterministic task choice and objective journal.
5. **IMPLEMENTED:** ordinary response KL / emotional original CE, matched updates. Runtime/memory smoke determines feasible quota before expensive execution.
6. **COMPLETE:** tone classifier and ASR-plus-predicted-tone 64-response baseline. CE 0.686550; all 64 replies completed. Reference-tone reuse proves identical selected inputs and does not repeat inference. Both output trees are copied and node-hash verified. Independent 64-case and root 24-case blinded response ratings are locked; aggregation pending.
7. **PARENT COMPLETE / FINAL PLANNED:** fully matched multi-turn evaluation completed for the full-pass parent; 36 matched responses per speech/TEXT/ASR pipeline plus speech-history controls. All 15 files copied and hash-verified. Final selected branch must still be evaluated.

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

## Validation and progress notes

- 09:07 UTC: full-pass GPU job at 2,418/4,775 updates, healthy; only one GPU process. CPU classifier runs with CUDA hidden and two CPU threads.
- 09:12 UTC: local `uv run pytest -m 'not integration'` passed 536 tests; two infrastructure tests deselected. New objective and branch feature Ruff checks pass. Node source remains the original `46ed242` while the full-pass job is active; new core source will deploy only after completion.
- 09:29 UTC: isolated node candidate source `54b52ef` passed 542 tests, two infrastructure tests deselected, in 9.40 seconds with CUDA hidden. Exact source archive and Git bundle are prepared; the live training checkout is still `46ed242`.
- 09:33 UTC: prepared node evaluation configs and registered stopped single-GPU jobs for first full-pass evaluation, tone-assisted ASR, matched conversation evaluation and the three controlled training branches. Frozen selection safeguards require at least three new material validation failures exceeding corrected failures before a qualitative veto. Test responses do not choose the branch.
- 09:55 UTC: full pass completed with daemon exit status zero. Deployed frozen GPU source `04eb486` by fast-forward after preserving the two untracked classifier source copies. This source passed 544 isolated node CPU tests and exact held-out selection checks; ordinary-KL gradient/memory smoke is running before long branches. Unrelated scheduler services are preserved.
- 09:57 UTC: ordinary-KL GPU gate passed all three actual examples, including 1,260 target tokens. Positive projector gradients/updates; all 1,881,825,088 frozen LLM parameters unchanged and without gradients. Peak allocated memory 6.513 GB.
- 10:06 UTC: first checkpoint, tone baseline and matched conversation evaluations are complete and backed up. Early qualitative inspection finds persistent ingredient/time/agency errors and weak later-turn instruction following; these are documented rather than inferred from CE alone. Controlled branches continue, with validation-only selection still pending.
- 10:26 UTC: independent tone64 and root24 ratings remain locked. Tone-aware ASR versus plain ASR: +0.21875 tone points [95% family CI 0.06061, 0.38235], grounding −0.09375 [−0.34618, 0.17144]; grounding noninferiority is not established. Exact Neu64 loss comparison uses 1,802 identical target tokens: speech parent 0.745869, true text 0.746307, ASR 0.779865, ASR+tone 0.686550.
- 10:26 UTC: additional ASR-plus-historical-initial-tone multi-turn job is implemented and STOPPED, ready between training jobs. Isolated inference source `e54d5a4`; actual CPU preflight validates all six held-out inputs and prediction hashes. Only the first user message carries tone metadata; later neutral messages do not. Core LLM/training source is unchanged versus `04eb486`, and the active training checkout remains untouched.
