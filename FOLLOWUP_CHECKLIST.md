# 10 Hz and lexical alignment follow-up — 7 October 2026

Authorized start: 08:51 UTC / 10:51 Berlin. Target report 13:51 UTC / 15:51 Berlin; maximum requested window 14:51 UTC / 16:51 Berlin. Root owns the single GPU queue and repository README. Model, data and evaluation subagents have separate implementation ownership.

Previous sealed results, older audio and supervision remain unchanged. Existing speech features are reused. No TTS, extra dataset generation, LoRA, encoder tuning or convolution search is part of this follow-up.

## Implementation and validation

- [x] Read complete node guide; GPU initially empty; node 42 GB and local approximately 14 GB free.
- [x] Snapshot and hash original 10 Hz optimizer/checkpoint; immutable continuation accepts only run name and cumulative update ceiling.
- [x] Validate continuation launcher on actual manifest and fixed validation panel; Ruff passes.
- [x] Typed optional transcript-reconstruction mixture, objective journal and exact resume tests (commits `ca7f8f4`, `9207a45`).
- [x] Ordinary-only response-distribution KL implemented; emotional examples retain original response CE. Actual frozen-gradient GPU gates passed, including the 1,260-token longest-target case.
- [x] Train-only CPU tone classifier and predictions, disjoint fixed validation/test families (commit `bd90c52`). CPU job completed in 22.72 seconds; validation balanced accuracy 99.57%, test 100%. These are synthetic intended-delivery labels, not natural emotion accuracy.
- [x] ASR plus predicted tone and reference-tone ceiling evaluated under otherwise identical prompting. Classifier prediction/report hashes are bound and oracle reuse proves identical inputs before avoiding duplicate inference.
- [x] Fully matched speech/TEXT/ASR multi-turn evaluation implemented and executed for the full-pass parent; final selected checkpoint remains pending.
- [x] Understandable factual diagnostics, blinded emotional cards/locked ratings and pre-results validation selection safeguards implemented. Tone-assisted ASR review is complete; final projector review remains pending.

## GPU queue and controlled runs

Only root may launch GPU work. Proposed objective branches share the full 10 Hz parent, seed, sample order, learning rate 0.0002, effective batch eight, and 2,000 additional optimizer updates. Original full-pass continuation preserves learning rate 0.001.

1. **COMPLETE:** `followup_mean_10hz_epoch1_ce`, continued original 2,000→4,775 updates, exactly 38,193 unique examples/one pass. Supervisor `followup10hz-fullpass`, launched 08:58 UTC, exited successfully 09:53:39 UTC; frozen base source `46ed242` plus captured launcher SHA. All 22 run files are backed up locally and hash-verified.
2. **COMPLETE:** full 10 Hz checkpoint held-out evaluation, fixed TEST256 / 136 generations. Pooled CE 1.482761, 121 completed / 15 capped replies. All 16 output files copied and verified against node hashes.
3. **COMPLETE:** response-only CE continuation control, 2,000 additional updates, exited successfully 10:45:02 UTC. All 21 run files are backed up and verified. Macro validation CE 0.997702; ordinary CE 1.588295; robust Neu paired margin +0.104584. All 24 fixed validation replies reviewed.
4. **COMPLETE:** 30% transcript reconstruction / 70% original assistant-response CE, matched updates; launched 10:48:25 UTC, exited successfully 11:28:29 UTC. All 22 files backed up and byte/hash verified. Macro validation CE 0.998907; ordinary CE 1.593912; robust Neu paired margin +0.104849. All 24 validation outputs reviewed: one material correction and one new material regression, below the predeclared veto.
5. **RUNNING:** ordinary response KL / emotional original CE, matched updates; launched 11:29:15 UTC after confirmed transcript exit0 and empty GPU process query. Actual gradient/memory gate already passed.
6. **COMPLETE:** tone classifier and ASR-plus-predicted-tone 64-response baseline. CE 0.686550; all 64 replies completed. Reference-tone reuse proves identical selected inputs and does not repeat inference. Both output trees are copied and node-hash verified. Independent 64-case and root 24-case blinded response ratings are locked and aggregated.
7. **PARENT COMPLETE / FINAL PLANNED:** fully matched multi-turn evaluation completed for the full-pass parent; 36 matched responses per speech/TEXT/ASR pipeline plus speech-history controls. All 15 files copied and hash-verified. Final selected branch must still be evaluated.

Runtime choices and any failed operations will be recorded here. Test responses do not select a checkpoint; validation precedes final test comparison. Additional training beyond these controlled branches requires scientific justification and available time within the requested window.

### Optional second-pass policy, fixed before branch outcomes

Root may extend only the validation-selected 6,775-update branch to 9,550 updates, keeping its objective, learning rate, seed and optimizer trajectory unchanged. Execute only if its macro and ordinary validation CE improve over the 4,775 parent, and measured incremental training time (with 15% allowance) predicts completion by 13:36 UTC, retaining at least 75 minutes before the six-hour report limit. Otherwise finish the planned evaluations and report without this extension.

The extended checkpoint becomes the final choice only if macro validation CE decreases, ordinary CE is no more than 0.01 above its 6,775 parent, and robust Neu matching margin is at least parent minus 0.02. All fixed 24 validation responses receive the same material-regression review; three new failures exceeding corrected failures veto the extension. Test responses never decide this choice. These thresholds are pragmatic safeguards, not calibrated meaningful-effect cutoffs. Extended versus 6,775 results have different update budgets and cannot establish a controlled objective effect.

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
- 10:41 UTC: optional strict objective-preserving second-pass continuation is implemented in `9e9d649` and prepared as an isolated archive/Git bundle. Its complete node CPU suite passed 569 tests, two infrastructure tests deselected, in 10.37 seconds with CUDA hidden. It has not been executed; the live training source remains `04eb486`. Existing trajectories and inference outputs remain immutable.
- 10:57 UTC: CE-control continuation and additional initial-tone ASR conversation evaluation completed with exit status zero, backed up and verified. Tone metadata improved controlled next-step specificity from 2/6 to 4/6, but rollout closure fell from 4/6 to 3/6 and useful delayed emotional adaptation was not demonstrated. Transcript branch is healthy; no other GPU process is running. These are small-panel diagnostics, not population accuracy estimates.
- 11:07 UTC: registered the three branch TEST jobs, all STOPPED with autostart disabled. They use isolated source `04eb486`, so later continuation deployment cannot silently change inference code. Added only Git metadata at that exact commit to the isolated archive; tracked source bytes remain clean. This prevents the conversation evaluator's post-generation Git provenance lookup from failing. No second GPU process launched.
- 11:29 UTC: transcript30 completed all 6,775 cumulative updates with exit0 and no other GPU process. Whole-run backup is underway; ordinary KL started at 11:29:15 UTC. Final 88-case review will use the predeclared comparison-specific shuffle `followup-selected-speech-v1`, avoiding reused positions from the decoded tone64 comparison. Reviewer context and recognisable baseline replies still prevent complete blinding. Selection thresholds remain unchanged.
- 11:38 UTC: actual node CPU preflight passed the conversation reference-reuse contract, binding all prompts, source/runtime/model revisions, cache/audio hashes and 132 response rows. Its four-file output is copied and verified; final selected speech replies remain pending. Old TEXT/ASR responses will not be regenerated.
- 11:48 UTC: ordinary KL is healthy at 5,587/6,775 updates; one GPU process, approximately 15.6 GiB process memory. Actual transcript mixture was 4,777/16,000 example exposures (29.86%) but 109,208/1,363,528 target tokens (8.01%). Its validation results are near the CE control, without clear lexical improvement. Optional second-pass preparation will use an immutable isolated source containing the validated continuation and selection types, preserving the active checkout and recorded numerical core.
