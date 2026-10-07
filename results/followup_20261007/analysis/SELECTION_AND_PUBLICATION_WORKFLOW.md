# Follow-up CPU selection, locked TEST review and publication workflow

Live node experiments retain the root's frozen GPU source. The selection,
rating/report and publication steps below run locally on authoritative completed
copies. They do not modify an inventoried run, cached baseline or older sealed
replay. Root owns final copying, README editing and package sealing.

## Exact validation selection

Canonical CLI: `python -m scripts.select_followup_branch --config <JSON>`.
`ValidationSelectionConfig` composes the pre-results `AlignmentSelectionPolicy`
file, CE-control run directory, auxiliary run/review paths and a new output
directory. Each `AlignmentBranchReview` contains actual control and auxiliary
generation `FileArtifact`s and all **24 ordered ValidationCaseReview** entries.
Each entry explicitly explains the current finding, with zero or more canonical
`ValidationContentChange` records. Empty changes means the case was inspected
without a material change; a missing case is rejected.

The driver verifies identical primary IDs, words, histories and target references;
actual generation hashes/bytes; and run/case identities of every recorded change.
It then invokes the existing `select_alignment_branch` without modifying any
threshold. Output `decision.json` is the canonical `AlignmentSelectionDecision`,
with `provenance.json` binding policy, completed result/candidate files and full
reviews; `decision.md` exposes metrics and every flagged correction/regression.

Frozen rule at 6775 updates: ordinary CE ≤ CE-control +0.05; robust Neu matching
margin min(raw, resized) ≥ CE-control −0.02. Among eligible runs within 0.03 of
best macro CE, choose ordinary CE, then macro CE, then name. Qualitative veto
requires at least three new material failures **and more regressions than clear
corrections** across the full 24-case panel. Partial/generic responses, lexical
omissions alone, tone-label ambiguity and teacher-resemblance differences are not
automatically material contradictions. Every decision is VAL-only. TEST is never
used to select or to retroactively alter the rule.

## Optional second complete pass

The pre-results policy is saved in `artifacts/followup10hz/extension_policy.json`.
`python -m scripts.select_followup_extension --config <JSON>` consumes an actual
9550-update continuation of the selected 6775 branch and a full same-ID 24-case
validation review against that selected parent. It requires strict macro CE
improvement, ordinary CE ≤ parent +0.01, robust Neu margin ≥ parent −0.02, and
the same material-content veto (at least three regressions, exceeding corrections).
The typed accepted/rejected decision retains the immutable 6775 choice, both
actual result/candidate hashes and saved continuation provenance. Objective, LR,
data trajectory and configuration remain unchanged except run name and endpoint.
This policy was written before any extension outcome; execution is time-dependent.
No 9550 outcome or acceptance is assumed.

## Selected speech versus ASR: 88 delivery cases

After the selected checkpoint's actual TEST copy is complete, run:

```
python artifacts/followup10hz/prepare_selected_review.py --decision <decision.json> --run-result <selected result.json> --evaluation <actual evaluation directory>
```

This checks the selected run/endpoint and evaluation input hash before preparing
the cards. For an actually accepted 9550 extension, supply
`--extension-decision <accepted extension decision.json>` in addition to the
unchanged 6775 branch decision. A rejected extension cannot enter this review;
the typed final checkpoint choice is persisted beside its quality configuration.
For the 6775 branch, the selection provenance must bind the actual result hash.
Both paths then prepare
the existing canonical quality comparison: 24 old Qwen +64 Neu cases, complete
pairs, seed-42 slot flips domain-separated by the predeclared salt
`followup-selected-speech-v1`. Empty salt preserves historical flips exactly.
Output is
`analysis/quality/selected_speech_vs_asr`; root's 24 cards are an eight-old /16-Neu
family-balanced subset selected before ratings. Cards expose actual user words,
history, intended delivery and A/B replies, but no reference target, condition
mapping, recognized transcript or method name. Wider experiment context is known.
This is individual-slot blinding, not complete reviewer blinding.

Root and evaluator independently produce canonical `CaseRating` rows with the
pre-fixed 0–2 tone/helpfulness and grounding axes, brief reasons, confidence and
ambiguity notes. Freeze exactly once with
`python -m scripts.report_followup_quality --directory <cards directory> --ratings <working JSONL> --freeze <locked JSONL>`.
Each reviewer locks all assigned cases before viewing source mapping or the other
reviewer's ratings. No ratings are revised after decoding. The 64-case earlier
ASR+predicted-tone review remains immutable and is analyzed separately.

Then aggregate using the same CLI without `--freeze`; report all cases, both
ordinal means/deltas, W/T/L, half-credit ties, 95% family bootstrap intervals
(2000 draws, seed 42), and tone gains with versus without grounding loss. Compare
root24 and evaluator24 score/direction agreement separately from full evaluator88.
No result is named human accuracy, calibrated acceptability, natural emotion
perception or teacher-fidelity success. Intended synthetic labels were not
validated by listening; equal-step ordinal means are heuristic.

Plot locked summaries with
`python -m scripts.plot_followup_quality --report <quality_summary.json> --output <new figures directory> --first-label '<selected speech label>' --second-label ASR`.
Methods are categorical, with explicit n, direction and family-bootstrap CIs.

## Curated publication manifest: results/followup_20261007

The following is the planned small public artifact set. Do not publish pending
experiments as complete. Copy only after final measured reports/reviews are done;
retain provenance hashes/full technical IDs in machine files. Original models,
WAVs, feature cache, optimizer states and journals remain in the complete result
backup rather than this small repository entry point.

| Published destination | Actual source / owner | Purpose |
|---|---|---|
| `selection/decision.json`, `decision.md`, `provenance.json` | New local `analysis/selection` / evaluator | Fixed VAL decision, all input hashes and change reasons |
| `selection/policy.json` | Existing pre-results policy / root | Preserves exact heuristic thresholds |
| `selection/extension_decision.json`, `.md`, `extension_policy.json` (only if executed) | Actual optional second-pass evidence / evaluator | Separate accepted/rejected9550 assessment, immutable6775 choice retained |
| `figures/validation_branches.png`, `.pdf`, `.csv` | `plot_followup_validation` from actual6775 decision / evaluator | All three matched validation objectives, exact endpoint and frozen rule; not TEST selection |
| `metrics/measured_comparison.json`, `cohort_metrics.csv`, `measured_comparison.md` | Final `followup_report` output / evaluator | Same TEST256/cohort estimands for old2.5Hz, parent10Hz, final selected, TEXT and ASR; shared old baselines |
| `figures/heldout_comparison.png`, `.pdf` | Final measured report / evaluator | Categorical CE/reference proxy plots and actual Neu paired-margin CIs |
| `metrics/neu64_matched_metrics.json`, `.md` | Exact64 `followup_matched_metrics` final report / evaluator | Identical-ID/1802-token comparison of speech/TEXT/ASR/predicted tone/oracle reuse |
| `quality/tone64/quality_summary.json`, `reviewer_comparison.json`, `.md` | Existing locked tone64 analysis / evaluator | Tone-cue effect, root24 agreement, limitations and lock hashes |
| `quality/selected88/quality_summary.json`, `quality_report.md` and reviewer agreement | Completed selected speech88 analysis / evaluator | Direct emotional helpfulness/grounding benefits independent of teacher resemblance |
| `quality/tone64/quality_comparison.csv`, `.png`, `.pdf` | Existing canonical quality plot CLI / evaluator | Paired ordinal deltas/W-T-L/95% CIs |
| `quality/selected88/quality_comparison.csv`, `.png`, `.pdf` | Final canonical quality plot CLI / evaluator | Same reporting for selected speech vs ASR |
| `reviews/validation_control.md`, auxiliary reviews | Actual all24 validation audits / evaluator | Literal errors, corrections, context and caps; no TEST selection |
| `reviews/conversation_parent.md`, `conversation_asr_tone.md`, selected speech review | Actual matched36 pipelines plus training-like control / evaluator | Topic/action recovery, closing compliance and uncertain late-cue carryover |
| `provenance/tone_classifier_report.json`, prediction identity/reuse receipts | Existing backed-up classifier/tone inference / root/data | TRAIN-only classifier and exact oracle-label equality on fixed64 |
| `provenance/resource_ledger.json`, `.md`, dataset/source receipt | Final root/data report | Deduplicated parent counters versus new work and scope limitations |
| `PUBLICATION_MANIFEST.json` | Root final copy helper | Final FileArtifact hashes of copied public files, after writers stop |

Root's README links this small tree and the complete off-node verified backup.
Actual full saved replies remain available: they are not replaced by cherry-picked
snippets or concealed scoring uncertainty. Optional second-pass continuation is
a separate endpoint under the already selected objective, not a new TEST-driven
branch-selection contest; if executed, its actual provenance and comparisons
must be added explicitly rather than silently replacing 6775 artifacts.

The ignored operational helper `prepare_final_report_configs.py` writes the
canonical five-method `final_core_comparison.json`, identical-panel
`final_neu64_matched.json` and conservative `content_final_core.json` configs
only after actual validation selection and TEST result provenance are available.
It does not create outcome placeholders, ratings or model outputs. Execute the
three existing CPU report CLIs with those configs; inspect rendered PNGs before
root's curated copy. Literal-screen flags are review prompts, not claimed factual
accuracy: omitted repetition or different surface negation can be appropriate.
