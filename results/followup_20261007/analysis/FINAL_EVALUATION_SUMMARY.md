# Follow-up evaluation: final selected checkpoint and limits

The frozen validation rule selects **10 Hz response CE at 6775 updates**. Neither 30% transcript mixing nor ordinary-only teacher KL improved the matched validation content/CE tradeoff. The completed 9550 extension improves CE again but is **rejected** by the predeclared material-content safeguard: three new material errors versus one correction across the same24 validation cases. Selection used no TEST outputs. Both the auxiliary branches and rejected endpoint remain fully reported rather than hidden.

## Matched evaluations

The [core five-method table and figure](final_core_comparison/measured_comparison.md) compare the same256 held-out losses and136 generations. CE is token-weighted inside each cohort; macro CE gives the three cohorts equal weight. The older2.5 Hz and full-pass10 Hz references are descriptive endpoints, not equal-compute competitors to6775.

| Method | Ordinary CE,96 | Old emotional CE,80 | Neu CE,80 | Macro CE |
|---|---:|---:|---:|---:|
| Older speech2.5 Hz,4775 | 1.63412 | .85160 | .74219 | 1.07597 |
| Speech10 Hz,4775 | 1.62198 | .82798 | .73522 | 1.06173 |
| **Selected speech10 Hz,6775** | **1.61058** | **.76407** | **.64642** | **1.00702** |
| Actual words TEXT | 1.55649 | .73676 | .73896 | 1.01074 |
| Plain ASR | 1.56788 | .77696 | .77026 | 1.03837 |

TEXT/ASR omit delivery metadata and are words-only emotional comparators, while emotional teacher targets had privileged intended-tone metadata. The selected emotional CE being lower than TEXT therefore does not mean it has better literal understanding. The ordinary speech gap remains, and concrete free-generation errors include five PM becoming three PM, an already-working Wi-Fi connection becoming broken again, and reversed action/confirmation agency. All88 emotional generations end with EOS; ordinary48 has34 EOS and14 token-capped replies. Those caps cannot be counted as complete answers.

The [matched6775 branch table](descriptive6775_test_comparison/measured_comparison.md) keeps all three objectives together. Transcript mixing has ordinary1.61537 and macro1.00883; KL ordinary1.66679 and macro1.02391. KL raises whole-prefix teacher-token agreement to .8468 despite worse ordinary CE and material free-generation errors. Full-prefix agreement is conditioned on the teacher's prior target tokens and must not be called free-running accuracy.

The [rejected9550 endpoint](rejected9550_test_comparison/measured_comparison.md) has ordinary1.60525 and macro.99964, but ordinary EOS falls to28/48. Its separate TEST values are descriptive and did not overturn the earlier validation rejection. The [learning progression](learning_progression/progression.md) shows fixed38193 unique training examples, repeated exposure counts16000/38193/54193/76386 and the LR change after4775; it is not a data-scaling curve.

## Acoustic conditioning and tone-cue baseline

On40 Neu TEST pairs/22 design families, selected speech has raw matching margin .12021 [95% family CI .08154,.15368] and resized margin .10410 [.06982,.13490]. Strict assignment wins35/40=.875; one pair ties because its two targets are identical. This is target-preference evidence, **not emotion classification accuracy**. The wrong-audio control is linearly resized to hold pseudo-token count and changes feature statistics. Generations and factual errors remain essential evidence alongside it.

On the exactly matched64 Neu IDs/1802 target tokens, selected speech CE is .655528, plain ASR .779865, TEXT .746307, and ASR+predicted-tone .686550. The [exact64 table](final_neu64_matched/matched_metrics.md) prevents confusing these with the80-example Neu loss cohort. Predicted and oracle intended labels match on all64, so oracle outputs are explicitly reused with distinct provenance. The classifier's perfect intended-label TEST prediction on440 synthesized clips does not establish real human emotion perception.

## Locked model-assisted response ratings

The selected speech versus ASR review assessed **all88 emotional cases** in fresh randomized A/B slots, then locked ratings before revealing mappings or root scores. Slots were partially blinded; known ASR replies, broader context and shared rubric prevent claiming fully independent human annotation. The two0–2 axes are tone-appropriate helpfulness and literal grounding, not resemblance to teacher replies. Their equal-step means are heuristic ordinal summaries.

| Panel | Tone delta,95% family CI | Tone W/T/L | Grounding delta,95% CI | Grounding W/T/L |
|---|---|---|---|---|
| Evaluator all88 | +.250 [.128,.360] | 28/53/7 | −.284 [−.609,.054] | 18/35/35 |
| Evaluator Neu64 | +.3125 [.167,.458] | 23/37/4 | −.21875 [−.583,.167] | 12/27/25 |
| Evaluator old24 | +.0833 [−.278,.292] | 5/16/3 | −.4583 [−.938,.083] | 6/8/10 |
| Root locked24 subset | +.125 [−.125,.375] | 8/12/4 | −.1667 [−.4167,.125] | 2/16/6 |
| Evaluator same root24 | −.0833 [−.3333,.125] | 3/17/4 | −.4167 [−.8333,0] | 2/13/9 |

All88 contain15 tone improvements without grounding loss and13 with grounding loss. The all88 evaluator has18 prelocked ambiguity notes. Root/evaluator exact score agreement is33/48 tone slots and39/48 grounding slots; corresponding win/tie/loss direction agreement17/24 and21/24. Small-panel reviewer disagreement is visible, not resolved by changing ratings. The [full88 scored replies](quality/selected_speech_vs_asr/quality_report.md), [reviewer comparison](quality/selected_speech_vs_asr/reviewer_comparison.md) and [figure](quality/selected_speech_vs_asr/figures/quality_comparison.png) expose every score and reason. Grounding CI crossing zero does not demonstrate noninferiority.

The separate ASR+predicted-tone versus ASR locked64 review shows tone+.21875 [.060606,.382353], grounding−.09375 [−.346181,.171440]. Root's24 subset shows tone+.1667 with a CI crossing zero. These reviews use different comparisons and should not be converted into an indirect calibrated ranking against selected speech.

## Multi-turn findings

Every one of the132 merged replies plus36 separate initial-tone-ASR replies was read. The [full audit](final_selected_conversation_review.md) separates108 fully matched pipeline replies from24 speech-history ablations. Only three design families are represented. Selected speech gives no concrete relevant next step in any of the six branches per arm. Training-like current-audio/text-history closure improves to4/6; retained-audio closure remains0/6, and rollout has0/6 clear plus one borderline “proceed without further discussion.” Reused TEXT/ASR produce specific actions in2/6 controlled and4/6 rollouts, with their own factual caveats. There is no convincing useful delayed acoustic-tone retention in this panel.

## What should change next

Improve lexical and instruction grounding first: explicitly train negation, numbers, ownership/agency, already-completed state, and conversation-ending requests using checked targets. The tested30% transcript replacement does not establish that all lexical-alignment methods fail; it also reduces assistant-response training exposure. Retained acoustic-history layouts need matching multi-turn training before evaluating late emotion memory. Audit teacher role confusion and same-text tone plausibility, then verify audible delivery with listening before claiming natural emotion benefit. Current synthetic-label response-style gains should not obscure factual regressions or generic late replies.

All metrics derive from preserved actual records. Model/GPU/resource accounting and authoritative inventory verification are owned by the parent report; reused references and inherited training timers must not be added as new compute. No training, audio generation, inference or changes to sealed prior outputs were performed by this analysis.

The [same24 ordinary-ID fidelity panel](matched_fidelity24/summary.md) avoids comparing256 current examples against24 controls. Selected6775 CE is1.614342 on these24/5245 tokens, versus1.811697 for shuffled and1.888764 for zero audio. First-token agreement is.9167/.5000/.2083; first8 agreement.7760/.5260/.4219. Whole-prefix agreement.7872/.7455/.7907 shows why late-prefix agreement alone is misleading: zero audio is slightly higher than correct audio despite worse early agreement and CE.
