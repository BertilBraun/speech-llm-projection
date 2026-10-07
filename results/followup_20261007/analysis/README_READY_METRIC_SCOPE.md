# What the follow-up metrics can establish

The shared pool contains **38,193 training examples**, rather than40,000 training
examples:19,999 ordinary,9,094 old Qwen emotional and9,100 new Neu emotional.
Original held-out family assignments were preserved. Copy-time exclusions remove
one ordinary training dialogue with a held-out prompt overlap and three oversize
old emotional training pairs; original data and supervision are unchanged.
An example is a response-training pair, not a distinct base sentence: emotional
cohorts contain two delivery variants per base. Held-out design families are
distinct, while broad domains/intents/situations recur across splits.

The4775-update parent completes one pass of38,193 exposures. Matched branches
start from that exact projector, Adam state and cursor, then add2,000 updates /
16,000 exposures. All three branches use LR2e-4 after the parent's LR1e-3;
the rate changes after4775 updates. Response-only CE therefore reaches
54,193 cumulative exposures; transcript mixing replaces29.85625% of the added
examples with transcription targets, reducing response-token supervision.
The ordinary-only KL branch changes its objective rather than the audio/text
data or context. Compare their held-out **response CE**, not unlike-objective
training loss. A9550 continuation, if accepted, completes exactly two passes /
76,386 exposures; it is a different endpoint from the matched6775 comparison.

Pooled CE is token-weighted and dominated by longer ordinary replies. The
three-cohort macro gives each cohort equal weight; per-cohort CE and token
counts remain necessary. The fixed TEST loss set has256 examples; the fixed
generation panel has136:48 ordinary,24 old emotional and64 Neu. Exact64 Neu
tone-baseline comparisons must use those same64 loss IDs/tokens, rather than
silently compare with all80 Neu loss examples in the larger TEST cohort.

Teacher-prefix fidelity evaluates next-token choices while supplying the saved
teacher response prefix. Later tokens become easier because the correct answer
is already in that prefix. First-token and first-eight-token agreement are
therefore shown alongside full-prefix agreement. These are diagnostic measures
of conditional prediction, **not free-running correctness**: a reply can agree
with the opening tokens and then substitute a food, number, agent or event.
The saved Qwen teacher is a policy reference, not factual or emotional gold;
its own hallucinations and agency defects must not earn quality credit.

Emotional matching margins compare correct versus swapped audio against both
same-word teacher targets, canceling a general preference for one response.
Report raw and length-resized wrong-audio margins separately, with family
bootstrap intervals. Resizing removes pseudo-token-length cues but alters state
statistics. Strict pair-assignment wins count positive margins, with numerical
ties separately; they are not emotion-classification accuracy. The unrelated
0–2 direct-response rubric uses half-credit ties, so its win fraction is a
different estimand and must not be substituted for assignment wins.

ASR+predicted tone receives recognized words and a TRAIN-only classifier label.
On the fixed64 Neu TEST cases, labels equal the privileged intended labels, so
oracle replies are explicitly reused rather than regenerated. This is exact
intended synthesis-setting agreement, not validation of natural or perceived
emotion. Direct response ratings separately assess tone-sensitive helpfulness
and literal grounding, using individual-slot blinding and locked judgments;
two Codex reviewers with shared rubric/context are not independent human raters.
Ordinal mean differences are points on a heuristic0–2 scale, not percentage
accuracy. A confidence interval crossing zero does not establish noninferiority.
