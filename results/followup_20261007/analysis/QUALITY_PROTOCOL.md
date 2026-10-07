# Pre-results model-assisted response review

Fixed comparisons: 64 Neu emotional responses for ASR+predicted tone versus plain ASR;
after validation-only branch selection, 88 emotional responses (64 Neu +24 old Qwen)
for selected speech versus the same plain ASR. No source examples, teacher replies,
old locked ratings, or sealed replay will be edited. First4775 parent outputs are
descriptive and are not a matched6775 branch contender.

Each card contains the true words, supplied history, intended synthetic delivery,
and deterministically flipped A/B replies. It omits teacher replies, actual ASR
transcripts, model names, source mapping, and file paths. SHA(seed42:exampleID)
fixes slots. Root's 24-case subset retains complete pairs and is selected by
family/contrast before response ratings. Root and evaluator lock their own scores
before reading each other's scores or decoding the source map. Broader experiment
context is known, so this is individual-slot blinding, not complete blinding.

Tone/helpfulness: 0 means incompatible, dismissive, or emotionally unhelpful;
1 means appropriate but generic; 2 means specifically helpful and sensitive to
the annotated delivery without unsupported mental diagnosis. Grounding: 0 means
a material factual reversal, missed request, or invented agency; 1 means generic,
partial, or mildly unsupported; 2 means faithful and useful. Axes remain separate:
invented facts primarily reduce grounding, not automatically tone. An empathic
reply can score tone2/grounding0. No score rewards resemblance to the teacher.

Ties are allowed. Each slot receives a brief reason; each case receives confidence
and any ambiguity note. Literal words and intended labels can fit imperfectly,
especially old sarcasm/frustration labels. We will neither relabel cases nor infer
that the supplied emotional labels were actually audible. No clips have been
listened to for these judgments. Scores are ordinal model-assisted judgments,
not human annotations, calibrated acceptability, or natural emotion accuracy.

Report all assessed cases, each axis's means and wins/ties/losses, half-credit
ties, and 95% family-bootstrap intervals (2000 draws, seed42). Shared explicit
design-family IDs are clusters, including shared old/new families. Separately
report tone gains without grounding loss versus tone gains with grounding loss.
Equal-step means are heuristic rather than validated psychometrics. Main results
retain all fixed cases; any ambiguity/confidence exclusion is explicitly posthoc.

The branch decision is made before TEST review: at6775, ordinary CE no more than
CE-control+.05 and robust Neu raw/resized margin at least CE-control−.02. Eligible
branches within.03 of the best macro CE are ordered by ordinary CE, macro CE,
then name. A qualitative veto requires at least3 new material failures on all24
fixed VAL cases and more regressions than clearly corrected control failures.
All case IDs/reasons are retained. Thresholds are pragmatic safeguards, not
calibrated meaningful-effect cutoffs. TEST ratings never select the branch.

## Pre-selected-TEST blinding amendment

Before any selected88 cards or ratings were produced, slot randomization for that
new comparison was domain-separated with the explicit salt
`followup-selected-speech-v1`. It hashes `salt:42:exampleID` rather than the already
decoded prior comparison's `42:exampleID`. This addresses accidental reuse of
known label positions; it does not change the selected cases, root24 family/pair
sampling seed42, selection thresholds or rating rubric. The configuration records
the salt. Empty salt preserves all old flips exactly, and existing locked tone64
and historical ratings are untouched. Known ASR replies and wider experiment
context can still reveal likely sources: neither full blinding nor independent
human annotation is claimed.
