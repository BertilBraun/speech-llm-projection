# Matched6775 branch result and frozen validation choice

All24 ordinary-KL primary replies and eight ordinary histories were read against
the identical CE-control panel. Actual words/history/reference IDs and generation
hashes were verified; full replies are in `control6775_vs_ordinarykl_replies.md`
and the canonical review is
`selection/reviews/followup_mean_10hz_ordinarykl_6775.json`.

| Endpoint6775 | Ordinary48 CE | Old40 CE | Neu40 CE | Macro CE | Robust Neu margin | EOS / cap |
|---|---:|---:|---:|---:|---:|---:|
| CE control | 1.5882947180 | 0.7655269874 | 0.6392832394 | 0.9977016483 | 0.1045839522 | 22 / 2 |
| Transcript30 | 1.5939120854 | 0.7604249582 | 0.6423853865 | 0.9989074767 | 0.1048491370 | 24 / 0 |
| Ordinary teacher KL | 1.6506066309 | 0.7558085783 | 0.6435695690 | 1.0166615927 | 0.1052255820 | 22 / 2 |

The ordinary-KL branch exceeds the frozen ordinary-CE tolerance by **+0.06231191**
versus control, beyond allowed+0.05, so it is ineligible. This is an existing
heuristic safeguard, not a newly tuned significance test. It has the best old
emotional CE of the three and slightly higher Neu paired margin; these do not
override the predeclared ordinary-content guard.

KL Neu raw paired margin is +0.12249361, 95% family-bootstrap interval
[0.06628527,0.18794075]; resized +0.10522558
[0.05978651,0.15825565]. Both use20 bases /20 families. Strict raw matching wins
are17/20, interval[70%,100%], with no numerical ties. Old emotional raw margin is
+0.03132156 [0.00442550,0.06124649], resized+0.05258349
[0.02567105,0.08210201]. These teacher-forced margins describe reference-target
assignment/input sensitivity, not emotion accuracy or response appropriateness.
Resizing wrong-audio states changes their statistics; intended labels are
synthetic settings rather than verified audible emotion.

Concrete new KL errors are explicit: `qwen:utterance_00766_frustrated` changes
the user's **90%** score to **95%**, and `ordinary:cf5f5a90ab8079a0bbc4` invents
Darcy's sister's death while still failing to answer the adaptation question.
They are coded as two new material regressions, with no clear correction of a
material failure. The separate qualitative veto requires at least three new
failures exceeding corrections, so that veto itself does not trigger.

Both long ordinary explanation cases remain capped. The flipped-classroom core
is correct but ends mid-heading; the adaptation reply remains off-task.
Wrong oat-milk/cheese ingredients are byte-identical to CE control. Drawing
retrieval and appointment-confirmation agency errors persist. A happy camera
reply usefully suggests double-checking, but the control already avoided a
material technical diagnosis, so this practical improvement is not mislabeled
as correction of a material control error. Hamster and title precision remain
weak. Every remaining case is documented in the full24 review.

The canonical frozen6775 selector now chooses **CE control**:
`selection/decision.json`, `provenance.json` and `decision.md`. Control and
transcript mixing both satisfy numerical guards and the0.03 macro band; control
wins the predeclared ordinary-CE tie-break. KL is excluded by the ordinary guard,
not because its macro lies outside0.03 (it does not). Both auxiliary content
audits and every correction/regression are hash-bound in the receipt. TEST has
not been consulted for this choice. Any optional9550 continuation is a separate
validation acceptance decision; it cannot silently replace the immutable6775
choice or claim an outcome before it exists.

This is one-seed evidence. Lower saved-target CE is not a complete quality
metric, particularly with sampled teacher replies and existing teacher
perspective/grounding defects. No broad conclusion that all teacher KL is harmful
follows from one ordinary-only configuration and2000 added updates. The final
direct speech-versus-ASR rating and matched conversation audit remain necessary.
