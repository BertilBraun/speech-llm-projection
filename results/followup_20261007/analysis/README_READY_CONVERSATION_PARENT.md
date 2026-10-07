# Existing parent conversation result: concise interpretation

The completed 10Hz parent fails these small delayed-response tasks more often
than the frozen word-based pipelines. In the matched three-scenario panel,
retained-audio speech provides a concrete next step in **0/6 controlled and 0/6
rollout** final-task cases, and explicitly acknowledges ending in **0/6 and 0/6**.
TEXT and plain ASR each provide concrete actions in **2/6 controlled and 4/6
rollout**, and explicit closure in **6/6 and 4/6**. ASR with a classifier-predicted
initial tone cue improves controlled concrete actions to **4/6**, with **4/6**
rollout actions; closure is **6/6 controlled and 3/6 rollout**.

These counts come from a model-assisted reading of all actual replies, not
calibrated accuracy. Each pipeline has 12 controlled replies plus 24 own-history
rollout replies; each individual task denominator 6 represents two intended
deliveries of only three base situations. All are EOS-complete. Lack of a new
question does not mean closure: speech frequently offers to continue or repeats
its prior reply after the user asks to end.

The scenarios concern an already-shared revised draft, attendance timing at an
arranged cafe, and an accurate/thorough spreadsheet. Speech repeats generic
offers instead of concrete actions. TEXT/ASR recover the draft and cafe better,
but all pipelines struggle to give a specific spreadsheet-related next action.
ASR + initial tone adds a controlled attendance check, yet its rollout also says
“whenever the usual cafe arrives,” confusing venue and people; the draft branch
sometimes proposes review before sharing despite the draft already being shared.

Retained-audio history was not trained. The separate training-like control with
current speech plus textual history also yields **0/6 concrete actions and 0/6
closures**, so the parent failure cannot all be blamed on the new history layout.
The ASR tone cue is placed only in the first USER message and carried in textual
history; it is not the single-turn SYSTEM-cue baseline. Neutral follow-ups have
no tone metadata.

Different late wording is insufficient evidence of useful emotional memory.
No specific clearly beneficial delayed emotional adjustment was established in
these three scenarios. The initial tones are synthetic intended labels, not
listened/perceived emotions. These are **parent 4775 results**; the final
validation-selected checkpoint must be reported separately, with the immutable
TEXT/ASR references reused only after exact inference/input identity verification.

Detailed evidence: `parent4775_conversation_review.md`,
`asr_tone_conversation_review.md`, `asr_tone_conversation_coverage.json`.
