# Emotional benefit and multi-turn evaluation

Added 6 October 2026 following the user's request. This is an evaluation protocol,
not a completed experiment. Projector training, LoRA, additional synthesis and serving
deployment remain paused. The completed 5,000-utterance / 10,000-clip dataset and its
sealed backup are unchanged.

## Research questions

1. Does hearing emotional delivery improve the appropriateness of the response beyond
   what the same words alone provide, while preserving factual and instruction accuracy?
2. Does a projector trained only on single-turn emotional examples support coherent
   multi-turn conversations without multi-turn fine-tuning?
3. Does the benefit survive when earlier user turns are retained as speech embeddings,
   rather than supplied as privileged transcripts?

Single-turn training makes multi-turn generalization plausible because Qwen already
understands chat histories. It does not establish that Qwen can interpret several
projected utterances in that history: that is an untested input distribution. The earlier
20k checkpoint also used bounded textual history; it is not evidence for an emotional
student trained without history or for retained speech history.

## Emotion helps when a response becomes more appropriate

Use the same-text delivery pairs in the existing held-out family splits. Compare:

| System | Current user input | Purpose |
| --- | --- | --- |
| Emotional student | Projected audio, no transcript or tone annotation | Candidate system |
| Pre-emotional projector | Identical projected-audio path | Benefit of emotional training |
| Transcript Qwen | Exact words, no tone annotation | Words-only control |
| ASR Qwen | Recognized words, no tone annotation | Conventional speech pipeline |
| Annotated-text Qwen | Exact words plus intended delivery | Privileged reference policy |

Use the same response-style instruction and decoding budget wherever practical. The
existing checkpoint's no-system-prompt policy differs from the proposed concise student
policy: include prompt-matched checkpoint comparisons and label prompt changes so that
improvements are not attributed to emotional training alone. The annotated-text reference
sees intended synthesis labels, not independently verified emotions or the actual audio.
Keep its metadata privilege explicit.

The primary benefit measure is the paired change in acceptable-response rate relative
to transcript Qwen, with separate scores for content accuracy, tone appropriateness,
grounding and naturalness. Report absolute rates, percentage-point differences and
family-cluster confidence intervals. Preserve results for tone-neutral cases where
both replies are appropriate; require neither different wording nor an explicit emotion
word. A sympathetic reply that invents events or ignores the request is a failure.

Use the existing blind judge rubric and same-text four-way teacher-preference margin
from `artifacts/emotional_evaluation_plan.md` as complementary diagnostics. Calibrate
the judge with human-reviewed examples. Judge quality without showing teacher targets;
teacher replication is a separate metric. Listen to a balanced pilot and record perceived
delivery and uncertainty before interpreting a tone-label score as acoustic understanding.
Report performance against intended labels and against independently reviewed perceived
delivery separately; do not alter the canonical training records.

## Coherent multi-turn fixtures

Start with 24 held-out scenarios, four user turns each, and two plausible delivery
branches at one critical turn: 48 trajectories / 192 user-turn responses per system.
This is a small diagnostic suite, not a precise population estimate. Cover factual
recall, corrections, pronoun references, requests to stop, clarification and changes
in emotional delivery across a conversation. Include neutral continuations so that a
model is not rewarded for treating all later speech as persistently distressed.

Author each dialogue as a coherent scenario with explicit facts and constraints, a
turn script, the critical same-text tone pair and a grading rubric. Keep scenarios
independent of training families and inspect overlap. Reuse existing held-out clips
only where their exact words fit the scenario; unrelated concatenation is at most a
separately labelled robustness check. Do not turn arbitrary dataset targets into prior
assistant messages or assume they form a coherent conversation.

Example scenario:

1. User establishes that a meeting is Friday at three and that no email should be sent.
2. User corrects the time to four, retaining Friday and the no-email instruction.
3. User says "I finally got an update about the meeting arrangements" with two plausible
   deliveries, such as happy and frustrated. A good response respects delivery without
   inventing what the update said.
4. User asks "What time did we settle on, and what should you avoid doing?"

The final turn has concrete checks: Friday at four, and no email. The critical turn has
a tone-appropriateness rubric rather than one mandatory response string. These are
proposed fixtures; no new clips or targets have been generated for them.

## Isolate current-tone sensitivity, then test full rollouts

First evaluate each critical turn with an identical, fixed preceding history. Swap only
its same-text audio pair and score the response. This isolates the effect of current
delivery; a difference between full trajectories could otherwise come from different
earlier assistant answers.

Then run the complete four-turn scripts with the model's own assistant responses carried
forward. Report these as scripted rollouts: future user turns are fixed, not an adaptive
human conversation. Inspect whether generated replies make later scripted turns unnatural,
and record those cases instead of silently rewriting the scripts for one system.

Evaluate three history conditions with the same single-turn-trained student:

| History condition | Earlier user turns | Earlier assistant turns |
| --- | --- | --- |
| No history control | Omitted | Omitted |
| Text history | Exact transcripts, then a separate ASR-history variant | Actual generated text |
| Speech history | Previously finalized projected embeddings | Actual generated text |

Text history measures a practical hybrid path but provides earlier words explicitly and
loses their delivery. Speech history tests the audio-only conversation goal and preserves
earlier acoustic information. Gold transcripts in the diagnostic condition are privileged;
never insert the current transcript or tone label into the speech candidate. Keep the
annotated-text reference separate from these student conditions.

For speech history, put each retained embedding sequence inside its original user role
delimiters, followed by its generated assistant message. Rebuild the complete fixed prompt
per turn initially; use the runtime's normal decoding cache within that turn. Persistent
cross-turn cache reuse is a separate serving optimization, not needed to answer the
generalization question. Treat this assembly path as an unvalidated extension requiring
prompt/mask/position checks before model evaluation.

Keep four turns within an explicit context budget, record speech and text token counts,
and do not silently apply the old two-message / 256-token history truncation. All new
evaluation clips must fit the current 30-second audio limit. Preserve the three existing
over-limit training clips and decide their training handling explicitly; do not truncate
them without recording that decision.

## Outputs and acceptance criteria

Save the exact scenario, audio IDs/hashes, intended and reviewed delivery labels,
history condition, prompt policy, decoding policy, every generated turn, judge evidence,
termination status and latency. For annotated references, save every annotation supplied.

Report turn-level content and tone scores, factual/constraint check accuracy, acceptable
response rate by turn number and all-turns-acceptable conversation rate. Report correction
retention, ignored boundaries, invented facts, inappropriate persistence of earlier mood
and premature termination explicitly. Bootstrap the complete scenario with both delivery
branches and all systems together; 24 scenarios imply wide uncertainty. Do not treat 192
turns as independent samples.

The desired result is improved emotional appropriateness with preserved content accuracy,
plus retained facts and constraints across turns. Select tolerable content degradation and
judge thresholds before inspecting student results. If speech-history performance fails
while text history succeeds, document that limitation; do not claim general multi-turn
success or automatically start multi-turn training or LoRA.
