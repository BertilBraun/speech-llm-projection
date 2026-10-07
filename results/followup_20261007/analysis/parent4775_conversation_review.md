# Full10Hz parent: matched multi-turn review

Read all108 matched records (36 each speech/trueTEXT/ASR) plus all12 current-audio
with textual-history speech controls and all12 omit-initial speech controls:
132 actual saved responses in total. All132 are EOS-complete; no cap is inferred
from text. No inference, waveform, prompt or inventoried result was modified.

Each pipeline has12 controlled delayed replies (turn3/4 for6 initial delivery
cases) and24 own-history rollout replies (4turns×6cases). There are only3 literal
base scenarios/families, so these counts are descriptive model-assisted checks,
not a broad benchmark or evidence about natural emotional memory.

## Predefined task interpretation

At turn3, a useful next step must specifically address the earlier situation
and propose a concrete action, rather than promise to help later or repeat the
previous reply. At turn4, explicit wrap-up acknowledgment is distinguished from
merely containing no question. A clear proposal to continue discussing/starting
steps contradicts ending. Vague “I'll keep it brief” is not counted as an explicit
closure. These are conservative audit criteria, not calibrated accuracy metrics.

| Pipeline / history arm | Controlled turn3 concrete useful step | Rollout turn3 concrete useful step | Controlled turn4 explicit wrap-up | Rollout turn4 explicit wrap-up | Controlled / rollout explicit current continuation |
|---|---:|---:|---:|---:|---:|
| Speech retained acoustic history | 0/6 | 0/6 | 0/6 | 0/6 | 2/6;3/6 |
| TrueTEXT at every user turn | 2/6 | 4/6 | 6/6 | 4/6 | 0/6;0/6 |
| ASR words at every user turn | 2/6 | 4/6 | 6/6 | 4/6 | 0/6;0/6 |
| Speech current turn + textual history (controlled only) | 0/6 | not run | 0/6 | not run | 6/6;not run |
| Speech current turn + omitted initial cue (controlled only) | 0/6 | not run | 0/6 | not run | 0/6;not run |

All turn4 replies contain no new question, including the speech failures. That
surface check alone therefore masks the failure to honor ending the conversation.
Both text pipelines have2 draft-rollout endings that merely promise brief/focused
conversation; these are not credited with explicit wrap-up either.

## Per-family literal grounding and useful action

**neu_base_04512: changed/shared draft after colleague notes, happy/sad.**
Retained speech controlled turn3 is exactly “I'm here to help you figure that out.”
for both deliveries. Own-history happy rollout repeats its opening project
enthusiasm; sad rollout repeats “...keeping the details from your colleague's notes...”
from turn2. Neither gives a specific next action. TEXT and ASR explicitly recover
the shared draft and colleague feedback. Controlled ASR proposes waiting for
colleagues' feedback; TEXT proposes review with the colleague, though its “before
sending it out” wording is mildly awkward after the draft was already shared.
Both rollout baselines propose checking alignment before finalizing. Speech
text-history control still gives only “I'm ready to help you move forward...”.
Its ending says “...move forward with the next step”, directly opposing closure.

**neu_base_00356: cafe arranged, uncertain whether everyone arrives on time,
happy/sad.** Speech controlled turn3 merely offers to determine the next step.
Happy rollout repeats a promise to include details in the next update; sad rollout
offers generic help. Earlier sad rollout even says “the usual cafe not arriving
on time”, substituting the venue for the people. Both word-based controlled
turn3 replies fail too: “I'll keep it brief.” Their own-history rollouts improve:
TEXT suggests checking whether anyone arrived; ASR names staff/contact and cafe
attendance. The relevant proposed action is present in both. Speech current-audio
with textual-history control again merely promises brevity, then proposes moving
forward together after the user explicitly asks to end.

**neu_base_04315: shared spreadsheet already thorough/accurate, angry/sad.**
Retained speech controlled turn3 is generic; sad branch even asks which previous
steps to explore. No actual spreadsheet-related next action is proposed. Sad
rollout repeats that prior details were clear/accurate; angry rollout offers help
when the user supplies more details. TEXT controlled turn3 says “I'll keep it brief”;
ASR says “I'll help you outline the next steps”. Both own-history rollouts promise
a simple next step but do not give one. Hence none receives concrete-action credit
for this family. Both word-based first-turn replies also narrate how they will
respond rather than simply respond; this defect belongs to the frozen text LLM
and inherited policy, not exclusively the projector.

## Acoustic-cue interpretation

In retained-audio controlled turn3/4,3/6 same-base tone comparisons have exactly
identical replies; the other3 differ in generic wording rather than a specific
useful emotion-sensitive action. The textual-history speech control yields6/6
identical paired replies, as expected from identical lexical histories/current
follow-up audio without the initial acoustic cue. TEXT/ASR controlled pairs are
also6/6 identical and their rollout pairs12/12 identical; those pipelines received
identical words and no tone labels. This is expected for words-only baselines,
not proof that they could recover tone.

The retained-acoustic-history layout was not trained; it is an inference-only
out-of-distribution test. Crucially, the training-like current-audio/text-history
arm also fails concrete next steps and endings, so poor results cannot all be
attributed to that layout change. There is no convincing useful delayed-tone
benefit from these3 cases. More varied wording or a brief reply does not establish
cue carryover. The forthcoming predicted-initial-tone ASR baseline is separate,
and the final selected branch must be audited separately from this earlier parent.
