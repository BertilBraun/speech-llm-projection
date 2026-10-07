# ASR + predicted initial tone: 36-response conversation audit

Read all 36 actual saved replies: 12 controlled turn-3/4 replies and 24 four-turn
rollout replies, for six initial delivery cases in three base scenarios. The
canonical completion records show 36 EOS-complete, zero token-limited; summed
generation time is 44.97175056 seconds. This is a descriptive model-assisted
audit, not human annotation or a calibrated accuracy estimate.

The CPU helper `artifacts/followup10hz/audit_tone_conversation.py` checked exact
36-ID coverage against the earlier TEXT and ASR pipelines, unchanged neutral
follow-up ASR words, and the initial predicted annotation against the hash-bound
classifier provenance. Evidence hashes and canonical counts are in
`asr_tone_conversation_coverage.json`. Inventoried inference outputs were read
only. No audio, prompt, model or rating was regenerated or changed.

## Matched task findings

Criteria are unchanged from the parent review: turn 3 must recover the situation
and propose a concrete action; turn 4 must explicitly acknowledge ending rather
than merely avoid a question or promise brevity. Counts below are manual audit
counts with the reasoning shown below, not benchmark accuracy.

| Pipeline | Controlled turn-3 concrete action | Rollout turn-3 concrete action | Controlled turn-4 explicit closure | Rollout turn-4 explicit closure |
|---|---:|---:|---:|---:|
| 10Hz speech with retained audio history | 0/6 | 0/6 | 0/6 | 0/6 |
| True TEXT, all user turns | 2/6 | 4/6 | 6/6 | 4/6 |
| Plain ASR, all user turns | 2/6 | 4/6 | 6/6 | 4/6 |
| ASR + predicted initial USER tone | 4/6 | 4/6 | 6/6 | 3/6 |

The predicted cue improves the controlled cafe next-step replies from the plain
baseline's generic “I'll keep it brief” to an explicit attendance check. It does
not improve the number of concrete rollout actions or endings. All six final
turns contain no new question, including three replies without clear closure.
None explicitly proposes continuing the *current* conversation; conditional
future availability is not treated as a current continuation request.

## Full per-family reading

**neu_base_04512, colleague feedback and an already-shared updated draft,
happy/sad.** Both controlled next-step replies are concrete: happy suggests
reviewing changes together; sad suggests waiting for colleagues' feedback. Both
endings explicitly acknowledge wrapping up. Happy rollout also proposes team
review before finalizing. Sad rollout suggests reviewing for clarity “before
sharing it with your team,” although the user already shared it: the action is
specific, but the chronology is awkward and should not be hidden by action
credit. Neither rollout ending explicitly closes: happy says “keep this
conversation brief and focused,” and sad says future interactions will be
brief/focused. Earlier sad/happy wording differs in encouragement but does not
establish useful delayed emotional adaptation.

**neu_base_00356, the usual cafe is arranged, attendance timing uncertain,
happy/sad.** Controlled turn 3 is identical for both tones: “check in with the
staff or a few friends to see if anyone has arrived yet.” This is a concrete
attendance-related action, although staff relevance is unestablished. Both
controlled endings acknowledge wrapping up. Happy rollout introduces a “cafe's
schedule” instead of people's timing, then twice says “whenever the usual cafe
arrives” — a venue/person confusion. Its next-step reply still contains the
concrete attendance check, but adds an unsupported sign-up sheet. Sad rollout
suggests asking the manager about a later start, a specific but assumption-heavy
option; its ending explicitly acknowledges wrap-up. The happy ending repeats
the impossible cafe-arrival phrase and does not close. Tone metadata has not
prevented lexical/agency confusion.

**neu_base_04315, the provided spreadsheet is thorough and accurate,
angry/sad.** Controlled turn 3 remains “I'll keep it brief” for both labels. Both
controlled endings acknowledge closure; angry mentions future availability,
which is not a proposal to continue now. Rollout angry asks to review earlier
details/discuss proceeding; sad suggests a quick follow-up to confirm next
steps. Neither gives a specific spreadsheet-related action, so neither receives
concrete-action credit. Both rollout endings explicitly close and promise no
new questions. Initial sad response says “I'm glad,” while angry response simply
echoes appreciation. The literal appreciation versus intended anger/sadness is
ambiguous without listening, and no confident emotion-appropriate behavioral
advantage is evident.

## What the cue changes, and what it does not prove

Only the first USER message contains
`[User delivery metadata: <classifier prediction>]`, retained in later textual
history. Neutral follow-ups have no tone label and the generic system policy is
unchanged. This differs from the single-turn tone baseline, where the same kind
of cue is appended to SYSTEM content; it is a conversation-specific annotation
comparator, not identical prompt placement. The LLM receives recognized words
and a classifier label, not audio. Intended labels remain synthetic settings,
not verified perceived emotion or clinical user states.

Thirty-three of 36 replies differ from plain ASR. Three of six controlled
same-base delivery comparisons are exactly identical; none of the twelve
rollout same-base comparisons is identical. Such string differences do not
establish useful emotional memory. Draft happy versus sad changes review versus
waiting; cafe sad later suggests a delayed start. These are plausible options,
but their emotional relevance is uncertain and some introduce unsupported
details. Under the strict requirement for a *specific, clearly appropriate*
late emotional adjustment, there is no convincing positive example here.

These three scenarios therefore support a narrow result: retaining an explicit
initial tone cue can change later wording and sometimes make a controlled next
step more concrete. It does not reliably fix content confusion, provide a clear
rollout advantage, or demonstrate useful delayed acoustic-cue carryover. The
branch-selected speech result must be evaluated separately from this 4775-update
parent; the frozen TEXT/ASR pipelines do not change with projector training.
