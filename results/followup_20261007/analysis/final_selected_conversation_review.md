# Selected 6775 conversation audit

I personally read every saved reply in the hash-verified merge: 60 selected-speech replies, 36 unchanged TEXT replies, 36 unchanged ASR replies, and the separate 36 ASR-initial-predicted-tone replies. The compact all168 record is [here](final_conversation_all168_compact.md); full inputs and source receipts remain in `conversation_ce_control/merged`. The 132-record merge contains **108 fully matched pipeline replies plus 24 speech-history ablations**, rather than 132 uniformly matched observations. No additional inference was performed by this audit.

The same three held-out synthetic families cover an updated draft already shared, a cafe gathering with uncertain attendance, and a spreadsheet described as accurate and thorough. Each has two intended deliveries. These labels were not verified by human listening. The neutral later audio is shared. A useful next step must name a concrete action relevant to the initial situation; promising to find a next step does not qualify. Explicit closure means accepting the end of the conversation, not merely omitting a question mark. Counts below are model-assisted qualitative coding, not certified accuracy, and six branches represent only three independent design families.

| Pipeline / arm | Concrete relevant next step at turn3 | Explicit closure at turn4 | Comment |
|---|---:|---:|---|
| Selected speech, retained-audio controlled | 0/6 | 0/6 | Generic next-step assurances; ends become moving-forward instructions |
| Selected speech, text-history controlled | 0/6 | 4/6 | Draft/cafe branches wrap up; spreadsheet still continues |
| Selected speech, omit-initial controlled | 0/6 | 0/6 | Repeated “cannot end a conversation or stop asking questions” |
| Selected speech, four-turn rollout | 0/6 | 0/6 clear; 1/6 borderline | Happy draft says “proceed without further discussion,” but frames continuation rather than explicit ending |
| TEXT, controlled | 2/6 | 6/6 | Draft action is specific but also says before sending despite already shared |
| TEXT, rollout | 4/6 | 4/6 | Draft review/cafe attendance checks; spreadsheet remains generic |
| Plain ASR, controlled | 2/6 | 6/6 | Waiting for draft feedback is sensible and grounded |
| Plain ASR, rollout | 4/6 | 4/6 | Draft/cafe actions; spreadsheet generic |
| ASR with initial predicted tone, controlled | 4/6 | 6/6 | Draft/cafe actions; spreadsheet generic |
| ASR with initial predicted tone, rollout | 4/6 topic-specific, with factual caveats | 3/6 | Cafe sign-up-sheet invention and “cafe arrives”; sad draft reverts to before sharing |

The TEXT/ASR rows are reused frozen-model outputs whose effective prompt, history, decoding, fixtures and models were checked by the canonical merge receipt. Their timing is retained from the original evaluations and must not be counted as new GPU work. ASR-initial-tone is separate: its cue occurs only in the first USER history message; it differs from the single-turn SYSTEM cue. Later neutral turns receive no tone annotation.

## Concrete cases

For the already-shared draft, selected retained speech answers the next-step question with “I need to know exactly what you need next.” Its own rollout answers “I'm here to help you find the next step,” without recovering the draft or sharing state. Plain ASR instead says to wait for colleagues' feedback before further changes. TEXT also recovers the draft but its controlled reply's “before sending it out” qualification conflicts with the already-shared state. Thus a specific action can still contain a grounding defect; counts do not imply perfect factual replies.

For the cafe, selected retained speech says only that it will keep the next step clear and concise; no attendance check is proposed. TEXT and ASR rollouts suggest a concrete check-in. The historical-tone rollout sometimes treats the venue as a person (“whenever the usual cafe arrives”) or invents a sign-up sheet. It therefore does not establish reliable delayed emotion benefit merely because output strings differ.

For the accurate spreadsheet, all selected speech next-step arms remain generic. At closure both intended deliveries say “We can move forward with the next step,” including the training-like text-history arm. Plain TEXT/ASR also fail the next-step task but generally recognize the explicit wrap-up instruction. This reveals a current-input comprehension problem beyond the retained-audio-history layout being unfamiliar during training.

At closure, selected text-history draft/cafe cases say “We can wrap up here” (four clear successes), improving on the parent4775 text-history result of zero. Retained-audio control still gives future-discussion assurances in all six branches. The happy-draft rollout's “proceed without further discussion” is retained as borderline rather than inflated into clear compliance. The other five rollouts explicitly promise next-step assistance after a request to end.

## Acoustic-cue interpretation

Initial speech replies sometimes differ appropriately in warmth versus support. Later controlled retained-audio outputs are identical across deliveries in four of six family-by-turn comparisons, and every controlled turn4 pair is identical. The differing turn3 replies still provide no concrete situation-specific action or clearly useful adjustment based on earlier tone. All text-history controlled pairs are identical, as expected when the two initial words are identical and their audio is not retained.

There is **no convincing useful delayed acoustic-emotion retention** in this small panel. This is not proof that an encoder has no tone information: separate probe and paired-loss diagnostics address that different question. Retained-audio history is an inference layout not trained here; failure of the training-like current-audio/text-history arm nevertheless shows that layout mismatch alone does not explain the content and closure failures. A future experiment should train multi-turn instruction/closure grounding with clear role and state targets before claiming late emotion memory.

The original parental audit and the root's independent [final review](root_final_multiturn_review.md) remain separate. All original replies and zero-cap completion records are preserved; this report neither regenerates nor relabels them.
