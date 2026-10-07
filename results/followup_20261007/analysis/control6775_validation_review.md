# Matched CE-control endpoint: fixed 24 validation replies

Read every primary reply and all eight ordinary histories from the authoritative
6775-update CE-control copy. The CPU comparison helper asserts the ordered 24
IDs, literal user words, histories and saved targets are identical to the earlier
4775 parent; it saves full first/second replies and source hashes in
`parent4775_vs_control6775_replies.md/json`. This is a model-assisted content
reading, not human annotation, a TEST-based selection or a teacher-resemblance
score. It establishes the reference for forthcoming matched 6775 auxiliary
branches; no qualitative veto has yet been applied.

## Actual validation measurements

| Cohort | CE examples / tokens | Token-weighted CE | Generated EOS / cap |
|---|---:|---:|---:|
| Ordinary | 48 / 10,723 | 1.5882947180 | 6 / 2 |
| Old Qwen emotional | 40 / 978 | 0.7655269874 | 8 / 0 |
| New Neu emotional | 40 / 1,110 | 0.6392832394 | 8 / 0 |

Macro CE is 0.9977016483, versus the descriptive parent's 1.04715943. Primary
generations total 22 EOS-complete / 2 capped at 256 tokens. The new cap is the
flipped-classroom reply; the adaptation reply remains capped. Literal completion
labels are read from canonical records, not inferred from fluent endings.

New Neu paired CE margin: raw +0.12077904, 95% family-bootstrap interval
[0.06678508, 0.18526630]; resized +0.10458395,
[0.05778684, 0.15869344]. Robust selection margin is their minimum,
0.1045839522. Old emotional raw +0.03150010 [0.00545200, 0.05860265],
resized +0.05230318 [0.02595785, 0.08086103]. Each uses 20 paired bases / 20
families. Matching preferences are teacher-forced input-sensitivity diagnostics,
not tone-classification accuracy or independently validated response quality.
Wrong-audio resizing changes state statistics; intended labels are synthetic
settings rather than listened/perceived emotion gold.

## All eight ordinary cases

| Exact ID | Current-message finding and parent comparison |
|---|---|
| ordinary:cb12f37ac2ffb78c1585 | Relevant language/culture/translation enthusiasm and follow-up. No material grounding failure spotted; both endpoints coherent. |
| ordinary:14154499075c39d034f7 | Appropriate optimistic work/goals acknowledgment and asks what responsibilities are involved. Generic, but the user supplied no specific tasks. No material failure spotted. |
| ordinary:413d45dd5575ee18e8d9 | Now explicitly mentions new concepts **at home** and class discussion/problem-solving: a substantive improvement over the parent's incomplete flipped-classroom definition. The transition “students use that time ... at home” is clumsy and the canonical output is capped, despite looking like a finished response. Added first-person enthusiasm/“following its evolution” should not be treated as lived experience evidence. |
| ordinary:cf5f5a90ab8079a0bbc4 | Still answers a request for Pride and Prejudice **adaptations** with Elizabeth/Darcy character development, no adaptations; capped. This remains a clear missed request. A suspicious Elizabeth-illness/Darcy-tending aside does not improve grounding. The defective saved teacher's Winnie-the-Pooh claim is not factual gold and earns no reference credit. |
| ordinary:de5cb2bc961bb73e4721 | Relevant online-platform connection/trend appreciation and follow-up; no material failure spotted. |
| ordinary:8c95076b5a004c6e0c1c | Still does not recover hamster **agility courses** specifically. Adds unprovided feather-wand/lap-napping scenes and calls animals “little humans.” More recent-video phrasing than the parent, but a material unsupported-scene/topic error remains. Supplied history itself invents an assistant pet; that inherited perspective defect is disclosed rather than credited. |
| ordinary:f912451e8fd38f297083 | Still follows moonwalk/choreography history without acknowledging current title “This Is It.” Partial contextual relevance, no demonstrated precise recovery. An underspecified title fragment does not justify a blanket material-failure designation. |
| ordinary:d28987e37fb15c3cc9ff | Avoids inventing which politician/scandal; somewhat stiff generic refusal but no material factual claim spotted. |

## All eight old emotional cases

| Exact ID | Current-message finding and parent comparison |
|---|---|
| qwen:utterance_00766_frustrated | “Enjoying the challenge” adds an inference and omits 90%, but is coherent generic encouragement. Positive literal success versus frustrated annotation is ambiguous; the pair's teacher targets are identical. Not a clear current-message factual reversal. |
| qwen:utterance_00766_happy | Preserves better-than-anticipated score meaning without repeating the number. Appropriate positive acknowledgment; no material failure spotted. |
| qwen:utterance_01649_frustrated | **Almond milk→oat milk; chia seeds→cheese** remains exactly the same material ingredient substitution. Also claims it will edit the grocery list despite no tool. This is a key matched-branch challenge. |
| qwen:utterance_01649_neutral | Now generic “solid plan ... help with the shopping list”; avoids the parent's false ingredient substitutions but does not recover either ingredient. This is partial/detail-free success, not proof of lexical correction. |
| qwen:utterance_02251_happy | Removes the parent's explicit **tonight** reversal; generic excitement about the concert. No incorrect time now, but does not explicitly preserve next weekend. |
| qwen:utterance_02251_sad | Likewise removes invented concert-tonight attendance; generic positive excitement. Literal happiness versus sad annotation remains ambiguous. Removal of the false timing is useful, not complete time recovery. |
| qwen:utterance_00286_happy | Recovers **uphill part is tough**, removing the unprovided overlook destination. Clear content improvement. “Making progress” is mild inferred encouragement. |
| qwen:utterance_00286_sarcastic | Recovers uphill as a remaining challenge, removing “finally getting the hang of” implication. Helpful generic reply, without clear sarcasm-sensitive adaptation. No claim about actual perceived sarcasm. |

## All eight new Neu emotional cases

| Exact ID | Current-message finding and parent comparison |
|---|---|
| neu:neu_base_04082_angry | User reported asking whether the group wanted to **see their existing half-finished drawing**. Reply misreads this as a request to check/find the drawing and claims retrieval action. Parent's sketching misread has changed, not been resolved: clear activity/agency mismatch remains. |
| neu:neu_base_04082_happy | Similarly offers to **find** the drawing and introduces group files/meeting access. This is an unrequested retrieval task, still a material activity mismatch. |
| neu:neu_base_04095_angry | Exactly unchanged “I appreciate **you** confirming...” reverses the user thanking the assistant for confirmation. Morning/family words preserved; material agency reversal remains and was already present in the teacher. |
| neu:neu_base_04095_sad | Generic confirmation acknowledgment; morning slot/family appointment preserved. Little sadness-specific support, but no clear content failure spotted. |
| neu:neu_base_03973_fearful | Exactly unchanged grounded calendar uncertainty acknowledgment. Generic offer to help, no invented cause. Useful concern-sensitive opening. |
| neu:neu_base_03973_happy | Now acknowledges concern/not working rather than “trying to set up” an already-set calendar. Grounded uncertainty remains, but happy/fearful contrast narrows; this is not automatically a tone error without listening. |
| neu:neu_base_03052_happy | Removes “before the group message goes out” chronology invention. Current camera-location uncertainty is acknowledged; action becomes generic supportive language rather than a specific verification step. |
| neu:neu_base_03052_sad | Removes unprovided **temporary glitch/signal misinterpretation** diagnosis. Supportive uncertainty acknowledgment fits literal words; no actual diagnosis. Clear hallucination removal with limited practical specificity. |

## Reference for the later auxiliary comparison

The CE-only extra 2000 updates already improve several concrete weaknesses:
flipped-classroom core explanation, uphill difficulty, removal of wrong concert
timing, and removal of camera chronology/technical diagnoses. Some improvements
avoid details rather than positively reconstruct them. Persistent clear material
failures include the adaptation request, frustrated grocery substitutions,
hamster invented scenes, both drawing task misreads and angry appointment agency
reversal. These exact cases should anchor the auxiliary comparisons.

Any later qualitative veto uses **all fixed 24** matched replies: at least three
new clear material contradictions/missed requests versus this CE control, and
regressions must exceed clearly corrected material failures. One isolated error,
minor stylistic variation, omitted lexical repetition, tone-label ambiguity or
teacher-reference disagreement is not enough. All case IDs and reasons,
including corrections, must remain visible. No TEST response influences this
selection guard.
