# Full10Hz parent: descriptive fixed24 validation review

Reviewed all24 primary speech replies and all8 ordinary histories from the
authoritative completed4775-update copy. This is an independent model-assisted
content audit, not human annotation, teacher-resemblance scoring, a matched6775
branch comparison, or a qualitative branch veto. No inference was run.

VAL cohort CE: ordinary1.59857443, old Qwen emotional0.81929074,
Neu emotional0.72361313; three-cohort macro1.04715943. Each cohort CE is
token-weighted on48/40/40 examples. Primary generations23EOS/1cap out of24;
ordinary7/8EOS, both emotional cohorts8/8EOS. These are separate samples from
the forthcoming fixedTEST256CE/136-generation panel.

Neu same-word paired target margin is+.07148068 raw,95% familyCI
[.03778721,.11069761], and+.04917528 resized,[.01843907,.08362654],20pairs/20families.
Old emotional raw+.02367257,[.00224959,.04463255]; resized+.04388973,
[.01800220,.07490499]. These teacher-forced target preferences support input
sensitivity, not emotion classification accuracy or verified response quality.
Resizing wrong-audio features changes statistics; labels are intended synthetic
delivery settings, not verified audible emotion.

## All fixed ordinary cases

| Exact ID | Current-message assessment |
|---|---|
| ordinary:cb12f37ac2ffb78c1585 | Coherent language/translation/culture response. General enthusiasm plus relevant follow-up. No material contradiction spotted. |
| ordinary:14154499075c39d034f7 | Coherent optimistic work/goals acknowledgment with options; generic rather than a concrete responsibility plan, but the user supplied no specific tasks. |
| ordinary:413d45dd5575ee18e8d9 | Recognizes flipped-classroom topic and teacher facilitation, but places technology engagement in class and omits the defining outside-class preparation/inside-class practice reversal. Explanatory detail remains materially incomplete. |
| ordinary:cf5f5a90ab8079a0bbc4 | User asks for Pride and Prejudice adaptations. Reply gives Elizabeth/Darcy character analysis, no adaptations, and hits256-token cap. It is on the literary topic but misses the requested action. The saved teacher's Winnie-the-Pooh adaptation assertion is also defective and is not factual gold. |
| ordinary:de5cb2bc961bb73e4721 | Relevant online-community/trends appreciation and a natural follow-up. General but coherent. |
| ordinary:8c95076b5a004c6e0c1c | Current input asks about hamster agility videos; response speaks of little paws, chasing, missing them and a snuggle party. It follows prior pet context more than the new agility question. Earlier supplied assistant history already invents a guinea pig, so perspective errors are partly inherited from synthetic history. |
| ordinary:f912451e8fd38f297083 | “This Is It” receives a generic moonwalk/choreography response supported by earlier music-video history, but the exact title is not acknowledged. Partial contextual relevance, insufficient evidence of precise current-input recovery. |
| ordinary:d28987e37fb15c3cc9ff | Appropriately avoids inventing the unspecified politician/scandal and advises checking sources. Generic caution fits the underspecified input. |

## All fixed old emotional cases

| Exact ID | Reply / assessment |
|---|---|
| qwen:utterance_00766_frustrated | “That's wonderful to hear! I'm glad you felt confident enough to score90%.” Numeric90% is preserved; confidence is inferred. Positive words conflict ambiguously with frustrated annotation; teacher is identical across the pair. |
| qwen:utterance_00766_happy | “That's wonderful to hear! I'm glad the quiz was a success for you.” Coherent positive acknowledgment of success; generic. |
| qwen:utterance_01649_frustrated | “I'll add the oat milk and cheese to your grocery list right away.” Almond milk→oat milk and chia seeds→cheese are two concrete substitutions; unsupported action claim. |
| qwen:utterance_01649_neutral | Same incorrect oat-milk/cheese reply. No pair variation; both current ingredients missed. |
| qwen:utterance_02251_happy | “...you're ready to head out tonight.” Reverses explicit next-weekend-instead-of-tonight timing. |
| qwen:utterance_02251_sad | “...I'm glad you're enjoying the concert tonight.” Same timing reversal plus invented ongoing attendance; positive literal text/intended sadness itself is ambiguous. |
| qwen:utterance_00286_happy | “...a fantastic way to get to the overlook.” Unprovided destination and no acknowledgment of still-unmastered uphill section. |
| qwen:utterance_00286_sarcastic | “...finally starting to get the hang of the uphill part!” Weakens/reverses the explicit remaining difficulty. Sarcasm is not recovered clearly; no claim about actual audible sarcasm. |

## All fixed Neu emotional cases

| Exact ID | Reply / assessment |
|---|---|
| neu:neu_base_04082_angry | Offers to help sketch the drawing, then says it lacks the group agenda. User only reported asking the group to see an existing drawing; unsolicited task/agenda framing. Anger-specific acknowledgment absent. |
| neu:neu_base_04082_happy | Same offer to sketch, without agenda disclaimer. Cheerful style variation exists, but changes the user action to an unrequested assistant task. |
| neu:neu_base_04095_angry | Preserves morning/family-appointment words; “I appreciate you confirming...” reverses who thanked whom. Generic next-step promise, not clearly angry-aware. |
| neu:neu_base_04095_sad | Preserves confirmation and morning appointment; generic “I'm glad...” style, little sadness-specific support. Pair variation mostly teacher-like framing. |
| neu:neu_base_03973_fearful | “I understand that the shared calendar might not work as expected...” accurately acknowledges uncertainty and offers to determine next steps. Useful grounded concern acknowledgment. |
| neu:neu_base_03973_happy | “I'm glad you're trying to set up the calendar...” remains supportive and preserves uncertainty, though setup was already completed. Plausible positive/concern style contrast. |
| neu:neu_base_03052_happy | Offers to double-check potential camera-location error, a fitting action. “Before the group message goes out” invents chronology; the message was already received. |
| neu:neu_base_03052_sad | “...likely just a temporary glitch or a misinterpretation of the signal.” Unwarranted technical diagnosis of a possible erroneous group message; unsupported reassurance despite concern-sensitive opening. |

The chief actionable failures are lexical ingredient substitution, concert-time
reversal, missed adaptation request, and role/agency/chronology drift. Improved
CE and positive paired margins do not eliminate these. Transcript auxiliary/KL
branches should be judged against all24 matched6775 CE-control replies before
applying the precommitted pragmatic veto; none is applied from this earlier parent.
