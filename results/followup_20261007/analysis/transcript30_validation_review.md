# Transcript30 versus CE control: complete fixed24 validation review

All24 primary replies and eight ordinary histories were read. The comparison
helper verified identical ordered IDs, current words, histories and target
references; full replies and actual file hashes are retained in
`control6775_vs_transcript30_replies.md/json`. The canonical full24 review is
`selection/reviews/followup_mean_10hz_transcript30_6775.json`, with readable
case-by-case findings beside it. This is a model-assisted, unblinded validation
content audit, not human annotation, reference resemblance or TEST selection.

| Same endpoint6775 | Ordinary48 CE | Old40 CE | Neu40 CE | Macro CE | Primary EOS / cap |
|---|---:|---:|---:|---:|---:|
| Response CE control | 1.5882947180 | 0.7655269874 | 0.6392832394 | 0.9977016483 | 22 / 2 |
| Transcript30 | 1.5939120854 | 0.7604249582 | 0.6423853865 | 0.9989074767 | 24 / 0 |

Both meet the same48/40/40 validation count and cohort target-token counts:
10,723 /978 /1,110. Ordinary and Neu CE are slightly worse with transcript
mixing, old emotional CE slightly better; macro differs by only +0.001206.
These are one-seed endpoint comparisons, not an established practical effect.

Transcript30 Neu paired raw margin is +0.12515229, 95% family bootstrap interval
[0.06980134,0.18922219]; resized +0.10484914
[0.05447937,0.16384871], with20 bases /20 families. Robust margin is their
minimum, +0.10484914, versus control +0.10458395. Strict raw pair-assignment
wins are17/20 (85%, interval[70%,100%]), with no numerical ties. This is
teacher-forced target-to-audio matching, not emotion classification accuracy.
The resized control alters Whisper-state statistics and removes length cues;
it supports input sensitivity but does not prove semantic understanding.

Old emotional raw margin +0.03142036 [0.00782013,0.05462354] and resized
+0.04218132 [0.01456620,0.06874620] are also small. One of20 old target pairs
is identical. The synthetic intended deliveries are not verified perceived
emotions, and some literals conflict with their labels.

The two previously capped ordinary responses now reach EOS. Flipped-classroom
content preserves at-home learning/classroom discussion and becomes shorter;
the adaptation question still receives character discussion rather than any
adaptation. Completion is useful but does not repair that missed request.

The clear content correction is `qwen:utterance_01649_frustrated`: control says
“oat milk and cheese” for **almond milk and chia seeds**; transcript30 instead
gives a generic organized-list acknowledgment. It avoids the material false
ingredients without positively recovering either ingredient. The clear new
regression is `qwen:utterance_00286_happy`: grounded uphill difficulty is
replaced by an invented “view from the overlook.” These are recorded as one
correction and one regression. The predeclared veto (at least three new material
failures, exceeding corrections) is not met.

Other limitations remain visible: hamster agility becomes invented pet scenes;
the drawing is still mistaken for a retrieval task; angry appointment response
still reverses who confirmed it. Calendar responses add an unsupported
suitable-time goal and return to trying-to-set-up wording despite already-set
auto-sync. Those goal/setup drifts are flagged in the full review, rather than
silently called successful lexical recovery, but are not coded as a new explicit
material reversal under the conservative selection rubric. Concert and camera
cases remain generic and several are byte-identical to control.

The objective journal records4,777 transcription and11,223 response examples
in the added16,000 exposures:29.85625% transcription. Their109,208+1,254,320
supervised tokens total1,363,528, versus1,811,174 for the response-only control.
Equal updates/exposures therefore do not equal response-token supervision.
Mixed-objective training loss is not directly comparable with response-only
training loss; evaluation above always uses the original response CE.

No final branch selection has been run. The KL branch and its complete24-case
review must finish first; TEST outputs will not change the predeclared rule.
