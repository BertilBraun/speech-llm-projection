# Final follow-up resource scopes

Distinct new training-loop time: 13619.013832s.
Recorded nonoverlapping GPU-using process wall: 15440.313000s (15 process executions).

| Process | Scope | Wall seconds | Exit status |
|---|---|---:|---:|
| followup10hz-tone-classifier | cpu | 27.783 | 0 |
| followup10hz-code-check | cpu | 11.474 | 0 |
| followup10hz-fullpass | training | 3338.225 | 0 |
| followup10hz-ordinarykl-smoke | smoke | 81.872 | 0 |
| followup10hz-tone-baseline | evaluation | 37.547 | 0 |
| followup10hz-test-epoch1 | evaluation | 192.042 | 0 |
| followup10hz-conversation-epoch1 | evaluation | 103.540 | 0 |
| followup10hz-ce-control | training | 2351.898 | 0 |
| followup10hz-conversation-tone | evaluation | 52.467 | 0 |
| followup10hz-transcript30 | training | 2401.277 | 0 |
| followup10hz-ordinarykl | training | 2736.290 | 0 |
| followup10hz-test-ordinarykl | evaluation | 190.359 | 0 |
| followup10hz-selected9550-training | training | 3328.019 | 0 |
| followup10hz-selected9550-test | evaluation | 196.334 | 0 |
| followup10hz-test-ce_control | evaluation | 190.196 | 0 |
| followup10hz-test-transcript30 | evaluation | 188.204 | 0 |
| followup10hz-conversation-ce_control | evaluation | 52.043 | 0 |

| Evaluator artifact | Total evaluation seconds | Nested generation seconds |
|---|---:|---:|
| results_preview\followup10hz_20261007\evaluation_ce_control\speech\evaluation.json | 146.689857 | 78.141840 |
| results_preview\followup10hz_20261007\evaluation_epoch1\speech\evaluation.json | 147.600130 | 77.468183 |
| results_preview\followup10hz_20261007\evaluation_followup_mean_10hz_ce_control_9550\speech\evaluation.json | 152.177330 | 82.951890 |
| results_preview\followup10hz_20261007\evaluation_ordinarykl\speech\evaluation.json | 146.562599 | 77.606134 |
| results_preview\followup10hz_20261007\evaluation_transcript30\speech\evaluation.json | 144.522781 | 75.689788 |
| results_preview\followup10hz_20261007\runs\followup_mean_10hz_ce_control_6775\validation\evaluation.json | 42.791335 | 16.891853 |
| results_preview\followup10hz_20261007\runs\followup_mean_10hz_ce_control_9550\validation\evaluation.json | 43.489747 | 17.251701 |
| results_preview\followup10hz_20261007\runs\followup_mean_10hz_epoch1_ce\validation\evaluation.json | 43.955104 | 17.250157 |
| results_preview\followup10hz_20261007\runs\followup_mean_10hz_ordinarykl_6775\validation\evaluation.json | 43.415291 | 17.172001 |
| results_preview\followup10hz_20261007\runs\followup_mean_10hz_transcript30_6775\validation\evaluation.json | 38.838508 | 12.573976 |
| results_preview\followup10hz_20261007\baseline\asr_predicted_tone\evaluation.json | 30.034005 | 16.959977 |

Standalone evaluator-call total: 980.076687s, a partial component scope within process wall.

| Conversation summary | Nested generation seconds |
|---|---:|
| results_preview\followup10hz_20261007\conversation_epoch1\speech\summary.json | 51.782374 |
| results_preview\followup10hz_20261007\conversation_ce_control\speech\summary.json | 44.697436 |
| results_preview\followup10hz_20261007\conversation_epoch1\text\summary.json | 20.979485 |
| results_preview\followup10hz_20261007\conversation_epoch1\asr\summary.json | 21.055818 |
| results_preview\followup10hz_20261007\conversation_asr_tone\summary.json | 44.971751 |

- Whole-process wall includes startup, training/evaluation and shutdown; it is not CUDA compute or utilization.
- Training-loop increments and component evaluation timers overlap whole-process wall and are not added to it.
- Evaluator generation timers are nested; periodic training validation lies within recorded training-loop time.
- Oracle-tone evaluation was reused from predicted-tone output; its copied timer is excluded.
- Conversation summaries and their per-reply timers are nested; matched subsets/reused comparators are not counted again.
- Three neutral follow-up ASRs occurred once within the original conversation process; no new training feature extraction.
- Saved PyTorch allocated peak is a lineage maximum; reserved memory and point NVIDIA usage are separate.
- Provider billing duration/rate and GPU utilization-hours were not measured; no rental cost is inferred.
