# Follow-up resource ledger (completed artifacts only)

Captured 2026-10-07T13:31:02.661328+00:00.

| Run | Added updates/examples/target tokens | Added training-loop s | Cumulative training s (do not add) | Saved lineage peak allocated decimal GB |
|---|---|---:|---:|---:|
| followup_mean_10hz_ce_control_6775 | 2000/16000/1811174 | 2220.077849 | 7754.376689 | 7.739921 |
| followup_mean_10hz_ce_control_9550 | 2775/22193/2503347 | 3193.273594 | 10947.650283 | 7.739921 |
| followup_mean_10hz_epoch1_ce | 2775/22193/2507795 | 3247.800758 | 5534.298840 | 7.739921 |
| followup_mean_10hz_ordinarykl_6775 | 2000/16000/1811174 | 2670.044825 | 8204.343665 | 7.739921 |
| followup_mean_10hz_transcript30_6775 | 2000/16000/1363528 | 2287.816806 | 7822.115646 | 7.739921 |

Total distinct completed training increments: 13619.013832 s.

Classifier CPU function wall 22.717704 s, nested feature loading 14.913937 s and fitting 3.720707 s.

- Training increments subtract the exact SHA-bound parent checkpoint elapsed time; cumulative counters are references and must not be summed again.
- Training-loop wall includes recorded training overhead; it is not measured GPU utilization, CUDA kernel time, or the whole process including startup/evaluation.
- Classifier total CPU wall contains feature loading and fitting; nested timers must not be added to its total.
- Classifier memory observation was RSS 1,876,056 KiB at one instant, not a peak; its 10000×4608 float64 feature matrix occupies 368,640,000 bytes before copies.
- PyTorch allocated decimal GB is distinct from reserved memory and NVIDIA/NVML usage.
- Saved training peak is a lineage maximum: continuation copies training_resources.json and preserves max(previous peak, newly observed peak), so it can include the parent rather than measuring only the new branch.
- No new training-feature cache extraction, audio synthesis or teacher generation; three existing neutral follow-up clips (followup_short, next_step, stop) were transcribed once for the conversation comparator and reused thereafter. Their Whisper encoder/decoder runtime is contained in the conversation-job scope, not a separate extraction benchmark.
- Final inference/evaluation, smoke/startup/debug and rental-time measurements remain separate or unmeasured here. No rental rate or cost is inferred.
- Only completed immutable run captures are read; active branches are pending.
