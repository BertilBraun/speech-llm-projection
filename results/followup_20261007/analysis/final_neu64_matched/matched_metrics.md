# Exactly matched neu_emotional generation panel

CE, semantics and completion below use identical IDs. CE is token-weighted, not a mean of per-example losses. This subset has no separately measured clock. Semantics are reference similarity, not correctness or tone accuracy.

| System | IDs | Target tokens | CE | Semantic | EOS | Cap |
|---|---:|---:|---:|---:|---:|---:|
| speech_10hz_parent4775 | 64 | 1802 | 0.745869 | 0.677354290150106 | 64 | 0 |
| followup_mean_10hz_ce_control_6775 | 64 | 1802 | 0.655528 | 0.7058305903337896 | 64 | 0 |
| true_text | 64 | 1802 | 0.746307 | 0.7205655281431973 | 64 | 0 |
| plain_asr | 64 | 1802 | 0.779865 | 0.7150281346403062 | 64 | 0 |
| asr_predicted_tone | 64 | 1802 | 0.686550 | 0.9080769470892847 | 64 | 0 |
| asr_oracle_tone_reused | 64 | 1802 | 0.686550 | 0.9080769470892847 | 64 | 0 |

The larger cohort CE table uses different coverage and is reported separately. All intended-tone labels describe unverified synthetic settings. Predicted-tone and oracle-label provenance remain distinct even if their actual input labels coincide.
