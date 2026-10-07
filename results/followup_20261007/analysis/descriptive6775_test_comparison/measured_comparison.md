# Lexical-alignment follow-up: measured comparisons

TEST results are descriptive; branch selection uses the precommitted VAL rule. CE is token-weighted within each cohort; macro CE averages the three cohorts. Semantic similarity measures reference resemblance, not response correctness.

| System | Cohort | CE examples | Target tokens | CE | Semantic | Generated | EOS | Cap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| followup_mean_10hz_ce_control_6775 | ordinary | 96 | 20883 | 1.61058 | 0.6326 | 48 | 34 | 14 |
| followup_mean_10hz_ce_control_6775 | qwen_emotional | 80 | 1892 | 0.76407 | 0.6309 | 24 | 24 | 0 |
| followup_mean_10hz_ce_control_6775 | neu_emotional | 80 | 2232 | 0.64642 | 0.7058 | 64 | 64 | 0 |
| followup_mean_10hz_transcript30_6775 | ordinary | 96 | 20883 | 1.61537 | 0.5815 | 48 | 33 | 15 |
| followup_mean_10hz_transcript30_6775 | qwen_emotional | 80 | 1892 | 0.75988 | 0.6310 | 24 | 24 | 0 |
| followup_mean_10hz_transcript30_6775 | neu_emotional | 80 | 2232 | 0.65123 | 0.6905 | 64 | 64 | 0 |
| followup_mean_10hz_ordinarykl_6775 | ordinary | 96 | 20883 | 1.66679 | 0.6025 | 48 | 32 | 16 |
| followup_mean_10hz_ordinarykl_6775 | qwen_emotional | 80 | 1892 | 0.75974 | 0.6454 | 24 | 24 | 0 |
| followup_mean_10hz_ordinarykl_6775 | neu_emotional | 80 | 2232 | 0.64519 | 0.6924 | 64 | 64 | 0 |
| true_TEXT | ordinary | 96 | 20883 | 1.55649 | 0.6733 | 48 | 29 | 19 |
| true_TEXT | qwen_emotional | 80 | 1892 | 0.73676 | 0.7064 | 24 | 24 | 0 |
| true_TEXT | neu_emotional | 80 | 2232 | 0.73896 | 0.7206 | 64 | 64 | 0 |
| plain_ASR | ordinary | 96 | 20883 | 1.56788 | 0.6749 | 48 | 30 | 18 |
| plain_ASR | qwen_emotional | 80 | 1892 | 0.77696 | 0.7006 | 24 | 24 | 0 |
| plain_ASR | neu_emotional | 80 | 2232 | 0.77026 | 0.7150 | 64 | 64 | 0 |

| System | Pooled CE | Macro CE |
|---|---:|---:|
| followup_mean_10hz_ce_control_6775 | 1.46048 | 1.00702 |
| followup_mean_10hz_transcript30_6775 | 1.46459 | 1.00883 |
| followup_mean_10hz_ordinarykl_6775 | 1.50698 | 1.02391 |
| true_TEXT | 1.42150 | 1.01074 |
| plain_ASR | 1.43685 | 1.03837 |

| Checkpoint | Updates | VAL ordinary CE | VAL macro CE | Recorded cumulative training seconds | Peak PyTorch allocated decimal GB |
|---|---:|---:|---:|---:|---:|
| followup_mean_10hz_ce_control_6775 | 6775 | 1.58829 | 0.99770 | 7754.4 | 7.740 |
| followup_mean_10hz_transcript30_6775 | 6775 | 1.59391 | 0.99891 | 7822.1 | 7.740 |
| followup_mean_10hz_ordinarykl_6775 | 6775 | 1.65061 | 1.01666 | 8204.3 | 7.740 |

Training elapsed counters include inherited parent checkpoint time and interval evaluation. Shared parents must be deduplicated for resource totals; incremental branch time requires subtracting the source checkpoint elapsed counter. Setup, final held-out evaluation and fidelity are separate from this training counter. Reused frozen TEXT/ASR baselines retain original timing and add no new GPU work. Cohort clocks are unmeasured.

| Checkpoint | TEST Neu pairs/families | Raw matching margin [95% CI] | Resized matching margin [95% CI] | Strict raw assignment win [95% CI] | Tie fraction | Distinct-target pairs |
|---|---:|---:|---:|---:|---:|---:|
| followup_mean_10hz_ce_control_6775 | 40/22 | 0.12021 [0.08154, 0.15368] | 0.10410 [0.06982, 0.13490] | 0.8750 [0.7381, 0.9762] | 0.0250 | 39 |
| followup_mean_10hz_transcript30_6775 | 40/22 | 0.11680 [0.08033, 0.14785] | 0.10103 [0.06527, 0.13337] | 0.8500 [0.7105, 0.9546] | 0.0250 | 39 |
| followup_mean_10hz_ordinarykl_6775 | 40/22 | 0.12098 [0.08263, 0.15482] | 0.10744 [0.07260, 0.13966] | 0.9000 [0.7692, 1.0000] | 0.0250 | 39 |

Margins compare matched and swapped same-word audio against both saved teacher targets. The resized control linearly resamples wrong-audio encoder states to hold pseudo-token count, changing feature statistics. These are supporting conditioning diagnostics, not emotion classification accuracy or standalone semantic proof. Assignment wins require raw margin above the recorded numerical tie tolerance; ties receive no win credit here. Blinded response-rating win fractions separately use half-credit ties. Intended synthetic tone labels are not human-verified audible emotions.

| Checkpoint | Fidelity condition | Examples | First-token agreement | First-8 agreement | Full-prefix agreement |
|---|---|---:|---:|---:|---:|
| followup_mean_10hz_ce_control_6775 | speech | 256 | 0.8086 | 0.7114 | 0.7845 |
| followup_mean_10hz_ce_control_6775 | shuffled_speech | 24 | 0.5000 | 0.5260 | 0.7455 |
| followup_mean_10hz_ce_control_6775 | zero_speech | 24 | 0.2083 | 0.4219 | 0.7907 |
| followup_mean_10hz_transcript30_6775 | speech | 256 | 0.7930 | 0.7148 | 0.7865 |
| followup_mean_10hz_transcript30_6775 | shuffled_speech | 24 | 0.4583 | 0.5208 | 0.7459 |
| followup_mean_10hz_transcript30_6775 | zero_speech | 24 | 0.2083 | 0.4271 | 0.7905 |
| followup_mean_10hz_ordinarykl_6775 | speech | 256 | 0.8047 | 0.7129 | 0.8468 |
| followup_mean_10hz_ordinarykl_6775 | shuffled_speech | 24 | 0.5000 | 0.5469 | 0.8088 |
| followup_mean_10hz_ordinarykl_6775 | zero_speech | 24 | 0.2083 | 0.4219 | 0.7907 |

All fidelity positions use the same teacher target prefix; late-prefix agreement is easier and does not establish free-running response quality. Blinded 0–2 tone/grounding reviews remain separate model-assisted ordinal judgments.
