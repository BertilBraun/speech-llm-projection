# Lexical-alignment follow-up: measured comparisons

TEST results are descriptive; branch selection uses the precommitted VAL rule. CE is token-weighted within each cohort; macro CE averages the three cohorts. Semantic similarity measures reference resemblance, not response correctness.

| System | Cohort | CE examples | Target tokens | CE | Semantic | Generated | EOS | Cap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| followup_mean_10hz_ce_control_6775 | ordinary | 96 | 20883 | 1.61058 | 0.6326 | 48 | 34 | 14 |
| followup_mean_10hz_ce_control_6775 | qwen_emotional | 80 | 1892 | 0.76407 | 0.6309 | 24 | 24 | 0 |
| followup_mean_10hz_ce_control_6775 | neu_emotional | 80 | 2232 | 0.64642 | 0.7058 | 64 | 64 | 0 |
| followup_mean_10hz_ce_control_9550 | ordinary | 96 | 20883 | 1.60525 | 0.6265 | 48 | 28 | 20 |
| followup_mean_10hz_ce_control_9550 | qwen_emotional | 80 | 1892 | 0.75506 | 0.6559 | 24 | 24 | 0 |
| followup_mean_10hz_ce_control_9550 | neu_emotional | 80 | 2232 | 0.63861 | 0.6747 | 64 | 64 | 0 |

| System | Pooled CE | Macro CE |
|---|---:|---:|
| followup_mean_10hz_ce_control_6775 | 1.46048 | 1.00702 |
| followup_mean_10hz_ce_control_9550 | 1.45465 | 0.99964 |

| Checkpoint | Updates | VAL ordinary CE | VAL macro CE | Recorded cumulative training seconds | Peak PyTorch allocated decimal GB |
|---|---:|---:|---:|---:|---:|
| followup_mean_10hz_ce_control_6775 | 6775 | 1.58829 | 0.99770 | 7754.4 | 7.740 |
| followup_mean_10hz_ce_control_9550 | 9550 | 1.58071 | 0.98833 | 10947.7 | 7.740 |

Training elapsed counters include inherited parent checkpoint time and interval evaluation. Shared parents must be deduplicated for resource totals; incremental branch time requires subtracting the source checkpoint elapsed counter. Setup, final held-out evaluation and fidelity are separate from this training counter. Reused frozen TEXT/ASR baselines retain original timing and add no new GPU work. Cohort clocks are unmeasured.

| Checkpoint | TEST Neu pairs/families | Raw matching margin [95% CI] | Resized matching margin [95% CI] | Strict raw assignment win [95% CI] | Tie fraction | Distinct-target pairs |
|---|---:|---:|---:|---:|---:|---:|
| followup_mean_10hz_ce_control_6775 | 40/22 | 0.12021 [0.08154, 0.15368] | 0.10410 [0.06982, 0.13490] | 0.8750 [0.7381, 0.9762] | 0.0250 | 39 |
| followup_mean_10hz_ce_control_9550 | 40/22 | 0.13193 [0.09438, 0.16272] | 0.11091 [0.07517, 0.14158] | 0.8750 [0.7332, 0.9756] | 0.0250 | 39 |

Margins compare matched and swapped same-word audio against both saved teacher targets. The resized control linearly resamples wrong-audio encoder states to hold pseudo-token count, changing feature statistics. These are supporting conditioning diagnostics, not emotion classification accuracy or standalone semantic proof. Assignment wins require raw margin above the recorded numerical tie tolerance; ties receive no win credit here. Blinded response-rating win fractions separately use half-credit ties. Intended synthetic tone labels are not human-verified audible emotions.

| Checkpoint | Fidelity condition | Examples | First-token agreement | First-8 agreement | Full-prefix agreement |
|---|---|---:|---:|---:|---:|
| followup_mean_10hz_ce_control_6775 | speech | 256 | 0.8086 | 0.7114 | 0.7845 |
| followup_mean_10hz_ce_control_6775 | shuffled_speech | 24 | 0.5000 | 0.5260 | 0.7455 |
| followup_mean_10hz_ce_control_6775 | zero_speech | 24 | 0.2083 | 0.4219 | 0.7907 |
| followup_mean_10hz_ce_control_9550 | speech | 256 | 0.7930 | 0.7173 | 0.7864 |
| followup_mean_10hz_ce_control_9550 | shuffled_speech | 24 | 0.5000 | 0.5365 | 0.7476 |
| followup_mean_10hz_ce_control_9550 | zero_speech | 24 | 0.2083 | 0.4219 | 0.7907 |

All fidelity positions use the same teacher target prefix; late-prefix agreement is easier and does not establish free-running response quality. Blinded 0–2 tone/grounding reviews remain separate model-assisted ordinal judgments.
