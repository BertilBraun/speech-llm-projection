# Lexical-alignment follow-up: measured comparisons

TEST results are descriptive; branch selection uses the precommitted VAL rule. CE is token-weighted within each cohort; macro CE averages the three cohorts. Semantic similarity measures reference resemblance, not response correctness.

| System | Cohort | CE examples | Target tokens | CE | Semantic | Generated | EOS | Cap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| speech_2.5Hz_4775 | ordinary | 96 | 20883 | 1.63412 | 0.5717 | 48 | 40 | 8 |
| speech_2.5Hz_4775 | qwen_emotional | 80 | 1892 | 0.85160 | 0.6014 | 24 | 24 | 0 |
| speech_2.5Hz_4775 | neu_emotional | 80 | 2232 | 0.74219 | 0.6782 | 64 | 64 | 0 |
| speech_10Hz_4775 | ordinary | 96 | 20883 | 1.62198 | 0.5780 | 48 | 33 | 15 |
| speech_10Hz_4775 | qwen_emotional | 80 | 1892 | 0.82798 | 0.6255 | 24 | 24 | 0 |
| speech_10Hz_4775 | neu_emotional | 80 | 2232 | 0.73522 | 0.6774 | 64 | 64 | 0 |
| true_TEXT | ordinary | 96 | 20883 | 1.55649 | 0.6733 | 48 | 29 | 19 |
| true_TEXT | qwen_emotional | 80 | 1892 | 0.73676 | 0.7064 | 24 | 24 | 0 |
| true_TEXT | neu_emotional | 80 | 2232 | 0.73896 | 0.7206 | 64 | 64 | 0 |
| plain_ASR | ordinary | 96 | 20883 | 1.56788 | 0.6749 | 48 | 30 | 18 |
| plain_ASR | qwen_emotional | 80 | 1892 | 0.77696 | 0.7006 | 24 | 24 | 0 |
| plain_ASR | neu_emotional | 80 | 2232 | 0.77026 | 0.7150 | 64 | 64 | 0 |

| System | Pooled CE | Macro CE |
|---|---:|---:|
| speech_2.5Hz_4775 | 1.49531 | 1.07597 |
| speech_10Hz_4775 | 1.48276 | 1.06173 |
| true_TEXT | 1.42150 | 1.01074 |
| plain_ASR | 1.43685 | 1.03837 |

| Checkpoint | Updates | VAL ordinary CE | VAL macro CE | Recorded cumulative training seconds | Peak PyTorch allocated decimal GB |
|---|---:|---:|---:|---:|---:|
| speech_2.5Hz_4775 | 4775 | 1.61719 | 1.08683 | 5230.7 | 7.739 |
| speech_10Hz_4775 | 4775 | 1.59857 | 1.04716 | 5534.3 | 7.740 |

Training elapsed counters include inherited parent checkpoint time and interval evaluation. Shared parents must be deduplicated for resource totals; incremental branch time requires subtracting the source checkpoint elapsed counter. Setup, final held-out evaluation and fidelity are separate from this training counter. Reused frozen TEXT/ASR baselines retain original timing and add no new GPU work. Cohort clocks are unmeasured.

| Checkpoint | TEST Neu pairs/families | Raw matching margin [95% CI] | Resized matching margin [95% CI] | Strict raw assignment win [95% CI] | Tie fraction | Distinct-target pairs |
|---|---:|---:|---:|---:|---:|---:|
| speech_2.5Hz_4775 | 40/22 | 0.07862 [0.05601, 0.10498] | 0.06239 [0.03817, 0.08555] | 0.8750 [0.7959, 0.9688] | 0.0250 | 39 |
| speech_10Hz_4775 | 40/22 | 0.08309 [0.05666, 0.10553] | 0.06240 [0.03810, 0.08442] | 0.8500 [0.7105, 0.9574] | 0.0250 | 39 |

Margins compare matched and swapped same-word audio against both saved teacher targets. The resized control linearly resamples wrong-audio encoder states to hold pseudo-token count, changing feature statistics. These are supporting conditioning diagnostics, not emotion classification accuracy or standalone semantic proof. Assignment wins require raw margin above the recorded numerical tie tolerance; ties receive no win credit here. Blinded response-rating win fractions separately use half-credit ties. Intended synthetic tone labels are not human-verified audible emotions.

| Checkpoint | Fidelity condition | Examples | First-token agreement | First-8 agreement | Full-prefix agreement |
|---|---|---:|---:|---:|---:|
| speech_2.5Hz_4775 | speech | 256 | 0.7930 | 0.7095 | 0.7881 |
| speech_2.5Hz_4775 | shuffled_speech | 24 | 0.6250 | 0.5521 | 0.7569 |
| speech_2.5Hz_4775 | zero_speech | 24 | 0.3750 | 0.4427 | 0.7926 |
| speech_10Hz_4775 | speech | 256 | 0.7812 | 0.7017 | 0.7837 |
| speech_10Hz_4775 | shuffled_speech | 24 | 0.3750 | 0.5208 | 0.7535 |
| speech_10Hz_4775 | zero_speech | 24 | 0.2083 | 0.4219 | 0.7907 |

All fidelity positions use the same teacher target prefix; late-prefix agreement is easier and does not establish free-running response quality. Blinded 0–2 tone/grounding reviews remain separate model-assisted ordinal judgments.

| Tone-cue method | Neu examples | CE | Semantic | Generated | EOS | Cap |
|---|---:|---:|---:|---:|---:|---:|
| predicted_tone | 64 | 0.68655 | 0.9080769470892847 | 64 | 64 | 0 |
| oracle_tone_reuse | 64 | 0.68655 | 0.9080769470892847 | 64 | 64 | 0 |

This is the fixed Neu generation panel, not the larger CE panel above. Predicted cues use a TRAIN-only acoustic classifier. Privileged intended-label cues are an information-complete reference for these synthetic annotations. When predicted and intended labels match exactly, the oracle output is explicitly reused; it adds no inference runtime. Plain ASR remains a words-only comparator. Cue labels are not verified natural emotion perception.
