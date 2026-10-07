# Dataset and hardware facts for the follow-up

Source is the verified prior replay; inputs/supervision remain unchanged.

| Cohort | Split | Examples | Audio mean/median/p90/max s | User words mean/median/p90/max | Teacher words mean/median/p90/max |
|---|---|---:|---|---|---|
| ordinary | train | 19999 | 7.190/6.987/9.325/27.115 | 20.242/20.000/25.000/74.000 | 143.525/108.000/277.000/841.000 |
| ordinary | validation | 512 | 7.206/7.011/9.316/16.181 | 20.730/21.000/25.000/40.000 | 155.215/108.000/334.500/670.000 |
| ordinary | test | 512 | 7.338/7.135/9.712/18.239 | 20.705/21.000/25.000/46.000 | 142.895/100.500/311.200/571.000 |
| qwen_emotional | train | 9094 | 6.510/6.080/9.200/27.280 | 15.332/15.000/19.000/26.000 | 19.809/19.000/29.000/52.000 |
| qwen_emotional | validation | 460 | 6.532/6.160/8.808/16.320 | 15.643/15.000/20.000/28.000 | 19.680/18.000/29.000/46.000 |
| qwen_emotional | test | 440 | 6.849/6.320/9.760/17.840 | 15.545/15.000/19.000/25.000 | 19.932/19.000/28.000/44.000 |
| neu_emotional | train | 9100 | 5.308/5.140/6.760/26.440 | 13.605/13.000/17.000/23.000 | 23.009/22.000/31.000/72.000 |
| neu_emotional | validation | 460 | 5.456/5.280/7.244/11.580 | 13.774/14.000/17.000/23.000 | 23.413/23.000/32.000/56.000 |
| neu_emotional | test | 440 | 5.403/5.320/7.002/9.340 | 13.945/14.000/17.000/23.000 | 23.284/23.000/31.000/48.000 |

## Actual panels

- fixed validation selection: ordinary 48 examples.
- fixed validation selection: qwen_emotional 40 examples.
- fixed validation selection: neu_emotional 40 examples.
- prepared fixed test selection bank: ordinary 96 examples.
- prepared fixed test selection bank: qwen_emotional 80 examples.
- prepared fixed test selection bank: neu_emotional 80 examples.
- prior completed primary test CE panel: ordinary 96 examples.
- prior completed primary test CE panel: qwen_emotional 80 examples.
- prior completed primary test CE panel: neu_emotional 80 examples.
- prior completed primary test generation panel: ordinary 48 examples.
- prior completed primary test generation panel: qwen_emotional 24 examples.
- prior completed primary test generation panel: neu_emotional 64 examples.

## Ordinary clean-source construction

Reference: audio_cleaned_text, documented synthesis input; not independently transcribed audio.
Material original-turn versus audio-original lexical differences and lexical TTS substitutions were removed before Qwen teacher supervision was generated. The preparation's original targets were placeholders; current training targets come from its later completed teacher_examples.jsonl.

| Split | Candidate examples | Selected | Dialogues | Material alignment exclusions | Lexical substitution exclusions | Missing synthesis text | Collision dialogues excluded |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 30000 | 20000 | 15931 | 562 | 62 | 0 | 0 |
| validation | 5647 | 512 | 445 | 106 | 12 | 0 | 692 |
| test | 5128 | 512 | 432 | 83 | 8 | 0 | 642 |

TRAIN: first requested clean examples in original 30k seeded order; nested prefixes. Heldout: original seed-42 dialogue/hash splits and example order, lexical alignment and TTS substitutions removed; whole dialogues with cross-split path/pair or exact teacher prompt (cleaned user plus role/text history[-2:]) collisions excluded. Original data.build_examples test-pair exclusion also applies to the pool.

Exclusion counters describe candidate pools, not selected examples. Alignment and substitution counts may overlap. Collision counts are whole dialogues in each candidate pool. A later combined-copy normalized prompt audit removed one additional TRAIN dialogue/example; original source assets remain preserved. Cleaned synthesis text is a documented generation reference, not an independently verified waveform transcript; occasional synthesis/ASR anomalies remain possible.

## Cache and training targets

- Raw native cache: 41,017 files / 20,785,399,565 bytes (20.785 GB decimal); 269,645.498 audio s.
- Incremental extraction: 19,994 new files / 118,406.920 audio s; synchronized encoder scope 127.245 s, overall cache wall 379.745 s, nested heldout ASR 12.160 s.
- Final normalized Whisper Small states: 768 dimensions, 50 Hz, BF16, 76,800 bytes/audio second before serialization overhead. Encoder attends standard 30-second padded log-mels; retained ceil(actual seconds×50) states define downstream masks.
- One full training pass: 4,314,521 target tokens including EOS over 38,193 examples; maximum stored target 1260 tokens, configured cap 4097. Target token modes/quantiles per cohort are unknown in saved evidence.
- Ordinary history: TRAIN 7251 empty/12748 two-turn; VAL 202/310; TEST 184/328. Both emotional cohorts have zero history turns.
- Old paired identical targets: 112 exact/125 normalized of 4997 remaining pairs; Neu 93 exact/98 normalized of 5000 pairs. Equality is diagnostic, not a rejection criterion.
- Excluded only in combined copy: 3 over-30-second old pairs = 6 clips, and 1 ordinary TRAIN dialogue/example protecting normalized test-prompt overlap. Original files preserved.

## Point-in-time hardware observation

At 2026-10-07T09:22:02+00:00: NVIDIA GeForce RTX 3090, 24576 MiB VRAM, driver 550.107.02.
Workspace filesystem capacity 214,748,364,800 bytes (200.0 GiB); free 44,051,517,440 bytes (44.052 GB).
Container limit 48,412,753,920 bytes (45.088 GiB), from exact `/sys/fs/cgroup/memory/memory.limit_in_bytes`. Charged usage 41,957,748,736 bytes includes filesystem cache.
Host counters separately: total 50,430,664,704 bytes, available 44,175,462,400 bytes; swap 0. Do not add host available memory to the container allocation limit.
PyTorch allocated peaks, reserved memory and NVIDIA/NVML observed usage are different measurements; current facts do not substitute one for another.

## Limits and scope

- No frozen-model inference, audio transfer, new labels or training in this audit.
- Target token counts reuse actual Transformers 5.13.0/tokenizers 0.22.2 exposure evidence; per-cohort token quantiles/modes were not recorded and are unknown here.
- Native cache bytes include reused ordinary plus newly extracted emotional files; do not count all files as new extraction.
- Encoder extraction, cache wall and ASR timers have nested scopes; do not add them.
- Hardware values are point observations, not training peaks or allocated rental duration.
- Intended synthetic delivery labels are not human-verified emotional gold.
- Neu classifier and Neu audio are one Paul voice; vocoder/prosody signatures may help classification and do not establish natural-speech or speaker generalization.
- The production classifier has four synthetic intended classes: happy, sad, angry and fearful. It has no neutral/OOD class or calibrated abstention; perfect synthetic-label performance does not establish natural-emotion readiness.
- Teacher targets are model-generated; role/physical-action/tone mistakes remain preserved.
- Current ordinary inputs were selected after material alignment/substitution filters; historical raw V0 known mismatch rows are not deliberately carried into this clean ordinary cohort. Cleaning counters need not be mutually exclusive and must not be summed as unique excluded examples.
- Exact cross-cohort design families share one split; domains and pragmatic intents are intentionally reused, so these are not unseen-domain or unseen-intent tests.
- Prepared test bank and actual primary generation panel have separate denominators; the full-pass follow-up has now executed the same fixed TEST panel: 256 CE examples and 136 primary speech generations.

## Hashed source evidence

- `results_preview\overnight_40k_20261007\replay\provenance\combined_dataset\summary.json`: 12,892 bytes; SHA256 `5142c8294791ee3e9137f392830b56e49e5aa06c4817a2c5ca2f6e2f4fa261d7`.
- `results_preview\overnight_40k_20261007\replay\speech-projector\data_overnight_20261006\cache_stats_38193.json`: 963 bytes; SHA256 `aeaf8560fc43ef83673d7789cb9fcbfad3cba3342ac2b21d967b3ee861e2e208`.
- `results_preview\overnight_40k_20261007\replay\provenance\checkpoint_operations\token_exposure_and_continuation_audit.md`: 6,280 bytes; SHA256 `7be84fdcb889aa4c236ccc5743298b20d0fff5473fe719d0af21defd405e8aa4`.
- `results_preview\overnight_40k_20261007\replay\speech-projector\data_overnight_20261006\sources.jsonl`: 9,604,728 bytes; SHA256 `8471f5cde9cdd2d5c3f721d5829fe3ec6380bd81c5c21eeba35eda6262e19ae9`.
- `results_preview\overnight_40k_20261007\replay\speech-projector\data_teacher\preparation.json`: 744,656 bytes; SHA256 `b2c6da3e6202044617c6d744a1ab32ccc35eac1127c3ea26c9f92cbf51c6c23c`.
- `results_preview\overnight_40k_20261007\replay\speech-projector\data_overnight_20261006\fixed_validation.jsonl`: 163,209 bytes; SHA256 `6e4aafe88eb94f5944341ae60f65e6cf75a476f484aa38a6b9a74726c2a79397`.
- `results_preview\overnight_40k_20261007\replay\speech-projector\data_overnight_20261006\fixed_test.jsonl`: 320,823 bytes; SHA256 `a67964172fc7648c5780184c88a2021f69d078d0046737cb201183669107eb23`.
- `results_preview\overnight_40k_20261007\replay\speech-projector\results_overnight_20261006\overnight_mean_2p5hz_updates4000_epoch1\test\evaluation_losses.jsonl`: 61,114 bytes; SHA256 `e9cff05073044951a17f0000e1b6343039b52d6738ba48b4787878c03b165b87`.
- `results_preview\overnight_40k_20261007\replay\speech-projector\results_overnight_20261006\overnight_mean_2p5hz_updates4000_epoch1\test\evaluation_generations.jsonl`: 283,882 bytes; SHA256 `9ab7be29e463a99c6affa0f1b9540d4ac1d0298314467004000e751aa5df0537`.
