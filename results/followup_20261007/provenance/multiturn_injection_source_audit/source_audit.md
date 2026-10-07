# Multi-turn injection source audit

CPU/read-only audit of frozen inference 04eb486 and the completed 4775 parent panel. No Qwen model was instantiated, no CUDA work ran, and no original result/source was changed.

## Exact source identity

The following local bytes match SHA256 obtained directly from the frozen node archive:

- `speech_projector/llm.py`: `d21a7e2be349de9818ebe276786fcce0cd75dec4d85cf3ec7e0a310d87366e03`
- `speech_projector/prompts.py`: `ccd727fa60adf0715b08b8f31a6dbaa453706c9d9461b7037534d53b78e0171e`
- `speech_projector/projectors.py`: `cc4a2fc91d71a59c9e2ca31e53b7dd882fc6e6da6d8fae45e96ef231936f222c`
- `speech_projector/overnight_conversation_execution.py`: `48be2afee1884491ab67b1317ee5f7b24cac44acc2bd0756be95abf1bb5ad30c`
- `speech_projector/overnight_conversation.py`: `e8a7e429d67141d26cfa0211e8530ab21d43bb3ef0a813101f4322ba08967ab0`
- `scripts/prepare_overnight_conversation.py`: `6228ac1f85345a7e9a20d86b0c1c3b2ec7567c75dd30972453c29e702c899e03`

## Actual neutral inputs

| Case | Seconds | Valid Whisper frames | 10 Hz pseudo-tokens | Feature SHA256 |
| --- | ---: | ---: | ---: | --- |
| followup_short | 3.78 | 189 | 38 | `33921f63b1cc0f78d00e5545624f682b8bd795b2804eadfab9bf04a056dcc5e1` |
| followup_next_step | 3.56 | 178 | 36 | `51c7055c040f5a408d5b5e959f9d89471990e7ea54a38a150b30c4b577a53eae` |
| followup_stop | 4.28 | 214 | 43 | `c65e5487f113427b73c81690c9888b2f26595e88a58fbb40ef6df0d3306999b5` |

## Assembly and retention checks

- Fixture preparation preserves the exact ordered hash-bound inputs `(short, next_step, stop)`; the unused summary clip is not substituted. All six original cue examples have empty prior histories.
- Execution constructs `(initial, short, next_step, stop)` and passes `speeches[turn_index]` as the current input. Thus turn 2 uses next-step and turn 3 uses stop, including all three controlled-history conditions. Rollout turn 1 uses short. Initial speech is retained only as an earlier USER turn.
- Each historical USER speech body is bracketed by `<|im_start|>user\n` and `<|im_end|>\n`; each preceding ASSISTANT text has its own matching role/end delimiters. Current USER speech is appended once, followed by `<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`.
- Conversation generation is exact-length batch 1, with an all-ones mask over the complete assembled prompt. It does not call the separately implemented left-padding batch helper. That helper pads embeddings and mask on the same left side; valid speech rows are not masked out.
- Whisper final states are cropped to ceil(duration×50), so the inspected tensors contain only the recorded valid frames. Mean pooling averages the final partial group by its actual count. No 30-second padding rows are injected as conversational pseudo-tokens.
- All 60 completed parent requests fit the same strict pre-generation budget reconstruction: maximum 6 historical turns and 237 history positions, versus recorded limits 6 and 8192. The driver raises before generation if either limit would truncate; generic wrapper budgeting can crop requests, but this panel's explicit guard prevents that path.
- Every saved current-case identifier agrees with its turn-index feature selection. Controlled modes hold the current audio and assistant text fixed. Omit-initial removes the first USER+ASSISTANT pair; rollout uses its actual earlier generated assistant replies.
- Each call freshly assembles the conversation and lets Qwen maintain its hybrid decode state within that generation. No manually reused session KV/recurrent cache can substitute a previous current utterance here.
- Raw fixture bytes were independently SHA-verified against the node. Its canonical semantic hash was checked on Linux under frozen04; Windows Path serialization differs, so local reserialization is not used as that proof.

## Interpretation limit

No reproducible feature-index, role/delimiter, validity-mask, or silent history-truncation bug was found in this audit. This does not establish why next-step/closure responses fail. Retained speech histories differ from training's earlier text-history layout; acoustic alignment, lexical grounding, out-of-distribution conversational layout, or model behavior remain possible explanations. The text-history control still has current follow-up audio, so it does not remove current-audio understanding as a limitation.
