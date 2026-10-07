# Verified model and embedding contract

CPU-only audit on 7 October 2026; no GPU models were instantiated. Values below come from the backed-up pinned configurations/tokenizer, safetensors headers, implementation text and local CPU projector parameter counting.

## Frozen text model

- Checkpoint: `Qwen/Qwen3.5-2B`, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- The checkpoint's outer configuration is `qwen3_5` / `Qwen3_5ForConditionalGeneration`; the pipeline deliberately loads the **text-only** `Qwen3_5TextConfig` (`model_type=qwen3_5_text`) with `Qwen3_5ForCausalLM`. Vision tensors are not part of the speech pipeline.
- Hidden/embedding width: **2,048**. Vocabulary: **248,320**. Input embedding table: `[248320, 2048]`, **508,559,360 parameters**, about 1.017 GB in BF16. `tie_word_embeddings=true`: the language-model output projection shares this table; do not count it twice.
- Safetensors header count under `model.language_model.*`: **1,881,825,088** parameter elements, including the input table and excluding vision tensors. This is a header audit, not a newly instantiated model count.
- **24 text layers:** 18 linear-attention Gated DeltaNet layers and 6 full-attention layers, with full attention every fourth layer. Full-attention heads: 8 query / 2 KV, head dimension 256. Linear attention: 16 key/value heads, 128-dimensional key/value heads, convolution kernel 4. Configured maximum positions: 262,144; this is not evidence of tested long-context speech quality.
- Actual node Transformers 5.13 source creates `DynamicCache(config=self.config)`. Linear cache layers hold `conv_states` and `recurrent_states`; full-attention layers hold attention K/V. A generic KV tuple or dropping trailing K/V does not restore the linear recurrent state.

## Trainable 10 Hz interface

`[T,768]` frozen encoder states → per-frame LayerNorm → consecutive mean pooling by 5 → Linear 768→1024 → GELU → Linear 1024→2048. **2,888,192 trainable parameters**, including biases and LayerNorm. The final partial pooling group averages its actual frames. The trainable interface uses FP32 parameters; its output is cast to the frozen Qwen input-embedding dtype, BF16 on the node.

Qwen receives `inputs_embeds=[batch, sequence, 2048]`, comprising optional system text, bounded previous textual turns, the user delimiter, current speech pseudo-token vectors and the assistant suffix. No current transcript or delivery label enters the speech-input branch. Previous user turns are text in the training format; retained speech history is an explicitly evaluated inference variation.

## Exact pinned chat delimiters

| Literal string | Token IDs |
|---|---|
| `<\|im_start\|>user\n` | `[248045,846,198]` |
| `<\|im_start\|>assistant\n` | `[248045,74455,198]` |
| `<\|im_start\|>system\n` | `[248045,8678,198]` |
| `<\|im_end\|>\n` | `[248046,198]` |
| `<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n` | `[248046,198,248045,74455,198,248068,271,248069,271]` |

Tokenize delimiter strings with `add_special_tokens=False`. The empty thinking block selects answer-only formatting. `ChatPromptConfig` inserts **no system message**; each Example may instead carry its actual cohort-specific generic system policy. Actual delivery metadata remains teacher-only. Tokenizer EOS is `<|im_end|>` **248046**; padding is `<|endoftext|>` **248044**. The checkpoint text config's EOS is 248044, so generation explicitly uses the tokenizer's 248046 to stop at the assistant message end.

The controlled configurations retain the last 2 text turns within 256 history tokens, train assistant targets up to 4,097 tokens including EOS, and evaluate greedy generations capped at 256 new tokens with completion/cap evidence. Scoring right-pads to a multiple of 64, masks padding/history/speech from loss, and predicts the first target from its immediately preceding assistant-prefix position. Generation left-pads prompts and enables cache; scoring disables cache.

## Frozen speech states and padding

Whisper Small revision `973afd24965f72e36ca33b3055d56a652f456b4d`: 12 encoder layers, width **768**, 16 kHz mono waveform, 80 mel bins and 160-sample mel hop (100 Hz). The second encoder convolution strides by 2, giving **50 hidden states/second**, or 20 ms per state. Cache selects the **final encoder `last_hidden_state` after final LayerNorm**.

The extractor pads/truncates to the standard 30-second / 3,000-mel-frame window; the encoder returns 1,500 states and attends its padded window. The code does not pass an encoder attention mask that removes that padding. It subsequently retains `min(1500, ceil(actual_waveform_samples/16000*50))` states, preserving the actual duration and dropping padded positions from downstream compression. Cache tensors are contiguous CPU BF16 `[valid_frames,768]`: **76,800 payload bytes/audio-second** (75 KiB), plus serialization overhead. Whisper is bidirectional, so partial-utterance states are not immutable prefixes suitable for append-only Qwen caching.

## Evidence paths

- `results_preview/overnight_40k_20261007/replay/models/qwen_15852e8c16360a2fea060d615a32b45270f8a8fc/{config.json,tokenizer.json,tokenizer_config.json,model.safetensors-00001-of-00001.safetensors}`.
- `results_preview/overnight_40k_20261007/replay/models/whisper_973afd24965f72e36ca33b3055d56a652f456b4d/{config.json,preprocessor_config.json}`.
- `speech_projector/{llm.py,prompts.py,projectors.py,cache.py,decoding.py}` and `artifacts/followup10hz/fullpass_job.json`.
- Read-only node SDK: `/venv/main/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py` (lines 433–462,1177) and `/venv/main/lib/python3.12/site-packages/transformers/cache_utils.py` (LinearAttentionLayer / conv and recurrent state).
- Primary pinned configuration: https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json
- Primary runtime implementation: https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py
