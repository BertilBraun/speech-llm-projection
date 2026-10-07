# Speech projector serving handoff for the Rust scheduler

## Current checkpoint: 7 October 2026

The latest research selection is **`followup_mean_10hz_ce_control_6775`**, trained
on the 38,193-example mixed dataset including paired emotional audio. Use its
final `checkpoint/projector.safetensors`, SHA256
`ae0f59349e30d049b863b5174fe0e93d9bb4fcd899d448a96c460854f85b9e3a`.
The local asset is under
`results_preview/followup10hz_20261007/runs/followup_mean_10hz_ce_control_6775/`;
the node asset is under
`/workspace/speech-projector/results_followup_20261007/followup_mean_10hz_ce_control_6775/`.
The projector dimensions, audio handling and chat delimiters below still apply.
The older 20,000-example weights are historical. See the [main results](../README.md)
for measured grounding, emotional and multi-turn limitations. No production
server, vLLM speech-path parity, persistent session cache or request-latency
benchmark has been established. The classifier alternative also requires
natural-speaker validation before use with arbitrary microphone input.

Current evaluation uses **greedy decoding with a 256-new-token cap**. The mixed
model's examples supply their own generic `prompt` policy: ordinary examples use
chat without a system message; older Qwen emotion examples use the one-to-two
short-sentence policy; Neu examples use the exact four-delivery generic policy
quoted in the main README. These are not the current clip's tone label.
`RunConfig.prompt=chat` alone does not reproduce all emotional evaluations.
Select and record a serving policy explicitly; changing it is an untested
comparison. The older no-system/sampling settings below describe the historical
20k prototype.

The existing model entry point is
`FrozenQwen.generate_conversation(ConversationRequest(identifier, prompt, history, current), max_new_tokens)`.
It accepts typed text or speech current turns and typed history. Its retained
`SpeechHistoryTurn` mode was evaluated as an unfamiliar inference layout, not
trained or validated as persistent-session cache parity. Request assembly starts
a fresh prefill; the decoder's internal cache is used within that request.

## Historical handoff: 6 October 2026

Prepared 6 October 2026 for integration with a separately implemented Rust scheduler. The current model consumes a **complete user utterance**, then generates assistant text. Accept audio packets throughout the utterance, finalize them on the external end-of-turn event, and perform one logical Qwen prefill followed by cached autoregressive decoding. Text output can be streamed after that prefill.

The implemented inference reference is PyTorch plus Hugging Face Transformers. A persistent Python model worker is the simplest initial integration. vLLM is a promising faster Qwen backend with an embedding-input interface, but the **speech-projector path has not yet been validated in vLLM**. This handoff describes the existing model and a proposed serving contract; it does not deploy a server or resume training.

## Historical model and checkpoint

The original serving prototype used this completed Qwen-teacher distillation checkpoint:

| Component | Actual interface |
| --- | --- |
| Speech encoder | `openai/whisper-small`, frozen, final encoder hidden states |
| Encoder output | `[ceil(audio_seconds × 50), 768]`, cropped from the padded 30-second encoder output |
| Projector | LayerNorm(768), mean pooling in blocks of 5, Linear(768,1024), GELU, Linear(1024,2048) |
| Projector size | 2,888,192 parameters; saved FP32 checkpoint about 11.6 MB |
| Speech pseudo-tokens | Approximately 10 per audio second; 5 seconds produces 50 embeddings |
| Qwen input width | 2,048, cast to the language model dtype |
| Language model | Frozen text backbone of `Qwen/Qwen3.5-2B`, BF16 on CUDA |
| Output | Assistant text tokens; this pipeline does not produce speech |

Local deployment assets:

- Config: `results_teacher/teacher_20000_mlp_10hz/config.json`.
- Final weights: `results_teacher/teacher_20000_mlp_10hz/checkpoint/projector.safetensors`.
- Best monitored weights: `results_teacher/teacher_20000_mlp_10hz/best_projector.safetensors`; the selected step was also 5,000. Both local files currently have SHA256 `018209b9ca336619cfd695e40693c621ac8455d6146509042c1eab162455389b`. Record which file and hash are loaded.
- Node equivalents are under `/workspace/speech-projector/`.
- Qwen revision: `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Whisper revision used in the existing cached node models: `973afd24965f72e36ca33b3055d56a652f456b4d`.
- Training source revision: `efc806ca97ed4c20de8fcedd536cc5d6c6a03a3d`.

Load the projector state with `safetensors.torch.load_file` into `Projector(config.projector)`. Preserve FP32 projector computation, LayerNorm epsilon `1e-5` and exact GELU; cast its output to Qwen BF16. Load the Qwen text class, not its vision tower: the reference uses `Qwen3_5TextConfig.from_pretrained` and `Qwen3_5ForCausalLM.from_pretrained`. The deployment manifest should pin both model revisions, tokenizer files, projector hash, prompt policy, dtype and decoding policy. The research wrapper currently loads model names without an explicit revision argument; a serving adapter should resolve pinned snapshots.

This checkpoint learned lexical conditioning from speech. It has **not** been trained on the newly generated paired emotional dataset. Intended emotion labels and teacher responses in that dataset do not establish emotional performance of this checkpoint.

## Runtime ownership

```text
Rust scheduler
  session and turn IDs, audio transport, ordered buffers, end-of-turn,
  admission control, deadlines, cancellation, output routing
        |
        | completed audio + prior text history, or current text
        v
Persistent Python GPU worker
  audio preprocessing -> Whisper -> projector -> Qwen prompt embeddings
  Qwen prefill -> cached token decoding -> incremental text events
```

Keep model weights resident. Use `.eval()` and `torch.inference_mode()` during serving; disable training gradient checkpointing in the serving configuration. The current `FrozenQwen` methods return completed responses, so token streaming and cancellation require a small serving adapter. They are not existing network endpoints.

The Rust process does not need to link LibTorch or implement Qwen kernels. Use a local Unix socket, gRPC stream, or another typed IPC boundary to a Python worker. Choose one transport for the scheduler project. Binary audio payloads avoid JSON/base64 overhead on every microphone packet. GPU tensors and runtime cache objects stay inside the model worker.

The Qwen3-TTS and vLLM-Omni processes used to **generate the dataset** are separate from this inference pipeline. They are not needed to understand microphone speech. A future output-TTS stage can consume assistant text events separately.

## Proposed scheduler contract

The following event names are a proposed contract, not implemented APIs:

| Input event | Required information | Behavior |
| --- | --- | --- |
| `StartAudioTurn` | Session ID, unique turn ID, audio format, sample rate, channel count, prior text history, decoding policy | Allocate an ordered per-turn buffer and record the model/prompt version |
| `AudioChunk` | Turn ID, increasing chunk index, sample count, binary PCM | Buffer samples; acknowledge accepted position; do not start assistant decoding |
| `CommitAudioTurn` | Turn ID, final chunk index, final sample count | Check completeness, freeze the buffer, submit one inference request |
| `TextTurn` | Session/turn IDs, current literal user text, prior text history, decoding policy | Bypass Whisper/projector and use ordinary Qwen embeddings |
| `CancelTurn` | Turn ID | Abort pending or active work, release resources, suppress subsequent output |

Outputs should distinguish `TurnAccepted`, `TextDelta`, `TurnFinished` and `TurnFailed`. `TextDelta` includes the turn ID, increasing output sequence number and a UTF-8 delta. The final event reports `eos`, `token_limit` or `cancelled`, token count and stage timings. Failures have a stable error code and explanation. Token-limit termination must not be reported as a naturally finished answer.

For duplicate commits, return the existing request state instead of launching duplicate GPU work. Do not commit a buffer with missing chunks. Audio arriving after commit belongs to a new turn or is rejected according to the transport contract. Barge-in should cancel the old turn and fence all output by turn ID so late tokens cannot reach the new conversation turn.

If the transport sends Opus or another compressed format, decode it before the model boundary and include that cost in latency measurements. Packet boundaries are transport details, not model token boundaries.

## Audio preprocessing and limits

The reference preprocessing in `speech_projector/data.py::load_audio` reads float32 samples, averages channels to mono, and resamples with `scipy.signal.resample_poly` to **16,000 Hz**. Its waveform is a finite, nonempty one-dimensional float32 array. Integer PCM must be converted to floating-point amplitude consistently, for example signed PCM16 divided by 32768. Do not peak-normalize every packet; that would alter acoustic dynamics and create packet-dependent inputs.

For resampling streamed input, preserve resampler state across packets or concatenate the full waveform and resample once at commit. Calling a stateless resampler independently for each packet changes boundary samples and can accumulate length differences. Preserve the actual number of samples after resampling.

At commit:

1. Produce the contiguous mono 16 kHz waveform and its actual duration.
2. Use `WhisperFeatureExtractor` with `padding="max_length"`, `sampling_rate=16000` and the standard 30-second window.
3. Run only Whisper's encoder. It yields `[1,1500,768]` for the padded window.
4. Retain the first `min(1500, ceil(num_samples / 16000 × 50))` states.
5. Apply the projector and construct the Qwen prompt.

The encoder follows standard Whisper padding behavior: it attends to the padded window; the current implementation does not exclude silent padding from encoder attention. Only the downstream retained sequence is cropped. Preserve this behavior for checkpoint parity.

**Support utterances up to 30 seconds initially.** The research example schema bounds duration at 30 seconds, and the extractor would otherwise truncate. Reject oversize turns explicitly, or have the product intentionally split them into separate turns. Concatenating independently encoded 30-second windows into one user message is an unvalidated extension.

At 10 Hz, `pseudo_token_count = ceil(ceil(duration_seconds × 50) / 5)`. Pooling averages an incomplete final block using its real frame count. Preserve the order **LayerNorm before pooling**, and do not pad the final block into a biased average. A 30-second turn produces at most 300 pseudo-tokens.

## Exact Qwen prompt layout and delimiters

The completed 20k checkpoint has `prompt.kind="chat"`: **no system message**. The code supports a system message, but adding the new concise emotional-teacher system prompt to this checkpoint changes its trained context. Make that an explicit, evaluated policy change rather than a hidden serving default.

The sequence passed to Qwen is:

```text
[optional configured system message embeddings]
[previous textual user/assistant message embeddings]
<|im_start|>user\n
[projected speech embeddings, or embedded current transcript tokens]
<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
[new assistant tokens generated autoregressively]
<|im_end|>
```

The brackets above describe tensors; their contents are **not literal prompt text**. Delimiter strings are tokenized with `add_special_tokens=False` and embedded using Qwen's own input embedding matrix. There is no learned audio-start token, audio-end token, or special microphone packet token in this training format. Projected vectors occupy the body of the current `user` message. The exact prefix IDs are `[248045,846,198]`; suffix IDs are `[248046,198,248045,74455,198,248068,271,248069,271]`.

The external end-of-turn event triggers appending the **existing suffix**, which closes the user message and opens the assistant message. There is no separate magic user-to-assistant token: the boundary is the sequence `<|im_end|>\n<|im_start|>assistant\n`, followed by the empty non-thinking block.

Pinned tokenizer identifiers, verified from its configuration:

| Marker | Token ID | Meaning here |
| --- | ---: | --- |
| `<|im_start|>` | 248045 | Message starts; `user`, `assistant`, `system` are tokenized role text after it |
| `<|im_end|>` | 248046 | Message ends; assistant generation stop token |
| `<think>` | 248068 | Start of the empty reasoning block supplied in the suffix |
| `</think>` | 248069 | End of that block |
| `<|endoftext|>` | 248044 | Tokenizer padding token |

**Stop on tokenizer EOS 248046.** The checkpoint's text-model config lists EOS 248044, which differs from the tokenizer. An earlier evaluation bug used that value and allowed invented follow-on roles. The current `generation_parameters` explicitly uses `tokenizer.eos_token_id`. Preserve that override in any runtime migration. Newline characters are part of the template and must also be preserved. [Pinned tokenizer configuration](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/tokenizer_config.json).

## Model call and different input kinds

Existing canonical interfaces are `SpeechInput(embeddings: Tensor)` and `TranscriptInput(text: str)` in `speech_projector/inputs.py`. Speech embeddings must have shape `[speech_tokens,2048]`. The current transcript is never inserted into the speech branch, and emotion labels are not inference inputs. Project each valid, unpadded feature sequence before Qwen batch padding: the projector has no feature-mask argument.

The research wrapper exposes `generate(example, utterance)` and `generate_batch(examples, utterances, max_new_tokens)`. Its `Example` contains training-only target and file metadata. A serving adapter should accept the actual history, current input and decoding configuration directly, reusing the canonical `Turn` and input types, rather than requiring clients to fabricate training records or supply target responses.

For the initial forward pass, concatenate Qwen's embedded text prefix, projected speech and embedded suffix into `inputs_embeds` with shape `[batch,sequence,2048]`. Use a matching `[batch,sequence]` attention mask: 1 for real positions, 0 for batch padding. The current generation batch left-pads prompts to the longest sequence rounded to a multiple of 64; training scoring uses right padding. No padding positions should count as speech or history.

Conceptually, a worker does:

```text
prefill: Qwen(inputs_embeds=complete_prompt, attention_mask=mask, use_cache=True)
choose first assistant token from the final real prompt position's logits
decode: Qwen(input_ids=new_token, past_key_values=runtime_cache, use_cache=True)
repeat until tokenizer EOS, token budget or cancellation
```

This is an algorithm outline, not a complete manual decoding implementation. Prefer the model's generation/cache utilities first. A custom loop must maintain masks and position handling correctly, use the model's cache type, and sample each token exactly once. Token decoding also needs an incremental detokenizer: a token is not necessarily a complete UTF-8 word or character. Hugging Face `generate()` already caches during decoding; it does not re-prefill the entire conversation for every generated token.

Text input bypasses both audio models. An ASR baseline passes Whisper's recognized text into the same transcript branch. ASR is optional for that baseline or for constructing earlier textual history; the speech-projector current-turn path does not require Whisper's transcription decoder.

## Why audio packets cannot be incrementally committed to Qwen

Whisper Small's encoder is bidirectional. Its earlier hidden states can change when later audio arrives. The full-window feature extraction also need not produce the same mel values as independent packet extraction. An unfinished pooling block changes when more frames join it. Consequently, projected embeddings from a partial utterance are not guaranteed to be an immutable prefix of the final embeddings.

Appending those changing representations to Qwen's cache would preserve stale states and give different inputs from training. Re-encoding a longer buffer and appending its entire projection would additionally duplicate earlier audio. A Qwen runtime's chunked-prefill feature splits **an already fixed prompt** for scheduling; it does not solve this acoustic revision problem.

The serving contract should therefore use one **logical complete-utterance prefill** at end-of-turn. Its implementation may internally divide the fixed prompt into chunks. CPU buffering and resampling can happen during capture. Exact static text-prefix prefill may also overlap capture if the worker safely snapshots or owns that prefix cache, but speech embeddings still require finalization. Provisional audio inference would need full invalidation/recomputation and separate accuracy/latency validation; it is not a supported append-only streaming mode.

## Cache ownership and conversation history

The current wrapper uses `use_cache=False` for teacher-forced scoring and `use_cache=True` for generation. Each generation request starts from a newly built prompt; it does **not** expose a persistent conversation-cache API. Rust should retain an opaque request/session handle, not allocate or serialize model cache tensors.

Qwen3.5-2B is hybrid: 24 layers comprise 6 full-attention and 18 linear-attention layers. Full-attention layers retain K/V tensors; linear layers retain convolution and recurrent state. A generic all-layer K/V tuple or a Rust implementation that only crops attention tensors is insufficient. Use the runtime's Qwen-compatible cache implementation. [Pinned model configuration](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json), [Transformers 5.13 Qwen implementation](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py).

For scale intuition, BF16 K/V alone grows by approximately `6 × 2 × 2 × 256 × 2 = 12,288` bytes per token per sequence, about 12 MiB for 1,024 positions. This calculation excludes recurrent/convolution state, allocator overhead, padding and runtime reservations. It is not a total VRAM estimate. Count speech pseudo-tokens in the sequence budget just like text positions.

**Multi-turn history is a separate design decision.** Training used the last two previous turns as text, with a maximum of 256 history tokens including message delimiters. The helper walks from newest to oldest and retains the end of each selected message as the remaining budget permits. Persisting previous speech pseudo-tokens indefinitely would change that layout and the context-length distribution.

For exact initial parity, rebuild the bounded textual history and create a fresh request for each completed utterance. This requires previous user text from the application or optional background ASR of an earlier turn. The current user audio must still enter only as projected speech. If no user transcripts are available, choose explicitly between an empty-history mode, an evaluated summary policy, or a future model trained/evaluated with retained speech history. Do not silently treat earlier speech embeddings as equivalent to the training text history.

Cache reuse is valid only for an **identical ordered prefix** under the same model, projector, prompt, dtype and position policy. Changing the system prompt, truncating history, replacing speech with transcript text, editing a prior turn, or cancelling a partially retained answer requires a fresh prefill or a valid saved prefix snapshot. Recurrent state cannot generally be undone by dropping some trailing K/V positions. The safe initial cancellation policy is to discard the active request cache and rebuild the next request from committed history.

Automatic prefix caching across requests is an optimization distinct from cached decoding within a request. Start with correct fresh prefills; enable prefix reuse only after testing embedding-content identity and the hybrid runtime's behavior. Never reuse a speech cache merely because two requests have the same pseudo-token count or placeholder token IDs.

## Faster runtime integration

vLLM 0.28.0 is already installed in an isolated node environment and has executed Qwen3.5-2B **text** generation successfully. For speech, keep Whisper and the projector in PyTorch and pass their output into vLLM's prompt-embedding interface.

The pinned release documents `enable_prompt_embeds=True` for engine usage and `--enable-prompt-embeds` for serving. Offline `prompt_embeds` has shape `[sequence_length,hidden_size]`. Its Completions API accepts embeddings for the **entire already-templated prompt**. Chat Completions can instead interleave content-only embedding parts with ordinary text; the server applies the chat template. Do not supply a fully templated sequence to the latter. The online encoding uses a base64-encoded Torch tensor, so a Python adapter is preferable to inventing Torch serialization in Rust. [vLLM 0.28 prompt embedding documentation](https://github.com/vllm-project/vllm/blob/v0.28.0/docs/features/prompt_embeds.md).

For first migration, complete-prompt embeddings offer the clearest parity with the existing wrapper. Check that Qwen's runtime implementation accepts this path, preserves the empty-thinking suffix, handles hybrid cache state, and honors EOS 248046. The released renderer moves prompt embeddings to CPU for process serialization, so include those transfers in profiling rather than assuming an entirely GPU-resident handoff. [Pinned vLLM renderer](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/renderers/base.py).

Let vLLM manage batching, request cache allocation and reclamation. Rust manages admission and application state; it does not need manual KV memory caching. Reserve GPU memory for Whisper/projector when setting vLLM's memory budget. Keep this environment isolated from the validated training environment.

A small projector export to ONNX or another runtime is technically straightforward, but the LLM dominates decoding and the projector is only 2.89 million parameters. Moving it first is unlikely to be the most useful optimization. A fully Rust/Candle, GGUF or TensorRT deployment would need proven support for this exact Qwen hybrid architecture, arbitrary input embeddings and the matching chat/cache behavior. No such complete migration has been validated here. Quantizing or replacing Qwen also changes the embedding interface that the projector learned and requires parity/quality evaluation.

## Latency estimates and measured evidence

No live microphone-to-first-token benchmark has been run. For an idle, warmed RTX 3090, a 3–10-second utterance and short bounded history, use **0.3–1.5 seconds from end-of-turn to first text** as an initial engineering budget, not a measured guarantee. Provisional planning allocations are 0.1–0.8 seconds for final audio preprocessing/Whisper, below 0.05 seconds for projection, and 0.05–0.5 seconds for Qwen prefill/first decode. These allocations are guesses to guide profiling, not evidence-derived stage timings; queueing, final-packet delivery and turn detection are additional costs.

Actual saved measurements on the node:

| Measurement | Result | Interpretation |
| --- | --- | --- |
| Transformers text profile, serial batch 1 | 31.2 generated tokens/s over 16 replies | Useful initial decoding reference; not speech TTFT |
| Warm Transformers text batch 64 | 1,120 aggregate tokens/s | Throughput across many requests, not one user's token rate |
| Warm vLLM text batch 64 | 11,891 tokens in 5.42 s, 2,190 aggregate tokens/s | Different output workload; not a controlled speech speedup or TTFT comparison |
| Trained speech checkpoint, validation batch 32 | 84,011 tokens in 220 s, about 381 aggregate tokens/s | Cached features were used; Whisper encoding, projection before queuing, and network transport excluded |
| Whisper feature-cache extraction | 10,000 examples in 67.2 encoder-timed seconds, batch 8 | Offline batch throughput; not a 6.7 ms single-request claim |

At a provisional 20–60 tokens/s single-session budget, a 30–60-token reply takes about **0.5–3 seconds after the first token**. Treat optimized-runtime latency as unknown until the complete speech route is measured. Text streams can begin immediately after the first token; waiting for a full reply is unnecessary.

Cold startup is substantially slower: a saved vLLM Qwen initialization took about 150 seconds including initialization/compilation, and some Transformers shapes showed seconds of first-use kernel overhead. Start and warm workers before declaring them ready. Keep readiness separate from process liveness.

Measure timestamps for commit received, last packet available, preprocessing finished, encoder finished, projector finished, prefill finished, first token emitted and final token emitted. Report p50/p95 TTFT, inter-token latency, completed-turn latency, queue time, utterance duration, history length, pseudo-token count, concurrency and NVIDIA-observed VRAM. For isolated GPU stage measurements, synchronize appropriately; synchronizing every production token would distort serving latency.

## Integration acceptance checks

Before production scheduling, compare the adapter against the saved PyTorch reference using fixed audio/features and transcript inputs:

1. Identical audio decoding/resampling, valid encoder length, projector weights and tensor order.
2. Identical tokenized prefix/suffix, no current transcript or emotion-label leakage in the speech route, and the correct EOS override.
3. Same prompt attention masks and next-token logits within documented BF16 tolerance. Greedy outputs should be compared, while allowing investigated near-tie numeric differences rather than silently changing sampling.
4. EOS and token-cap events distinguished; ordered incremental detokenization and cancellation tested.
5. Two sessions never share mutable cache state; history changes and barge-in cannot emit stale output.
6. Real speech TTFT and peak memory measured with the intended concurrency, including encoder/runtime co-residency.

The historical trained config used sampling with temperature 1, top-p 1, top-k 20 and presence penalty 2, with a very large 4,096-token cap. Preserve it for reproduction; use explicit, separately versioned serving settings for latency-controlled shorter replies. Presence-penalty behavior must agree on whether it applies only to generated tokens: the reference implementation applies it only to generated tokens when using `inputs_embeds`.

## Code and measurement references

- `speech_projector/data.py::load_audio`: current waveform conversion.
- `speech_projector/cache.py::extract_features`: exact Whisper preprocessing and retained-state rule.
- `speech_projector/projectors.py::Projector`: normalization, compression and projection.
- `speech_projector/prompts.py`: exact chat strings.
- `speech_projector/inputs.py`: speech and transcript input types.
- `speech_projector/llm.py::FrozenQwen`: embedding assembly, masks and generation.
- `speech_projector/decoding.py`: explicit EOS/cache/sampling configuration.
- `results_teacher/teacher_20000_mlp_10hz/config.json` and `result.json`: actual trained deployment candidate.
- `results_teacher/profiling/teacher_profile.json` and `teacher_warm_profile.json`: measured Transformers text profiles.
- `results_preview/emotional_dataset/qwen_smoke_offline/warm_sixty_four.json`: measured vLLM text workload.
- `results/dataset/cache_stats_20000.json`: historical Whisper extraction throughput.

Immediate scheduler implementation target: ordered audio buffering, external end-of-turn, a complete-utterance worker request, streamed text events, request-owned runtime cache and cancellation. Incremental acoustic prefill and persistent speech-history caches require separate model validation.
