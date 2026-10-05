"""Reusable final Whisper encoder states and held-out ASR transcripts."""

import argparse
import math
import time
from pathlib import Path

import torch
from pydantic import Field
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer
from transformers.modeling_outputs import BaseModelOutput

from speech_projector.data import load_audio, load_examples
from speech_projector.models import AsrTranscript, Example, Record, Split


class CacheConfig(Record):
    root: Path
    train_examples: int = Field(gt=0)
    batch_size: int = Field(default=8, gt=0)
    model_name: str = "openai/whisper-small"
    device: str = "cuda"
    transcribe_heldout: bool = True


class CacheStatistics(Record):
    encoder_model: str
    hidden_dimension: int
    native_states_per_second: float
    selected_hidden_state: str
    storage_dtype: str
    bytes_per_audio_second: int
    feature_count: int
    extracted_count: int
    total_audio_seconds: float
    extracted_audio_seconds: float
    feature_bytes: int
    extraction_seconds: float
    cache_wall_seconds: float | None
    cache_examples_per_second: float | None
    extraction_audio_seconds_per_second: float
    asr_seconds: float
    peak_vram_gb: float
    masking: str


def load_feature(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def load_asr(path: Path) -> list[AsrTranscript]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [AsrTranscript.model_validate_json(line) for line in stream]


def select_examples(configuration: CacheConfig) -> list[Example]:
    manifest = configuration.root / "examples.jsonl"
    return (
        load_examples(manifest, Split.TRAIN, configuration.train_examples)
        + load_examples(manifest, Split.VALIDATION)
        + load_examples(manifest, Split.TEST)
    )


def extract_features(configuration: CacheConfig) -> CacheStatistics:
    examples = select_examples(configuration)
    transcripts_path = configuration.root / "asr_transcripts.jsonl"
    transcripts = load_asr(transcripts_path)
    transcribed = {transcript.example_id for transcript in transcripts}
    pending = [
        example
        for example in examples
        if not example.feature_path.exists()
        or (
            configuration.transcribe_heldout
            and example.split != Split.TRAIN
            and example.example_id not in transcribed
        )
    ]
    dtype = torch.bfloat16 if configuration.device == "cuda" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        configuration.model_name, dtype=dtype
    ).to(configuration.device)
    model.requires_grad_(False)
    model.eval()
    extractor = WhisperFeatureExtractor.from_pretrained(configuration.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(configuration.model_name)
    native_rate = (
        extractor.sampling_rate / extractor.hop_length / model.model.encoder.conv2.stride[0]
    )
    hidden_dimension = model.config.d_model
    assert hidden_dimension == 768
    assert native_rate == 50
    extraction_seconds = 0.0
    extracted_audio_seconds = 0.0
    asr_seconds = 0.0
    extracted_count = 0
    if configuration.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    cache_started = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, len(pending), configuration.batch_size):
            batch = pending[offset : offset + configuration.batch_size]
            audio = [load_audio(example.audio_path) for example in batch]
            inputs = extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length",
                return_attention_mask=True,
            )
            features = inputs.input_features.to(device=configuration.device, dtype=dtype)
            if configuration.device == "cuda":
                torch.cuda.synchronize()
            started = time.monotonic()
            outputs: BaseModelOutput = model.model.encoder(features, return_dict=True)
            if configuration.device == "cuda":
                torch.cuda.synchronize()
            extraction_seconds += time.monotonic() - started
            for index, example in enumerate(batch):
                if example.feature_path.exists():
                    continue
                valid_frames = min(
                    outputs.last_hidden_state.shape[1],
                    math.ceil(len(audio[index]) / 16000 * native_rate),
                )
                hidden = (
                    outputs.last_hidden_state[index, :valid_frames]
                    .to(device="cpu", dtype=torch.bfloat16)
                    .contiguous()
                )
                example.feature_path.parent.mkdir(parents=True, exist_ok=True)
                partial = example.feature_path.with_suffix(".part")
                torch.save(hidden, partial)
                partial.replace(example.feature_path)
                extracted_count += 1
                extracted_audio_seconds += len(audio[index]) / 16000
            asr_indices = [
                index
                for index, example in enumerate(batch)
                if configuration.transcribe_heldout
                and example.split != Split.TRAIN
                and example.example_id not in transcribed
            ]
            if asr_indices:
                started = time.monotonic()
                indices = torch.tensor(asr_indices, device=configuration.device)
                asr_outputs = BaseModelOutput(
                    last_hidden_state=outputs.last_hidden_state.index_select(0, indices)
                )
                generated = model.generate(
                    features.index_select(0, indices),
                    encoder_outputs=asr_outputs,
                    attention_mask=inputs.attention_mask.to(configuration.device).index_select(
                        0, indices
                    ),
                    language="en",
                    task="transcribe",
                    max_new_tokens=128,
                )
                text = tokenizer.batch_decode(
                    generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                if configuration.device == "cuda":
                    torch.cuda.synchronize()
                asr_seconds += time.monotonic() - started
                with transcripts_path.open("a", encoding="utf-8") as stream:
                    for index, transcript in zip(asr_indices, text, strict=True):
                        record = AsrTranscript(
                            example_id=batch[index].example_id, text=transcript.strip()
                        )
                        stream.write(record.model_dump_json() + "\n")
                        transcribed.add(record.example_id)
            print(
                f"Cached {min(offset + len(batch), len(pending))}/{len(pending)} "
                f"encoder_seconds={extraction_seconds:.1f} asr_seconds={asr_seconds:.1f}",
                flush=True,
            )
    statistics = CacheStatistics(
        encoder_model=configuration.model_name,
        hidden_dimension=hidden_dimension,
        native_states_per_second=native_rate,
        selected_hidden_state="final encoder last_hidden_state after final LayerNorm",
        storage_dtype="bfloat16",
        bytes_per_audio_second=hidden_dimension * 2 * int(native_rate),
        feature_count=len(examples),
        extracted_count=extracted_count,
        total_audio_seconds=sum(example.duration for example in examples),
        extracted_audio_seconds=extracted_audio_seconds,
        feature_bytes=sum(example.feature_path.stat().st_size for example in examples),
        extraction_seconds=extraction_seconds,
        cache_wall_seconds=time.monotonic() - cache_started,
        cache_examples_per_second=len(pending) / (time.monotonic() - cache_started),
        extraction_audio_seconds_per_second=extracted_audio_seconds / extraction_seconds
        if extraction_seconds
        else 0.0,
        asr_seconds=asr_seconds,
        peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9
        if configuration.device == "cuda"
        else 0.0,
        masking=(
            "Whisper standard log-mel padded/truncated to30s; encoder attends padded states "
            "as standard Whisper. Cache retains ceil(actual waveform duration*50) states; "
            "downstream compressor masks by actual cached sequence length. "
            "No transcript stored in features."
        ),
    )
    statistics_path = configuration.root / f"cache_stats_{configuration.train_examples}.json"
    statistics_path.write_text(statistics.model_dump_json(indent=2), encoding="utf-8")
    print(statistics.model_dump_json(indent=2), flush=True)
    return statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train-examples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-asr", action="store_true")
    arguments = parser.parse_args()
    extract_features(
        CacheConfig(
            root=arguments.root,
            train_examples=arguments.train_examples,
            batch_size=arguments.batch_size,
            transcribe_heldout=not arguments.no_asr,
        )
    )


if __name__ == "__main__":
    main()
