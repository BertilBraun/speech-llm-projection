from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch
from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration

from speech_projector.cache import (
    AudioFeatureInput,
    CacheConfig,
    extract_additional_audio,
    load_feature,
    save_encoder_feature,
)


@pytest.fixture
def whisper() -> tuple[WhisperForConditionalGeneration, WhisperFeatureExtractor]:
    configuration = WhisperConfig(
        d_model=8,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
    )
    model = WhisperForConditionalGeneration(configuration).eval()
    model.requires_grad_(False)
    return model, WhisperFeatureExtractor()


def test_shared_feature_crop_preserves_uncompressed_bfloat_states(tmp_path: Path) -> None:
    states = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    path = tmp_path / "features.pt"
    assert save_encoder_feature(states, 1281, 50, path) == 5
    actual = load_feature(path)
    torch.testing.assert_close(actual, states[:5].to(torch.bfloat16))
    assert not path.with_suffix(".part").exists()


def test_real_audio_hook_resumes_verified_features_without_training_examples(
    tmp_path: Path, whisper: tuple[WhisperForConditionalGeneration, WhisperFeatureExtractor]
) -> None:
    model, extractor = whisper
    audio = tmp_path / "followup.wav"
    soundfile.write(audio, np.zeros(1600, dtype=np.float32), 16000)
    inputs = (AudioFeatureInput(audio, tmp_path / "followup.pt"),)
    configuration = CacheConfig(root=tmp_path, train_examples=1, device="cpu")
    result = extract_additional_audio(configuration, inputs, model, extractor, 50)
    assert result.extracted_count == 1
    assert result.entries[0].valid_frames == 5
    assert result.entries[0].audio_seconds == pytest.approx(0.1)
    assert load_feature(inputs[0].feature_path).shape == (5, 8)
    assert not (tmp_path / "examples.jsonl").exists()
    assert not (tmp_path / "asr_transcripts.jsonl").exists()
    original_statistics = (tmp_path / "additional_audio_cache.json").read_bytes()
    resumed = extract_additional_audio(configuration, inputs, model, extractor, 50)
    assert resumed.extracted_count == 0
    assert resumed.extraction_seconds == 0
    assert resumed.entries == result.entries
    assert (tmp_path / "additional_audio_cache.json").read_bytes() == original_statistics
    inputs[0].feature_path.write_bytes(b"changed cache bytes")
    with pytest.raises(ValueError, match="differs"):
        extract_additional_audio(configuration, inputs, model, extractor, 50)


def test_additional_audio_rejects_cropping_long_clips_and_duplicate_destinations(
    tmp_path: Path, whisper: tuple[WhisperForConditionalGeneration, WhisperFeatureExtractor]
) -> None:
    model, extractor = whisper
    audio = tmp_path / "too_long.wav"
    soundfile.write(audio, np.zeros(30 * 16000 + 1, dtype=np.float32), 16000)
    inputs = (AudioFeatureInput(audio, tmp_path / "unused.pt"),)
    configuration = CacheConfig(root=tmp_path, train_examples=1, device="cpu")
    with pytest.raises(ValueError, match="30-second"):
        extract_additional_audio(configuration, inputs, model, extractor, 50)
    with pytest.raises(ValueError, match="unique"):
        extract_additional_audio(configuration, inputs * 2, model, extractor, 50)
    assert not inputs[0].feature_path.exists()
