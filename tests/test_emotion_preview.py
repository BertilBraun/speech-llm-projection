"""CPU-only preview plan, completion, and waveform persistence checks."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import soundfile
from numpy.typing import NDArray
from pydantic import TypeAdapter, ValidationError

from scripts.package_results import ModelRevision
from speech_projector.emotion_preview import (
    TTS_MODEL,
    TTS_REVISION,
    Delivery,
    PreviewCapFailure,
    PreviewFailure,
    PreviewManifest,
    PreviewPlan,
    PreviewProvenance,
    PreviewTermination,
    SynthesizedAudio,
    codec_termination,
    default_preview_plan,
    normalize_waveform,
    persist_clip,
    plan_digest,
    render_preview,
)


def waveform() -> NDArray[np.float32]:
    return np.linspace(-0.5, 0.5, 2400, dtype=np.float32)


def test_exact_matrix_and_seeded_separate_instructions() -> None:
    plan = default_preview_plan()
    assert plan.model_name == TTS_MODEL
    assert plan.revision == TTS_REVISION
    assert plan.speaker == "Ryan" and plan.language == "English"
    assert len(plan.cases) == 10
    assert tuple(case.seed for case in plan.cases) == tuple(range(42, 52))
    assert tuple(case.delivery for case in plan.cases[:5]) == tuple(Delivery)
    assert {case.text for case in plan.cases} == {"I'm good.", "That went really well."}
    assert len({case.instruct for case in plan.cases}) == 5
    assert "sarcastic" not in plan.cases[-1].text.lower()
    assert "sarcasm" in plan.cases[-1].instruct.lower()
    assert PreviewPlan.model_validate_json(plan.model_dump_json()) == plan
    assert plan_digest(plan) == plan_digest(default_preview_plan())


@pytest.mark.parametrize("remove_case,wrong_seed", [(True, False), (False, True)])
def test_invalid_matrix_rejected(remove_case: bool, wrong_seed: bool) -> None:
    plan = default_preview_plan()
    cases = plan.cases[:-1] if remove_case else plan.cases
    if wrong_seed:
        cases = (cases[0].model_copy(update={"seed": 7}), *cases[1:])
    with pytest.raises(ValidationError):
        PreviewPlan(cases=cases)


@pytest.mark.parametrize(
    "frames,expected",
    [
        (20, PreviewTermination.STOPPED_BEFORE_CAP),
        (254, PreviewTermination.STOPPED_BEFORE_CAP),
        (255, PreviewTermination.CAP_AMBIGUOUS),
        (256, PreviewTermination.CAP_AMBIGUOUS),
    ],
)
def test_conservative_codec_cap(frames: int, expected: PreviewTermination) -> None:
    assert codec_termination(frames, 256) == expected


@pytest.mark.parametrize("frames,cap", [(-1, 256), (4, 1)])
def test_invalid_cap_arguments(frames: int, cap: int) -> None:
    with pytest.raises(ValueError):
        codec_termination(frames, cap)


@pytest.mark.parametrize(
    "audio",
    [
        np.empty(0, dtype=np.float32),
        np.zeros(20, dtype=np.float32),
        np.ones((4, 2), dtype=np.float32),
        np.array([0.1, np.nan], dtype=np.float32),
        np.array([0.1, np.inf], dtype=np.float32),
    ],
)
def test_invalid_audio_rejected(audio: NDArray[np.float32]) -> None:
    with pytest.raises(ValueError):
        normalize_waveform(audio)


def test_float_wav_is_exact_and_hash_matches(tmp_path: Path) -> None:
    plan = default_preview_plan()
    audio = waveform()
    audio[0] = 1.05
    clip = persist_clip(tmp_path, plan, plan.cases[0], SynthesizedAudio(audio, 24000, 8, 0.7))
    recovered, sample_rate = soundfile.read(tmp_path / clip.audio.path, dtype="float32")
    assert sample_rate == clip.sample_rate == 24000
    np.testing.assert_array_equal(recovered, audio)
    assert clip.duration_seconds == 0.1
    assert clip.peak_amplitude == pytest.approx(1.05)
    assert (
        clip.audio.sha256 == hashlib.sha256((tmp_path / clip.audio.path).read_bytes()).hexdigest()
    )
    assert clip.audio.bytes == (tmp_path / clip.audio.path).stat().st_size
    assert (tmp_path / "clips/good_neutral.json").exists()
    with pytest.raises(ValueError, match="overwrite"):
        persist_clip(tmp_path, plan, plan.cases[0], SynthesizedAudio(audio, 24000, 8, 0.7))


def test_cap_failure_has_no_exported_audio(tmp_path: Path) -> None:
    plan = default_preview_plan()
    failure = PreviewCapFailure(
        case=plan.cases[0], codec_tokens=255, max_new_tokens=256, runtime_seconds=0.3
    )
    assert TypeAdapter(PreviewFailure).validate_json(failure.model_dump_json()) == failure
    with pytest.raises(ValueError, match="cap"):
        persist_clip(tmp_path, plan, plan.cases[0], SynthesizedAudio(waveform(), 24000, 255, 0.3))
    assert not (tmp_path / "audio").exists()


def test_complete_manifest_and_readable_index(tmp_path: Path) -> None:
    plan = default_preview_plan()
    clips = tuple(
        persist_clip(tmp_path, plan, case, SynthesizedAudio(waveform(), 24000, 8, 0.1))
        for case in plan.cases
    )
    provenance = PreviewProvenance(
        source_commit="test-source-commit",
        model_revision=ModelRevision(
            model_name=TTS_MODEL, snapshot_revisions=(TTS_REVISION,), main_revision=TTS_REVISION
        ),
        input_sha256=plan_digest(plan),
        started_at=datetime.now(timezone.utc),
        qwen_tts_version="test",
        torch_version="test",
        transformers_version="test",
    )
    manifest = PreviewManifest(
        plan=plan, clips=clips, provenance=provenance, runtime_seconds=1.0, peak_vram_gb=0.0
    )
    assert PreviewManifest.model_validate_json(manifest.model_dump_json()) == manifest
    index = render_preview(manifest)
    assert all(case.case_id in index for case in plan.cases)
    assert index.count("[Listen]") == 10
    assert "raw EOS was not observed" in index
    with pytest.raises(ValidationError, match="ten cases"):
        PreviewManifest(
            plan=plan,
            clips=clips[:-1],
            provenance=provenance,
            runtime_seconds=1.0,
            peak_vram_gb=0.0,
        )


def test_preview_records_are_frozen_and_extra_forbidden() -> None:
    plan = default_preview_plan()
    with pytest.raises(ValidationError):
        plan.cases[0].text = "altered"
    with pytest.raises(ValidationError):
        PreviewPlan.model_validate_json(plan.model_dump_json()[:-1] + ',"unexpected":1}')
