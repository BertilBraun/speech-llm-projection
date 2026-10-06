"""Check gallery integrity and batch timing denominators without external runtimes."""

from pathlib import Path

import numpy as np
import pytest
import soundfile

from scripts.neutts_pilot_state import NeuTtsPilotConfig, PilotDevice, digest, write_record
from scripts.prepare_neutts_models import NeuTtsPreparation, PinnedRepository
from scripts.report_full_emotion_pilot import (
    FullEmotionReportConfig,
    benchmark_table,
    load_run,
    render_report,
    verify_benchmark,
)
from speech_projector.neutts_batch_benchmark import (
    BatchClipEvidence,
    BatchMeasurement,
    NeuTtsBenchmarkConfig,
    NeuTtsBenchmarkResult,
)
from speech_projector.tts_pilot import (
    PilotEmotion,
    PilotTermination,
    TtsPilotCase,
    TtsPilotClip,
    TtsPilotManifest,
    TtsPilotResult,
)


@pytest.fixture
def benchmark(tmp_path: Path) -> tuple[NeuTtsBenchmarkResult, TtsPilotManifest]:
    cases = tuple(
        TtsPilotCase(
            case_id=emotion.value,
            utterance_id="same",
            text="The same literal sentence.",
            emotion=emotion,
            seed=42,
        )
        for emotion in (PilotEmotion.HAPPY, PilotEmotion.SAD)
    )
    manifest = TtsPilotManifest(cases=cases, warmup_text="Warm up.", warmup_seed=41)
    manifest_path = tmp_path / "cases.json"
    write_record(manifest_path, manifest)
    repositories = tuple(
        PinnedRepository(
            repository=name, revision="fixed", snapshot=tmp_path / name, download_seconds=None
        )
        for name in ("neuphonic/neutts-2e", "neuphonic/neucodec")
    )
    preparation = NeuTtsPreparation(
        source_commit="fixed",
        repositories=(repositories[0], repositories[1]),
        auxiliary_repositories=(),
        sdk_speakers=("paul",),
        sdk_emotions=("happy", "sad"),
    )
    pilot = NeuTtsPilotConfig(
        manifest=manifest_path,
        manifest_sha256=digest(manifest_path),
        output_directory=tmp_path,
        repository_directory=tmp_path / "repository",
        preparation=preparation,
        device=PilotDevice.CUDA,
    )
    clips: list[BatchClipEvidence] = []
    for index, case in enumerate(cases):
        path = tmp_path / f"{case.case_id}.wav"
        duration = 3 + index * 2
        soundfile.write(path, np.full(duration * 8000, 0.1, dtype=np.float32), 8000)
        clips.append(
            BatchClipEvidence(
                clip=TtsPilotClip(
                    case=case,
                    audio_path=path.name,
                    sha256=digest(path),
                    sample_rate=8000,
                    audio_seconds=duration,
                    generation_seconds=2,
                    real_time_factor=2 / duration,
                    termination=PilotTermination.STOP,
                ),
                prompt_tokens=100,
                speech_tokens=200,
                generated_token_ids=(10, 99),
            )
        )
    measurement = BatchMeasurement(
        requested_batch_size=2,
        repetition=0,
        batch_index=0,
        shared_batch_seed=42,
        padded_prompt_width=100,
        generation_token_cap=2000,
        frontend_seconds=0.1,
        backbone_seconds=1.5,
        codec_watermark_seconds=0.4,
        end_to_end_seconds=2,
        clips=tuple(clips),
    )
    result = NeuTtsBenchmarkResult(
        configuration=NeuTtsBenchmarkConfig(
            pilot=pilot, source_commit="fixed", repeats=1, batch_sizes=(2,)
        ),
        startup_seconds=1,
        reference_preparation_seconds=0.1,
        warmup_seconds=2,
        torch_version="test",
        transformers_version="test",
        neucodec_version="test",
        helper_sha256="test",
        reference_sha256="test",
        watermark_active=True,
        effective_max_new_tokens=2000,
        measurements=(measurement,),
        timing_scope="Synchronized batch elapsed time.",
    )
    return result, manifest


def test_report_counts_batch_time_once(
    tmp_path: Path, benchmark: tuple[NeuTtsBenchmarkResult, TtsPilotManifest]
) -> None:
    result, manifest = benchmark
    verify_benchmark(result, tmp_path, manifest)
    assert (
        benchmark_table(result)[2] == "| 2 | 1 | 2 | 2.000 | 8.000 | 0.250 | 1.000 | 2.000–2.000 |"
    )
    assert sum(item.clip.generation_seconds for item in result.measurements[0].clips) == 4


def test_faster_setting_gallery_uses_actual_batch_outputs_and_shared_latency(
    tmp_path: Path, benchmark: tuple[NeuTtsBenchmarkResult, TtsPilotManifest]
) -> None:
    result, _ = benchmark
    configuration = result.configuration.model_copy(update={"batch_sizes": (7,)})
    measurement = result.measurements[0].model_copy(update={"requested_batch_size": 7})
    result = result.model_copy(
        update={"configuration": configuration, "measurements": (measurement,)}
    )
    report_configuration = FullEmotionReportConfig(
        index_beams3_directory=tmp_path,
        index_beams1_directory=tmp_path,
        neu_directory=tmp_path,
        index_manifest=tmp_path / "cases.json",
        neu_manifest=tmp_path / "cases.json",
        actual_index_source_commit="a" * 40,
        benchmark_directories=(tmp_path,),
        output=tmp_path / "report",
    )
    report = render_report(report_configuration, (), (result,))
    assert "Actual batch-7 samples, first measured pass" in report
    assert "shared completion 2.000s; 8.000s total audio; throughput RTF 0.250" in report
    for evidence in measurement.clips:
        assert (tmp_path / evidence.clip.audio_path).resolve().as_posix() in report
    assert "not each clip's individual generation time" in report
    assert "generation /" not in report


def test_benchmark_validates_replica_pool_and_complete_pass(
    tmp_path: Path, benchmark: tuple[NeuTtsBenchmarkResult, TtsPilotManifest]
) -> None:
    result, manifest = benchmark
    originals = result.measurements[0].clips
    replica_clips = tuple(
        item.model_copy(
            update={
                "clip": item.clip.model_copy(
                    update={
                        "case": item.clip.case.model_copy(
                            update={
                                "case_id": f"{item.clip.case.case_id}_replica",
                                "utterance_id": f"{item.clip.case.case_id}_replica",
                            }
                        )
                    }
                )
            }
        )
        for item in originals
    )
    pool = manifest.model_copy(
        update={"cases": manifest.cases + tuple(item.clip.case for item in replica_clips)}
    )
    write_record(tmp_path / "cases.json", pool)
    repeated = result.measurements[0].model_copy(update={"batch_index": 1, "clips": replica_clips})
    result = result.model_copy(update={"measurements": result.measurements + (repeated,)})
    verify_benchmark(result, tmp_path, manifest)
    assert "| 2 | 1 | 4 | 4.000 | 16.000 | 0.250" in benchmark_table(result)[2]
    incomplete = result.model_copy(update={"measurements": result.measurements[:1]})
    with pytest.raises(ValueError, match="missing or duplicated cases"):
        verify_benchmark(incomplete, tmp_path, manifest)
    changed_case = pool.cases[-1].model_copy(update={"text": "A different sentence."})
    write_record(
        tmp_path / "cases.json",
        pool.model_copy(update={"cases": pool.cases[:-1] + (changed_case,)}),
    )
    with pytest.raises(ValueError, match="not complete replicas"):
        verify_benchmark(result, tmp_path, manifest)


@pytest.mark.parametrize("failure", ("duplicate", "cap", "corrupt"))
def test_benchmark_rejects_incomplete_or_corrupt_evidence(
    tmp_path: Path,
    benchmark: tuple[NeuTtsBenchmarkResult, TtsPilotManifest],
    failure: str,
) -> None:
    result, manifest = benchmark
    measurement = result.measurements[0]
    first = measurement.clips[0]
    match failure:
        case "duplicate":
            changed = measurement.model_copy(update={"clips": (first, first)})
            result = result.model_copy(update={"measurements": (changed,)})
            message = "missing or duplicated cases"
        case "cap":
            changed_clip = first.clip.model_copy(
                update={"termination": PilotTermination.TOKEN_LIMIT}
            )
            changed = measurement.model_copy(
                update={
                    "clips": (first.model_copy(update={"clip": changed_clip}), measurement.clips[1])
                }
            )
            result = result.model_copy(update={"measurements": (changed,)})
            message = "unresolved token-limit"
        case "corrupt":
            (tmp_path / first.clip.audio_path).write_bytes(b"corrupt")
            message = "waveform hash"
    with pytest.raises(ValueError, match=message):
        verify_benchmark(result, tmp_path, manifest)


def test_serial_gallery_accepts_unknown_eos_but_rejects_missing_clip(
    tmp_path: Path, benchmark: tuple[NeuTtsBenchmarkResult, TtsPilotManifest]
) -> None:
    result, manifest = benchmark
    clips = tuple(
        item.clip.model_copy(update={"termination": PilotTermination.NOT_EXPOSED})
        for item in result.measurements[0].clips
    )
    serial = TtsPilotResult(
        model_repository="test/model",
        model_revision="fixed",
        source_commit="original",
        backend_description="public SDK",
        speaker_description="Paul reference",
        startup_seconds=1,
        reference_preparation_seconds=0,
        warmup_generation_seconds=2,
        clips=clips,
        failures=(),
    )
    write_record(tmp_path / "result.json", serial)
    assert load_run("serial", manifest, tmp_path).audit.clip_count == 2
    write_record(tmp_path / "result.json", serial.model_copy(update={"clips": clips[:1]}))
    with pytest.raises(ValueError):
        load_run("serial", manifest, tmp_path)
