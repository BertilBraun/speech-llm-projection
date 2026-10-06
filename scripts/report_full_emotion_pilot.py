"""Verify full-emotion galleries and distinguish measured batch throughput from latency."""

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile
from pydantic import BaseModel, ConfigDict, Field

from scripts.prepare_full_emotion_pilot import INDEX_EMOTIONS, NEU_EMOTIONS
from scripts.report_tts_pilot import ModelAudit, ModelResults, verify_results
from speech_projector.neutts_batch_benchmark import (
    BatchMeasurement,
    NeuTtsBenchmarkResult,
    verify_recorded_audio,
)
from speech_projector.tts_pilot import PilotTermination, TtsPilotManifest, TtsPilotResult


class FullEmotionReportConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    index_beams3_directory: Path
    index_beams1_directory: Path
    neu_directory: Path
    index_manifest: Path
    neu_manifest: Path
    actual_index_source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    benchmark_directories: tuple[Path, ...] = ()
    output: Path


@dataclass(frozen=True)
class EmotionRun:
    label: str
    manifest: TtsPilotManifest
    model: ModelResults
    audit: ModelAudit


def load_run(label: str, manifest: TtsPilotManifest, directory: Path) -> EmotionRun:
    result = TtsPilotResult.model_validate_json((directory / "result.json").read_bytes())
    model = ModelResults(directory=directory, result=result)
    return EmotionRun(
        label=label, manifest=manifest, model=model, audit=verify_results(manifest, model)
    )


def verify_benchmark(
    result: NeuTtsBenchmarkResult, directory: Path, manifest: TtsPilotManifest
) -> None:
    for measurement in result.measurements:
        for evidence in measurement.clips:
            if (
                not (directory / evidence.clip.audio_path)
                .resolve()
                .is_relative_to(directory.resolve())
            ):
                raise ValueError("Benchmark waveform path escapes its result directory")
    verify_recorded_audio(result, directory)
    saved_manifest = TtsPilotManifest.model_validate_json((directory / "cases.json").read_bytes())
    if saved_manifest != manifest:
        raise ValueError("Benchmark manifest differs from the serial Neu comparison")
    expected_cases = {case.case_id: case for case in manifest.cases}
    groups: defaultdict[tuple[int, int], list[BatchMeasurement]] = defaultdict(list)
    for measurement in result.measurements:
        groups[(measurement.requested_batch_size, measurement.repetition)].append(measurement)
        if not measurement.clips or len(measurement.clips) > measurement.requested_batch_size:
            raise ValueError("Benchmark has an empty or oversized actual batch")
        for evidence in measurement.clips:
            clip = evidence.clip
            if expected_cases.get(clip.case.case_id) != clip.case:
                raise ValueError("Benchmark clip differs from the canonical case manifest")
            if clip.termination != PilotTermination.STOP:
                raise ValueError("Benchmark contains an unresolved token-limit completion")
            waveform, sample_rate = soundfile.read(
                directory / clip.audio_path, dtype="float32", always_2d=True
            )
            if waveform.shape[1] != 1 or not np.isfinite(waveform).all() or not np.any(waveform):
                raise ValueError("Benchmark waveform is not finite non-silent mono audio")
            if np.max(np.abs(waveform)) > 1 or sample_rate != clip.sample_rate:
                raise ValueError("Benchmark waveform amplitude or sample rate differs")
            if not math.isclose(len(waveform) / sample_rate, clip.audio_seconds):
                raise ValueError("Benchmark waveform duration differs from recorded evidence")
    expected_groups = {
        (batch_size, repetition)
        for batch_size in result.configuration.batch_sizes
        for repetition in range(result.configuration.repeats)
    }
    if set(groups) != expected_groups:
        raise ValueError("Benchmark is missing requested batch-size/repetition groups")
    for measurements in groups.values():
        identifiers = tuple(
            item.clip.case.case_id for group in measurements for item in group.clips
        )
        if len(identifiers) != len(expected_cases) or set(identifiers) != set(expected_cases):
            raise ValueError("Benchmark pass has missing or duplicated cases")
        if sorted(item.batch_index for item in measurements) != list(range(len(measurements))):
            raise ValueError("Benchmark pass has missing or duplicated batch indices")


def benchmark_table(result: NeuTtsBenchmarkResult) -> list[str]:
    lines = [
        "| Requested batch | Passes | Completed clips | Total wall s | Total audio s | "
        "Throughput RTF | Clips/s | Batch completion range s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for batch_size in result.configuration.batch_sizes:
        selected = tuple(
            item for item in result.measurements if item.requested_batch_size == batch_size
        )
        wall_seconds = sum(item.end_to_end_seconds for item in selected)
        audio_seconds = sum(item.audio_seconds for item in selected)
        clips = sum(len(item.clips) for item in selected)
        if not selected or wall_seconds <= 0 or audio_seconds <= 0:
            raise ValueError("No positive measured benchmark throughput for a requested batch size")
        minimum = min(item.end_to_end_seconds for item in selected)
        maximum = max(item.end_to_end_seconds for item in selected)
        lines.append(
            f"| {batch_size} | {len({item.repetition for item in selected})} | {clips} | "
            f"{wall_seconds:.3f} | {audio_seconds:.3f} | {wall_seconds / audio_seconds:.3f} | "
            f"{clips / wall_seconds:.3f} | {minimum:.3f}–{maximum:.3f} |"
        )
    return lines


def render_report(
    configuration: FullEmotionReportConfig,
    runs: tuple[EmotionRun, ...],
    benchmarks: tuple[NeuTtsBenchmarkResult, ...],
) -> str:
    lines = [
        "# Full emotion listening pilot and throughput measurements",
        "",
        "One fixed sentence, intended delivery labels and a fixed Paul reference. "
        "Index uses eight native components; Neu uses seven public SDK labels. "
        "Afraid/fearful retain their distinct API names; Neu has no melancholic result. "
        "These are listening examples, not verified emotion accuracy or a training run.",
        "",
        "Serial RTF = total measured generation time / total generated audio duration. "
        "Generation is CUDA-synchronized and excludes WAV writes; startup, reference preparation "
        "and warmup are separate. Index reference encoding is inside warmup; its reference "
        "column zero is not a claim of free conditioning. "
        "Startup scopes differ in first CUDA setup.",
        "",
        "| Serial configuration | States | Generation s | Audio s | Weighted RTF | "
        "Startup s | Reference s | Warmup s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        audit = run.audit
        lines.append(
            f"| {run.label} | {audit.clip_count} | {audit.generation_seconds:.3f} | "
            f"{audit.audio_seconds:.3f} | {audit.aggregate_real_time_factor:.3f} | "
            f"{audit.startup_seconds:.3f} | {audit.reference_preparation_seconds:.3f} | "
            f"{audit.warmup_generation_seconds:.3f} |"
        )
    lines.extend(
        [
            "",
            "Index beam comparisons share the full-emotion sentence, vector intensity "
            "and reference; compare quality by listening as well as speed. The original four-tone "
            "pilot used a different intensity and two sentences, "
            "so its times are a separate experiment.",
            "",
            "## Source and interpretation",
            "",
            f"Actual executed Index source supplied from the archived common tree: "
            f"`{configuration.actual_index_source_commit}`. The input/result source fields below "
            "are preserved unchanged even if they predate this executed-source correction. "
            "Keep the full common tree/source archive and logs with the outputs.",
            "",
            "Direct Index infer does not apply the WebUI normalize_emo_vec adjustment: configured "
            "one-hot intensity 1.0 is passed unchanged, including surprise and calm. "
            "This is a control description, not perceived intensity verification.",
            "",
        ]
    )
    for run in runs:
        result = run.model.result
        lines.extend(
            [
                f"**{run.label}**: model `{result.model_repository}` revision "
                f"`{result.model_revision}`; recorded source field `{result.source_commit}`.",
                "",
                f"Backend: {result.backend_description}",
                "",
                f"Speaker: {result.speaker_description}",
                "",
            ]
        )
    lines.extend(
        [
            "Reference cloning does not guarantee identical voices. Public SDK termination "
            "NOT_EXPOSED stays unknown; do not infer codec EOS from a returned waveform. "
            "Every gallery WAV hash, duration and case identity was checked. "
            "Emotion, naturalness and word preservation still require separate assessment.",
            "",
        ]
    )
    for result in benchmarks:
        lines.extend(
            [
                "## Neu backbone batch benchmark",
                "",
                "This is an explicit backbone batching implementation, not the public serial SDK "
                "baseline. Exact-length codec/watermark calls remain individual. "
                "Total batch wall time is counted once per batch; summing per-clip completion "
                "times would overcount. Whole-batch completion time is not time to first audio, "
                "nor independent per-request latency. "
                "Shared batch seeds and padding can change outputs.",
                "",
                f"Recorded source: `{result.configuration.source_commit}`; "
                f"helper SHA256 `{result.helper_sha256}`.",
                "",
                f"Startup {result.startup_seconds:.3f}s; reference "
                f"{result.reference_preparation_seconds:.3f}s; "
                f"warmup {result.warmup_seconds:.3f}s.",
                "",
                result.timing_scope,
                "",
                *benchmark_table(result),
                "",
            ]
        )
    if not benchmarks:
        lines.extend(
            [
                "Batch benchmark: no completed measurement was supplied; no throughput "
                "or latency estimate is inferred.",
                "",
            ]
        )
    lines.extend(["## Actual audio gallery", ""])
    for run in runs:
        lines.extend([f"### {run.label}", ""])
        for clip in run.model.result.clips:
            audio = (run.model.directory / clip.audio_path).resolve().as_posix()
            lines.extend(
                [
                    f"**{clip.case.emotion.value}** — {clip.case.case_id}",
                    "",
                    f"> {clip.case.text}",
                    "",
                    f"![{run.label} {clip.case.emotion.value}](<{audio}>)",
                    "",
                    f"{clip.generation_seconds:.3f}s generation / {clip.audio_seconds:.3f}s audio "
                    f"= RTF {clip.real_time_factor:.3f}; termination {clip.termination.value}.",
                    "",
                ]
            )
    return "\n".join(lines)


def report_full_emotions(configuration: FullEmotionReportConfig) -> Path:
    index_manifest = TtsPilotManifest.model_validate_json(configuration.index_manifest.read_bytes())
    neu_manifest = TtsPilotManifest.model_validate_json(configuration.neu_manifest.read_bytes())
    if (
        len(index_manifest.cases) != 8
        or len(neu_manifest.cases) != 7
        or {case.emotion for case in index_manifest.cases} != set(INDEX_EMOTIONS)
        or {case.emotion for case in neu_manifest.cases} != set(NEU_EMOTIONS)
    ):
        raise ValueError("Full-emotion report requires exactly eight Index and seven Neu states")
    texts = {case.text for manifest in (index_manifest, neu_manifest) for case in manifest.cases}
    if len(texts) != 1:
        raise ValueError("Full-emotion galleries must share one exact literal sentence")
    runs = (
        load_run("Index beams 3", index_manifest, configuration.index_beams3_directory),
        load_run("Index beams 1", index_manifest, configuration.index_beams1_directory),
        load_run("Neu public SDK serial", neu_manifest, configuration.neu_directory),
    )
    benchmarks = tuple(
        NeuTtsBenchmarkResult.model_validate_json((directory / "result.json").read_bytes())
        for directory in configuration.benchmark_directories
    )
    for directory, benchmark in zip(configuration.benchmark_directories, benchmarks, strict=True):
        verify_benchmark(benchmark, directory, neu_manifest)
    text = render_report(configuration, runs, benchmarks)
    configuration.output.mkdir(parents=True, exist_ok=True)
    path = configuration.output / "full_emotion_comparison.md"
    path.write_text(text, encoding="utf-8")
    (configuration.output / "report_config.json").write_text(
        configuration.model_dump_json(indent=2), encoding="utf-8"
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-beams3-directory", type=Path, required=True)
    parser.add_argument("--index-beams1-directory", type=Path, required=True)
    parser.add_argument("--neu-directory", type=Path, required=True)
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--neu-manifest", type=Path, required=True)
    parser.add_argument("--actual-index-source-commit", required=True)
    parser.add_argument("--benchmark-directory", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = FullEmotionReportConfig(
        index_beams3_directory=arguments.index_beams3_directory,
        index_beams1_directory=arguments.index_beams1_directory,
        neu_directory=arguments.neu_directory,
        index_manifest=arguments.index_manifest,
        neu_manifest=arguments.neu_manifest,
        actual_index_source_commit=arguments.actual_index_source_commit,
        benchmark_directories=tuple(arguments.benchmark_directory),
        output=arguments.output,
    )
    print(report_full_emotions(configuration))


if __name__ == "__main__":
    main()
