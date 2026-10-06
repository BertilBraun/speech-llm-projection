"""Verify full-emotion galleries and distinguish measured batch throughput from latency."""

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile
from pydantic import BaseModel, ConfigDict, Field, model_validator

from scripts.neutts_pilot_state import digest
from scripts.prepare_full_emotion_pilot import INDEX_EMOTIONS, NEU_EMOTIONS
from scripts.report_tts_pilot import ModelAudit, ModelResults, verify_results
from speech_projector.neutts_batch_benchmark import (
    BatchMeasurement,
    NeuTtsBenchmarkResult,
    validate_benchmark_manifest,
    verify_recorded_audio,
)
from speech_projector.tts_pilot import (
    PilotEmotion,
    PilotTermination,
    TtsPilotClip,
    TtsPilotManifest,
    TtsPilotResult,
)


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


class ClipAmplitudeAudit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    clip: TtsPilotClip
    verified_path: Path
    waveform_subtype: str
    sample_count: int = Field(gt=0)
    peak_absolute: float = Field(gt=0)
    samples_above_full_scale: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_sample_count(self) -> "ClipAmplitudeAudit":
        if self.samples_above_full_scale > self.sample_count:
            raise ValueError("Above-full-scale count exceeds total waveform samples")
        return self

    @property
    def fraction_above_full_scale(self) -> float:
        return self.samples_above_full_scale / self.sample_count


def audit_amplitude(clip: TtsPilotClip, directory: Path) -> ClipAmplitudeAudit:
    path = (directory / clip.audio_path).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Waveform path escapes its result directory")
    if digest(path) != clip.sha256:
        raise ValueError("Saved waveform hash differs from record")
    waveform, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
    if waveform.shape[1] != 1 or not np.isfinite(waveform).all() or not np.any(waveform):
        raise ValueError("Waveform is not finite non-silent mono audio")
    if sample_rate != clip.sample_rate:
        raise ValueError("Waveform sample rate differs from recorded evidence")
    if not math.isclose(len(waveform) / sample_rate, clip.audio_seconds):
        raise ValueError("Waveform duration differs from recorded evidence")
    information = soundfile.info(path)
    peak = float(np.max(np.abs(waveform)))
    if peak > 1 and information.subtype not in ("FLOAT", "DOUBLE"):
        raise ValueError("Non-floating waveform has an invalid full-scale amplitude")
    return ClipAmplitudeAudit(
        clip=clip,
        verified_path=path,
        waveform_subtype=information.subtype,
        sample_count=len(waveform),
        peak_absolute=peak,
        samples_above_full_scale=int(np.count_nonzero(np.abs(waveform) > 1)),
    )


def amplitude_note(audit: ClipAmplitudeAudit) -> str:
    return (
        f"Raw {audit.waveform_subtype} peak {audit.peak_absolute:.6f}; "
        f"samples above full scale {audit.samples_above_full_scale}/{audit.sample_count} "
        f"({audit.fraction_above_full_scale:.8%})."
    )


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
) -> tuple[ClipAmplitudeAudit, ...]:
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
    expected_content = Counter((case.emotion, case.text, case.seed) for case in manifest.cases)
    saved_content = Counter((case.emotion, case.text, case.seed) for case in saved_manifest.cases)
    replicas, remainder = divmod(len(saved_manifest.cases), len(manifest.cases))
    if (
        remainder
        or not replicas
        or set(saved_content) != set(expected_content)
        or any(saved_content[key] != count * replicas for key, count in expected_content.items())
        or saved_manifest.warmup_text != manifest.warmup_text
        or saved_manifest.warmup_seed != manifest.warmup_seed
    ):
        raise ValueError("Benchmark pool is not complete replicas of the serial Neu comparison")
    expected_cases = {case.case_id: case for case in saved_manifest.cases}
    groups: defaultdict[tuple[int, int], list[BatchMeasurement]] = defaultdict(list)
    amplitudes: list[ClipAmplitudeAudit] = []
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
            amplitudes.append(audit_amplitude(clip, directory))
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
    return tuple(amplitudes)


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
    amplitudes: tuple[ClipAmplitudeAudit, ...],
) -> str:
    amplitude_by_path = {item.verified_path: item for item in amplitudes}
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
    overshoots = tuple(item for item in amplitudes if item.samples_above_full_scale)
    lines.extend(
        [
            "## Raw waveform amplitude audit",
            "",
            f"Verified {len(amplitudes)} WAVs; {len(overshoots)} have samples outside [-1, 1]. "
            "FLOAT WAV preserves these raw samples; no originals were normalized or regenerated. "
            "Above-full-scale samples are a playback/conversion clipping risk, not evidence that "
            "the stored floating waveform is already clipped. Timings are unchanged. "
            "Full per-clip peaks/counts are in amplitude_audit.jsonl, including every repeat.",
            "",
        ]
    )
    if overshoots:
        lines.extend(
            [
                "| Raw clip path | Peak absolute | Above full scale / samples | Fraction |",
                "| --- | ---: | ---: | ---: |",
                *(
                    f"| {item.verified_path.as_posix()} | {item.peak_absolute:.9f} | "
                    f"{item.samples_above_full_scale}/{item.sample_count} | "
                    f"{item.fraction_above_full_scale:.8%} |"
                    for item in overshoots
                ),
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
    for directory, result in zip(configuration.benchmark_directories, benchmarks, strict=True):
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
                "The larger pool repeats the same sentence/emotion controls with unique case IDs; "
                "it measures bulk throughput, not additional linguistic or emotion diversity. "
                "Repeated timing passes reset the seed and are not independent quality samples.",
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
        gallery_batch_size = max(result.configuration.batch_sizes)
        batch_gallery = tuple(
            item
            for item in result.measurements
            if item.requested_batch_size == gallery_batch_size and item.repetition == 0
        )
        if batch_gallery:
            lines.extend(
                [
                    f"### Actual batch-{gallery_batch_size} samples, first measured pass",
                    "",
                    "These are the faster configuration's actual sampled outputs. "
                    "The shared whole-batch completion time below is not each clip's "
                    "individual generation time. Throughput RTF is reported separately above; "
                    "listen to these samples before drawing quality conclusions. "
                    "One first-occurring clip per emotion is shown; every replica is retained.",
                    "",
                ]
            )
            shown_emotions: set[PilotEmotion] = set()
            for measurement in batch_gallery:
                lines.extend(
                    [
                        f"Batch {measurement.batch_index}: shared completion "
                        f"{measurement.end_to_end_seconds:.3f}s; "
                        f"{measurement.audio_seconds:.3f}s total audio; "
                        f"throughput RTF {measurement.aggregate_real_time_factor:.3f}.",
                        "",
                    ]
                )
                for evidence in measurement.clips:
                    clip = evidence.clip
                    if clip.case.emotion in shown_emotions:
                        continue
                    shown_emotions.add(clip.case.emotion)
                    audio_path = (directory / clip.audio_path).resolve()
                    audio = audio_path.as_posix()
                    lines.extend(
                        [
                            f"**{clip.case.emotion.value}** — {clip.case.case_id}",
                            "",
                            f"> {clip.case.text}",
                            "",
                            f"![Neu batch {gallery_batch_size} "
                            f"{clip.case.emotion.value}](<{audio}>)",
                            "",
                            f"Audio {clip.audio_seconds:.3f}s; termination "
                            f"{clip.termination.value}; batch seed "
                            f"{measurement.shared_batch_seed}.",
                            amplitude_note(amplitude_by_path[audio_path]),
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
            audio_path = (run.model.directory / clip.audio_path).resolve()
            audio = audio_path.as_posix()
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
                    amplitude_note(amplitude_by_path[audio_path]),
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
    amplitudes = tuple(
        audit_amplitude(clip, run.model.directory)
        for run in runs
        for clip in run.model.result.clips
    )
    for directory, benchmark in zip(configuration.benchmark_directories, benchmarks, strict=True):
        saved_manifest = TtsPilotManifest.model_validate_json(
            (directory / "cases.json").read_bytes()
        )
        validate_benchmark_manifest(
            saved_manifest, tuple(case.emotion.value for case in neu_manifest.cases)
        )
        amplitudes += verify_benchmark(benchmark, directory, neu_manifest)
    text = render_report(configuration, runs, benchmarks, amplitudes)
    configuration.output.mkdir(parents=True, exist_ok=True)
    path = configuration.output / "full_emotion_comparison.md"
    path.write_text(text, encoding="utf-8")
    (configuration.output / "report_config.json").write_text(
        configuration.model_dump_json(indent=2), encoding="utf-8"
    )
    (configuration.output / "amplitude_audit.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in amplitudes), encoding="utf-8"
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
