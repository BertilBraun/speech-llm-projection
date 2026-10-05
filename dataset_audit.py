"""One-time metadata duplication and downloaded waveform audit."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile

from speech_projector.data import dialogue_split, load_examples, load_source
from speech_projector.models import Record, Split


class WaveformAudit(Record):
    example_id: str
    sample_rate: int
    channels: int
    actual_duration: float
    metadata_duration: float
    rms: float
    peak: float
    clipped_fraction: float
    transcript: str


class DatasetQuality(Record):
    duplicate_dialogue_groups: int
    cross_split_duplicate_dialogue_groups: int
    duplicate_check: str
    repeated_conversation_ids_across_model_pairs: int
    audio_substitution_changed: int
    audio_cleaning_changed: int
    random_waveform_samples: tuple[WaveformAudit, ...]
    quality_notes: tuple[str, ...]


def main() -> None:
    root = Path("/workspace/speech-projector/data")
    rows, _ = load_source(root / "metadata.parquet")
    conversations: defaultdict[str, list[tuple[int, str, str]]] = defaultdict(list)
    identities: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        conversations[row.dialogue_id].append((row.turn_index, row.speaker, row.text))
        identities[row.conversation_id].add(row.model_dir)
    repeated: defaultdict[tuple[tuple[int, str, str], ...], list[str]] = defaultdict(list)
    for identity, turns in conversations.items():
        repeated[tuple(sorted(turns))].append(identity)
    duplicates = [group for group in repeated.values() if len(group) > 1]
    crossing = [
        group
        for group in duplicates
        if len({dialogue_split(identity, 42) for identity in group}) > 1
    ]
    samples = load_examples(root / "examples.jsonl", Split.VALIDATION, 12)
    waveforms: list[WaveformAudit] = []
    for sample in samples:
        waveform, rate = soundfile.read(sample.audio_path, dtype="float32", always_2d=True)
        waveforms.append(
            WaveformAudit(
                example_id=sample.example_id,
                sample_rate=rate,
                channels=waveform.shape[1],
                actual_duration=len(waveform) / rate,
                metadata_duration=sample.duration,
                rms=float(np.sqrt(np.mean(waveform**2))),
                peak=float(np.max(np.abs(waveform))),
                clipped_fraction=float(np.mean(np.abs(waveform) >= 0.999)),
                transcript=sample.user_text,
            )
        )
    report = DatasetQuality(
        duplicate_dialogue_groups=len(duplicates),
        cross_split_duplicate_dialogue_groups=len(crossing),
        duplicate_check=(
            "Exact complete ordered (turn_index,speaker,text) sequences over all source rows; "
            "split seed42"
        ),
        repeated_conversation_ids_across_model_pairs=sum(
            len(pairs) > 1 for pairs in identities.values()
        ),
        audio_substitution_changed=sum(
            row.audio_substituted_text is not None
            and row.audio_original_text != row.audio_substituted_text
            for row in rows
        ),
        audio_cleaning_changed=sum(
            row.audio_cleaned_text is not None and row.audio_original_text != row.audio_cleaned_text
            for row in rows
        ),
        random_waveform_samples=tuple(waveforms),
        quality_notes=(
            "Synthetic LLM-to-LLM dialogue; turns often generic or emotionally repetitive.",
            "All 12 checked clips valid mono 24 kHz; duration matches metadata, zero clipping.",
            "Emojis and repeated phrases occur. Text baseline uses original text; synthesis "
            "cleaning can remove emoji/punctuation and 645 utterances had substitution changes.",
            "Four exact duplicate whole-dialogue groups do not cross the chosen seed42 split.",
            "No listening playback available; waveform and ASR inspect measurable quality only.",
        ),
    )
    (root / "dataset_quality.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
