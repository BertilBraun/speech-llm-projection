"""Compare once-cached Whisper transcriptions to held-out dataset text."""

import re
from pathlib import Path

import jiwer

from speech_projector.cache import load_asr
from speech_projector.data import load_examples
from speech_projector.models import Record, Split


class AsrComparison(Record):
    example_id: str
    original_text: str
    asr_text: str
    word_error_rate: float


class AsrAudit(Record):
    examples: int
    normalized_word_error_rate: float
    normalized_character_error_rate: float
    normalization: str
    comparisons: tuple[AsrComparison, ...]


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def main() -> None:
    root = Path("/workspace/speech-projector/data")
    examples = load_examples(root / "examples.jsonl", Split.VALIDATION) + load_examples(
        root / "examples.jsonl", Split.TEST
    )
    transcriptions = {
        record.example_id: record.text for record in load_asr(root / "asr_transcripts.jsonl")
    }
    original = [normalize(example.user_text) for example in examples]
    recognized = [normalize(transcriptions[example.example_id]) for example in examples]
    comparisons = tuple(
        AsrComparison(
            example_id=example.example_id,
            original_text=example.user_text,
            asr_text=transcriptions[example.example_id],
            word_error_rate=jiwer.wer(
                normalize(example.user_text), normalize(transcriptions[example.example_id])
            ),
        )
        for example in examples[:24]
    )
    report = AsrAudit(
        examples=len(examples),
        normalized_word_error_rate=jiwer.wer(original, recognized),
        normalized_character_error_rate=jiwer.cer(original, recognized),
        normalization="Lowercase; non-word punctuation removed; whitespace collapsed; original dataset text retains lexical synthesis substitutions so WER is approximate",
        comparisons=comparisons,
    )
    (root / "asr_quality.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
