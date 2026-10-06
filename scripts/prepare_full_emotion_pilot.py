"""Build full official emotion manifests without modifying the completed pilot."""

import argparse
from pathlib import Path

from speech_projector.index_tts_pilot import IndexPilotConfig, write_record
from speech_projector.tts_pilot import PilotEmotion, TtsPilotCase, TtsPilotManifest

TEXT = "I finally got a reply about the appointment, and they want me to call tomorrow."
INDEX_EMOTIONS = (
    PilotEmotion.NEUTRAL,
    PilotEmotion.HAPPY,
    PilotEmotion.ANGRY,
    PilotEmotion.SAD,
    PilotEmotion.AFRAID,
    PilotEmotion.DISGUSTED,
    PilotEmotion.MELANCHOLIC,
    PilotEmotion.SURPRISED,
)
NEU_EMOTIONS = (
    PilotEmotion.NEUTRAL,
    PilotEmotion.HAPPY,
    PilotEmotion.ANGRY,
    PilotEmotion.SAD,
    PilotEmotion.FEARFUL,
    PilotEmotion.DISGUSTED,
    PilotEmotion.SURPRISED,
)


def manifest(emotions: tuple[PilotEmotion, ...]) -> TtsPilotManifest:
    return TtsPilotManifest(
        cases=tuple(
            TtsPilotCase(
                case_id=f"appointment_{emotion.value}",
                utterance_id="appointment",
                text=TEXT,
                emotion=emotion,
                seed=42,
            )
            for emotion in emotions
        ),
        warmup_text=TEXT,
        warmup_seed=41,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--previous-index-config", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    write_record(arguments.output / "index_cases.json", manifest(INDEX_EMOTIONS))
    write_record(arguments.output / "neu_cases.json", manifest(NEU_EMOTIONS))
    previous = IndexPilotConfig.model_validate_json(arguments.previous_index_config.read_bytes())
    for beams in (3, 1):
        configuration = IndexPilotConfig(
            manifest_path=arguments.remote_root / "index_cases.json",
            output_directory=arguments.remote_root / f"index_beams{beams}",
            checkpoint_directory=previous.checkpoint_directory,
            reference_audio=previous.reference_audio,
            reference_sha256=previous.reference_sha256,
            model_revision=previous.model_revision,
            vendor_source_commit=previous.vendor_source_commit,
            source_commit=arguments.source_commit,
            emotion_intensity=1.0,
            max_mel_tokens=previous.max_mel_tokens,
            top_p=previous.top_p,
            top_k=previous.top_k,
            temperature=previous.temperature,
            num_beams=beams,
            repetition_penalty=previous.repetition_penalty,
        )
        write_record(arguments.output / f"index_beams{beams}_config.json", configuration)


if __name__ == "__main__":
    main()
