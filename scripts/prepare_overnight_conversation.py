"""Prepare seeded held-out conversation fixtures from real neutral evaluation audio."""

import argparse
from pathlib import Path

import soundfile
from pydantic import TypeAdapter

from scripts.inventory_results import stable_digest
from scripts.prepare_overnight_followups import followup_manifest
from speech_projector.journal import read_journal
from speech_projector.models import Example
from speech_projector.overnight_conversation import (
    DelayedCueConfiguration,
    DelayedCueFixtures,
    FollowupAudio,
    build_delayed_cue_fixtures,
)
from speech_projector.overnight_data import SourceSidecar
from speech_projector.tts_pilot import TtsPilotResult


def prepare_followups(
    result: TtsPilotResult, audio_root: Path, feature_root: Path
) -> tuple[FollowupAudio, FollowupAudio, FollowupAudio]:
    if result.failures:
        raise ValueError("Neutral evaluation audio contains unresolved synthesis failures")
    by_id = {item.case.case_id: item for item in result.clips}
    plan = {item.case_id: item for item in followup_manifest().cases}
    selected: list[FollowupAudio] = []
    for identifier in ("followup_short", "followup_next_step", "followup_stop"):
        if identifier not in by_id or by_id[identifier].case != plan[identifier]:
            raise ValueError(f"Missing or changed shared neutral follow-up: {identifier}")
        clip = by_id[identifier]
        path = Path(clip.audio_path)
        if not path.is_absolute():
            path = audio_root / path
        _, digest = stable_digest(path)
        if digest != clip.sha256:
            raise ValueError(f"Shared follow-up WAV differs from its recorded hash: {path}")
        metadata = soundfile.info(path)
        if (
            metadata.channels != 1
            or metadata.samplerate != clip.sample_rate
            or abs(metadata.duration - clip.audio_seconds) > 1 / metadata.samplerate
        ):
            raise ValueError(f"Shared follow-up waveform metadata differs from its record: {path}")
        selected.append(FollowupAudio(clip=clip, feature_path=feature_root / f"{clip.sha256}.pt"))
    return selected[0], selected[1], selected[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--followup-result", type=Path, required=True)
    parser.add_argument("--followup-audio-root", type=Path, required=True)
    parser.add_argument("--followup-feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    for path in (arguments.manifest, arguments.sources):
        contents = path.read_bytes()
        if contents and not contents.endswith(b"\n"):
            raise ValueError("Conversation fixture preparation requires complete source journals")
    examples = read_journal(arguments.manifest, Example)
    sources = read_journal(arguments.sources, TypeAdapter(SourceSidecar))
    followups = prepare_followups(
        TtsPilotResult.model_validate_json(arguments.followup_result.read_bytes()),
        arguments.followup_audio_root,
        arguments.followup_feature_root,
    )
    fixtures = build_delayed_cue_fixtures(examples, sources, followups, DelayedCueConfiguration())
    if arguments.output.exists():
        if DelayedCueFixtures.model_validate_json(arguments.output.read_bytes()) != fixtures:
            raise ValueError(
                "Refusing to overwrite different outcome-independent conversation fixtures"
            )
        return
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    pending_path = arguments.output.with_suffix(".json.part")
    pending_path.write_text(fixtures.model_dump_json(indent=2) + "\n", encoding="utf-8")
    pending_path.replace(arguments.output)
    print("Selected held-out bases: " + ", ".join(item.base_id for item in fixtures.scenarios))


if __name__ == "__main__":
    main()
