"""Generate the emotional audio dataset using the isolated Omni runtime."""

import argparse
from pathlib import Path

from speech_projector.emotional_audio import EmotionalAudioConfig
from speech_projector.emotional_audio_omni import OmniAudioConfig, generate_audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--deploy-config", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = OmniAudioConfig(
        audio=EmotionalAudioConfig.model_validate_json(arguments.config.read_bytes()),
        deploy_config=arguments.deploy_config,
    )
    summary = generate_audio(configuration)
    if summary.failed_case_ids:
        raise ValueError(f"{len(summary.failed_case_ids)} Omni audio cases require intervention")


if __name__ == "__main__":
    main()
