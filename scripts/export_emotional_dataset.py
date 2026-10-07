"""Build a local dataset release; never upload or regenerate source audio."""

import argparse
from pathlib import Path

from speech_projector.dataset_release import (
    DatasetExportConfig,
    export_dataset,
    verify_dataset_export,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    config = DatasetExportConfig.model_validate_json(arguments.config.read_bytes())
    if arguments.verify:
        verification = verify_dataset_export(config)
        print(
            f"Verified {verification.examples:,} embedded original WAVs / "
            f"{verification.shards} shards; {verification.audio_bytes / 1e9:.3g} GB."
        )
        return
    receipt = export_dataset(config)
    print(
        f"Exported {receipt.examples:,} examples / {receipt.utterances:,} pairs; "
        f"WAV payload {receipt.source_audio_bytes / 1e9:.3g} GB, "
        f"{len(receipt.shards)} Parquet shards."
    )


if __name__ == "__main__":
    main()
