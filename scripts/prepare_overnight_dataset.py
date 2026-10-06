"""Assemble new training inputs without rewriting any original corpus."""

import argparse
import subprocess
from pathlib import Path

from speech_projector.overnight_preparation import OvernightPreparationConfig, prepare_combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = OvernightPreparationConfig.model_validate_json(arguments.config.read_bytes())
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result = prepare_combined(configuration, revision)
    print(result.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
