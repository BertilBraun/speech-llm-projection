"""Validate and accept or reject the optional second full pass without using TEST outcomes."""

import argparse
from pathlib import Path

from speech_projector.followup_extension_selection import (
    FullPassExtensionConfig,
    select_full_pass_extension,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = FullPassExtensionConfig.model_validate_json(arguments.config.read_bytes())
    decision = select_full_pass_extension(configuration)
    print(f"Optional second pass: {decision.kind}")


if __name__ == "__main__":
    main()
