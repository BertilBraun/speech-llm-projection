"""Apply the frozen VAL-only rule after complete hash-bound 24-case content reviews."""

import argparse
from pathlib import Path

from speech_projector.followup_validation_review import (
    ValidationSelectionConfig,
    select_reviewed_branches,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = ValidationSelectionConfig.model_validate_json(arguments.config.read_bytes())
    decision = select_reviewed_branches(configuration)
    print(f"Validation-selected branch: {decision.selected_run}")


if __name__ == "__main__":
    main()
