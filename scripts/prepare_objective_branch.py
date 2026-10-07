"""Prepare an immutable optimizer continuation with explicitly changed supervision."""

import argparse
from pathlib import Path

from speech_projector.models import RunConfig
from speech_projector.overnight_continuation import prepare_objective_branch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = RunConfig.model_validate_json(arguments.config.read_bytes())
    provenance = prepare_objective_branch(arguments.source, arguments.output, configuration)
    print(provenance.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
