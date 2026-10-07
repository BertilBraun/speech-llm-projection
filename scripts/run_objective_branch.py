"""Train and validate one controlled lexical-alignment branch from its frozen parent."""

import argparse
from pathlib import Path

from speech_projector.objective_branch import ObjectiveBranchJob, run_objective_branch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    arguments = parser.parse_args()
    job = ObjectiveBranchJob.model_validate_json(arguments.config.read_bytes())
    outcome = run_objective_branch(job, smoke_only=arguments.smoke_only)
    print(outcome.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
