"""Select final judge inputs from saved generations and canonical source identities."""

import argparse
from pathlib import Path

from pydantic import TypeAdapter

from speech_projector.journal import read_journal
from speech_projector.models import EvaluationCondition, SampleGeneration
from speech_projector.overnight_data import SourceSidecar
from speech_projector.overnight_judge import (
    FinalJudgingQuota,
    FinalJudgingSet,
    build_final_judging_set,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", type=EvaluationCondition, default=EvaluationCondition.SPEECH)
    arguments = parser.parse_args()
    for path in (arguments.generations, arguments.sources):
        contents = path.read_bytes()
        if contents and not contents.endswith(b"\n"):
            raise ValueError(
                f"Final-only judge preparation refuses incomplete input journal: {path}"
            )
    generations = read_journal(arguments.generations, SampleGeneration)
    sources = read_journal(arguments.sources, TypeAdapter(SourceSidecar))
    requests = build_final_judging_set(
        generations, sources, FinalJudgingQuota(), arguments.condition
    )
    if arguments.output.exists():
        if FinalJudgingSet.model_validate_json(arguments.output.read_bytes()) != requests:
            raise ValueError(
                "Judge request output contains different selection or candidate inputs"
            )
        return
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    pending_path = arguments.output.with_suffix(".json.part")
    pending_path.write_text(requests.model_dump_json(indent=2) + "\n", encoding="utf-8")
    pending_path.replace(arguments.output)
    print(requests.configuration.model_dump_json())


if __name__ == "__main__":
    main()
