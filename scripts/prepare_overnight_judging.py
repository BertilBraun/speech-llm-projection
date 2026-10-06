"""Select final judge inputs from saved generations and canonical source identities."""

import argparse
from pathlib import Path

from pydantic import TypeAdapter

from scripts.inventory_results import stable_digest
from speech_projector.journal import read_journal
from speech_projector.models import EvaluationCondition, FileArtifact, SampleGeneration
from speech_projector.overnight_data import SourceSidecar
from speech_projector.overnight_judge import (
    FinalJudgingProgram,
    FinalJudgingQuota,
    build_final_judging_set,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True, action="append")
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    paths = tuple(arguments.generations) + (arguments.sources,)
    artifacts: list[FileArtifact] = []
    for path in paths:
        contents = path.read_bytes()
        if contents and not contents.endswith(b"\n"):
            raise ValueError(
                f"Final-only judge preparation refuses incomplete input journal: {path}"
            )
        size, digest = stable_digest(path)
        artifacts.append(
            FileArtifact(path=path, source_path=path.resolve(), bytes=size, sha256=digest)
        )
    generations = tuple(
        item for path in arguments.generations for item in read_journal(path, SampleGeneration)
    )
    sources = read_journal(arguments.sources, TypeAdapter(SourceSidecar))
    requests = FinalJudgingProgram(
        sets=tuple(
            build_final_judging_set(generations, sources, FinalJudgingQuota(), condition)
            for condition in (
                EvaluationCondition.SPEECH,
                EvaluationCondition.TEXT,
                EvaluationCondition.ASR,
            )
            if any(item.condition == condition for item in generations)
        ),
        source_artifacts=tuple(artifacts),
    )
    if arguments.output.exists():
        if FinalJudgingProgram.model_validate_json(arguments.output.read_bytes()) != requests:
            raise ValueError(
                "Judge request output contains different selection or candidate inputs"
            )
        return
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    pending_path = arguments.output.with_suffix(".json.part")
    pending_path.write_text(requests.model_dump_json(indent=2) + "\n", encoding="utf-8")
    pending_path.replace(arguments.output)
    print("Prepared conditions: " + ", ".join(item.condition.value for item in requests.sets))


if __name__ == "__main__":
    main()
