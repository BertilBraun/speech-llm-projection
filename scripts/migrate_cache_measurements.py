"""Mark the unmeasured initial cache wall throughput explicitly as unavailable."""

from pathlib import Path

from speech_projector.cache import CacheStatistics


def mark_unmeasured_wall_time(serialized: str) -> CacheStatistics:
    corrected = serialized.replace(
        '"extraction_seconds":',
        '"cache_wall_seconds": null, "cache_examples_per_second": null, "extraction_seconds":',
        1,
    )
    return CacheStatistics.model_validate_json(corrected)


def main() -> None:
    root = Path("/workspace/speech-projector/data")
    original = root / "cache_stats_initial_1000.json"
    initial = mark_unmeasured_wall_time(original.read_text(encoding="utf-8"))
    original.write_text(initial.model_dump_json(indent=2), encoding="utf-8")
    history = root / "cache_history.jsonl"
    lines = history.read_text(encoding="utf-8").splitlines()
    lines[0] = initial.model_dump_json()
    history.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path in root.glob("cache_stats_*.json"):
        CacheStatistics.model_validate_json(path.read_text(encoding="utf-8"))
    for line in lines:
        CacheStatistics.model_validate_json(line)
    print("Initial archive and all cache history records validated without recomputing features")


if __name__ == "__main__":
    main()
