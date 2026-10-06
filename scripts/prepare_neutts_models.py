"""Pin and download official NeuTTS-2E/NeuCodec into their isolated offline cache."""

from pathlib import Path
from time import perf_counter

from huggingface_hub import HfApi, ModelInfo, snapshot_download
from pydantic import BaseModel, ConfigDict

SDK_SPEAKERS = ("emily", "paul", "sophie", "steven")
SDK_EMOTIONS = ("angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised")


class PinnedRepository(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    repository: str
    revision: str
    snapshot: Path
    download_seconds: float | None


class NeuTtsPreparation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    source_commit: str
    repositories: tuple[PinnedRepository, PinnedRepository]
    auxiliary_repositories: tuple[PinnedRepository, ...]
    sdk_speakers: tuple[str, ...]
    sdk_emotions: tuple[str, ...]


def pin(repository: str, cache: Path, allow_patterns: tuple[str, ...] | None) -> PinnedRepository:
    information: ModelInfo = HfApi().model_info(repository)
    if information.sha is None:
        raise ValueError(f"HF metadata lacks an immutable revision: {repository}")
    started = perf_counter()
    snapshot = Path(
        snapshot_download(
            repository,
            revision=information.sha,
            cache_dir=cache,
            allow_patterns=allow_patterns,
        )
    )
    references = snapshot.parent.parent / "refs"
    references.mkdir(parents=True, exist_ok=True)
    (references / "main").write_text(information.sha, encoding="utf-8")
    result = PinnedRepository(
        repository=repository,
        revision=information.sha,
        snapshot=snapshot,
        download_seconds=perf_counter() - started,
    )
    print(result.model_dump_json(), flush=True)
    return result


def main() -> None:
    root = Path("/workspace/tts-comparison-20261006/neutts")
    cache = root / "hf_home/hub"
    preparation = NeuTtsPreparation(
        source_commit="ac69851f28fc63a487917e7c2e27f0d75c759cba",
        repositories=(
            pin("neuphonic/neutts-2e", cache, None),
            pin("neuphonic/neucodec", cache, None),
        ),
        auxiliary_repositories=(
            pin(
                "facebook/w2v-bert-2.0",
                cache,
                ("config.json", "model.safetensors", "preprocessor_config.json"),
            ),
        ),
        sdk_speakers=SDK_SPEAKERS,
        sdk_emotions=SDK_EMOTIONS,
    )
    (root / "preparation.json").write_text(
        preparation.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
