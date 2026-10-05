"""Download the fixed checkpoints while CPU implementation proceeds."""

from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import snapshot_download


def download(model_name: str) -> str:
    return snapshot_download(
        model_name,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.tiktoken", "*.model", "*.jinja"],
    )


def main() -> None:
    models = (
        "openai/whisper-small",
        "Qwen/Qwen3.5-2B",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    with ThreadPoolExecutor(max_workers=3) as executor:
        for path in executor.map(download, models):
            print(path, flush=True)


if __name__ == "__main__":
    main()
