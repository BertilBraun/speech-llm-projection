"""Download the independent compact judge without allocating GPU memory."""

from scripts.download_models import download


def main() -> None:
    print(download("Qwen/Qwen3-1.7B"), flush=True)


if __name__ == "__main__":
    main()
