"""Qwen non-thinking chat delimiters shared by transcript and speech inputs."""

from speech_projector.models import PromptConfig

USER_PREFIX = "<|im_start|>user\n"
ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def system_prefix(config: PromptConfig) -> str:
    return f"<|im_start|>system\n{config.system_text}<|im_end|>\n"
