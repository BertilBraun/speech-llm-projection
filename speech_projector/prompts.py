"""Qwen non-thinking chat delimiters shared by transcript and speech inputs."""

from speech_projector.models import ChatPromptConfig, PromptConfig, SystemPromptConfig

USER_PREFIX = "<|im_start|>user\n"
ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def system_prefix(config: PromptConfig) -> str:
    match config:
        case SystemPromptConfig(system_text=text):
            return f"<|im_start|>system\n{text}<|im_end|>\n"
        case ChatPromptConfig():
            return ""
