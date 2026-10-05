"""Small temporal interfaces into the frozen language model embedding space."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from speech_projector.models import Architecture, ProjectorConfig


def mean_pool(features: Tensor, factor: int) -> Tensor:
    """Pool consecutive states, averaging the final incomplete block without padding bias."""
    if factor < 1:
        raise ValueError("Compression factor must be positive")
    length = features.shape[-2]
    if length == 0:
        raise ValueError("Speech features must have at least one state")
    padding = (-length) % factor
    padded = functional.pad(features, (0, 0, 0, padding))
    pooled = padded.reshape(*features.shape[:-2], -1, factor, features.shape[-1]).sum(-2)
    counts = torch.full((pooled.shape[-2],), factor, device=features.device, dtype=features.dtype)
    counts[-1] = factor - padding
    return pooled / counts.unsqueeze(-1)


class Projector(nn.Module):
    def __init__(self, config: ProjectorConfig) -> None:
        super().__init__()
        self.config = config
        self.normalization = nn.LayerNorm(config.encoder_dimension)
        match config.architecture:
            case Architecture.LINEAR:
                self.projection = nn.Linear(config.encoder_dimension, config.embedding_dimension)
                self.temporal = None
            case Architecture.MLP:
                self.projection = nn.Sequential(
                    nn.Linear(config.encoder_dimension, config.hidden_dimension),
                    nn.GELU(),
                    nn.Linear(config.hidden_dimension, config.embedding_dimension),
                )
                self.temporal = None
            case Architecture.CONV:
                self.temporal = nn.Conv1d(
                    config.encoder_dimension,
                    config.encoder_dimension,
                    kernel_size=config.compression_factor,
                    stride=config.compression_factor,
                )
                self.projection = nn.Sequential(
                    nn.Linear(config.encoder_dimension, config.hidden_dimension),
                    nn.GELU(),
                    nn.Linear(config.hidden_dimension, config.embedding_dimension),
                )
        # Match the pretrained embedding scale so the initial interface is numerically modest.
        final_layer = (
            self.projection if isinstance(self.projection, nn.Linear) else self.projection[-1]
        )
        assert isinstance(final_layer, nn.Linear)
        nn.init.normal_(final_layer.weight, std=0.02 / math.sqrt(final_layer.in_features))
        nn.init.zeros_(final_layer.bias)

    def forward(self, features: Tensor) -> Tensor:
        normalized = self.normalization(features.to(self.normalization.weight.dtype))
        if self.temporal is None:
            compressed = mean_pool(normalized, self.config.compression_factor)
        else:
            padding = (-normalized.shape[-2]) % self.config.compression_factor
            padded = functional.pad(normalized.transpose(-1, -2), (0, padding), mode="replicate")
            compressed = self.temporal(padded).transpose(-1, -2)
        return self.projection(compressed)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
