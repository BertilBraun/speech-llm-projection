import pytest
import torch

from speech_projector.models import (
    ConvProjectorConfig,
    LinearProjectorConfig,
    MlpProjectorConfig,
    ProjectorConfig,
)
from speech_projector.projectors import Projector, mean_pool


def test_pool_partial_block() -> None:
    features = torch.arange(5, dtype=torch.float32).unsqueeze(-1)
    assert torch.equal(mean_pool(features, 2), torch.tensor([[0.5], [2.5], [4.0]]))


@pytest.mark.parametrize(
    "config",
    (
        LinearProjectorConfig(compression_factor=5, encoder_dimension=8, embedding_dimension=16),
        MlpProjectorConfig(
            compression_factor=5, encoder_dimension=8, embedding_dimension=16, hidden_dimension=12
        ),
        ConvProjectorConfig(
            compression_factor=5, encoder_dimension=8, embedding_dimension=16, hidden_dimension=12
        ),
    ),
)
@pytest.mark.parametrize("length", [1, 5, 50])
def test_shapes_and_backward(config: ProjectorConfig, length: int) -> None:
    projector = Projector(config)
    features = torch.randn(length, 8)
    projected = projector(features)
    assert projected.shape == ((length + 4) // 5, 16)
    projected.square().mean().backward()
    assert all(parameter.grad is not None for parameter in projector.parameters())
    assert sum(parameter.grad.abs().sum() for parameter in projector.parameters()) > 0
