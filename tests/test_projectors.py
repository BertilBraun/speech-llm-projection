import pytest
import torch

from speech_projector.models import (
    ConvProjectorConfig,
    LinearProjectorConfig,
    MlpProjectorConfig,
    ProjectorConfig,
    StackedMlpProjectorConfig,
)
from speech_projector.projectors import Projector, mean_pool, stack_frames


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
        StackedMlpProjectorConfig(
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


def test_stacking_preserves_order_and_zero_fills_only_partial_group() -> None:
    features = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    expected = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 0, 0]], dtype=torch.float32)
    assert torch.equal(stack_frames(features, 3), expected)
    assert torch.equal(stack_frames(features.unsqueeze(0), 3), expected.unsqueeze(0))


def test_stacked_projector_normalizes_each_frame_before_zero_padding() -> None:
    config = StackedMlpProjectorConfig(
        compression_factor=2, encoder_dimension=4, hidden_dimension=6, embedding_dimension=8
    )
    projector = Projector(config)
    features = torch.randn(3, 4)
    expected = projector.projection(stack_frames(projector.normalization(features), 2))
    assert torch.equal(projector(features), expected)
    assert projector.projection[0].in_features == 8
