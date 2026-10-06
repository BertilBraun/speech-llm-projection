from pathlib import Path

import pytest

from scripts.validate_typed_api import (
    GenerationParity,
    LossParity,
    ParityConfiguration,
    ParityReport,
    assert_parity,
)
from speech_projector.models import EvaluationCondition, GradientCheck


def parity_report(candidate_loss: float, candidate_tokens: int, response: str) -> ParityReport:
    return ParityReport(
        config=ParityConfiguration(
            run_directory=Path("run"),
            manifest=Path("examples.jsonl"),
            baseline_directory=Path("baseline"),
            output_path=Path("parity.json"),
            candidate_commit="candidate",
            expected_reference_commit="reference",
        ),
        run_name="test",
        reference_commit="reference",
        checkpoint_path=Path("projector.safetensors"),
        checkpoint_sha256="checkpoint",
        losses=(
            LossParity(
                example_id="example",
                condition=EvaluationCondition.SPEECH,
                reference_cross_entropy=1.0,
                candidate_cross_entropy=candidate_loss,
                absolute_error=abs(candidate_loss - 1.0),
                reference_target_tokens=10,
                candidate_target_tokens=candidate_tokens,
            ),
        ),
        generations=(
            GenerationParity(
                example_id="example",
                condition=EvaluationCondition.SPEECH,
                reference_response="Hello.",
                candidate_response=response,
                exact_match=response == "Hello.",
            ),
        ),
        gradient=GradientCheck(
            loss=1.0,
            projector_gradient_norm=0.1,
            projector_changed=True,
            frozen_parameters=100,
            llm_has_gradients=False,
            llm_weights_unchanged=True,
            peak_vram_gb=0.0,
            step_seconds=0.1,
        ),
        elapsed_seconds=0.1,
    )


@pytest.mark.parametrize(
    ("candidate_loss", "candidate_tokens", "response"),
    (
        (1.01, 10, "Hello."),
        (float("nan"), 10, "Hello."),
        (1.0, 9, "Hello."),
        (1.0, 10, "Hello. user\nFabricated turn"),
    ),
)
def test_parity_rejects_numeric_masking_or_generation_regression(
    candidate_loss: float, candidate_tokens: int, response: str
) -> None:
    with pytest.raises(ValueError):
        assert_parity(parity_report(candidate_loss, candidate_tokens, response))


def test_parity_accepts_equal_observations() -> None:
    assert_parity(parity_report(1.0, 10, "Hello."))
