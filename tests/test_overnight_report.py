"""Reporting must preserve measured coverage and reject inconsistent selection records."""

from pathlib import Path

import pytest

from scripts.package_results import FileArtifact
from speech_projector.data import distribution
from speech_projector.models import EvaluationCondition, RunResult, SampleGeneration
from speech_projector.overnight_data import Cohort, NeuEmotionalExampleSource
from speech_projector.overnight_evaluation import SweepSelectionPolicy, select_sweep
from speech_projector.overnight_preparation import (
    CohortCoverage,
    CombinedPreparation,
    NeuSourceConfig,
    OvernightPreparationConfig,
    QwenSourceConfig,
)
from speech_projector.overnight_report import render_generations, render_overnight_report
from speech_projector.tts_pilot import PilotEmotion
from tests.test_overnight_evaluation import candidate


def preparation() -> CombinedPreparation:
    artifact = FileArtifact(
        path=Path("manifest"), source_path=Path("manifest"), bytes=1, sha256="a"
    )
    configuration = OvernightPreparationConfig(
        ordinary_manifest=Path("ordinary"),
        ordinary_asr_manifest=Path("ordinary_asr"),
        qwen=QwenSourceConfig(
            targets=Path("qwen_targets"),
            generation_configuration=Path("qwen_config"),
            audio_directory=Path("qwen_audio"),
        ),
        neu=NeuSourceConfig(
            targets=Path("neu_targets"),
            generation_configuration=Path("neu_config"),
            audio_directory=Path("neu_audio"),
            utterances=Path("neu_utterances"),
        ),
        output_directory=Path("combined"),
    )
    return CombinedPreparation(
        configuration=configuration,
        source_commit="source",
        source_artifacts=(artifact,),
        manifest=artifact,
        sidecar=artifact,
        audio_inventory=artifact,
        fixed_validation=artifact,
        fixed_test=artifact,
        coverage=(
            CohortCoverage(
                cohort=Cohort.ORDINARY,
                split="train",
                examples=20000,
                duration_seconds=distribution((2.0, 4.0)),
            ),
        ),
        exclusions=(),
        cached_features_reused=1,
        missing_features=2,
        ordinary_asr_reused=1,
        generation_example_ids=("a",),
        boundary_checks=("Passed source family split check",),
    )


def result() -> RunResult:
    selected = candidate("quality", 1, 1, 0.1, 10)
    return RunResult(
        config=selected.configuration,
        git_commit="run_source",
        train_examples=40000,
        validation_examples=128,
        test_examples=256,
        projector_parameters=12,
        pseudo_tokens_per_second=10,
        mean_pseudo_tokens=20,
        steps=2000,
        runtime_seconds=1000,
        peak_vram_gb=4,
        examples_per_second=8,
        target_tokens_per_second=80,
        initial_validation_loss=2,
        initial_training_loss=2,
        final_fixed_training_loss=1,
        final_training_loss=1,
        validation=selected.validation.old_ordinary,
        checkpoint_path=Path("projector.safetensors"),
    )


def test_report_rejects_inconsistent_results_and_preserves_unmeasured_test() -> None:
    selected = candidate("quality", 1, 1, 0.1, 10)
    decision = select_sweep((selected,), SweepSelectionPolicy())
    recorded = result()
    text = render_overnight_report(preparation(), decision, (recorded,), (recorded,), ())
    assert "Not evaluated" in text
    assert "1000.0 | 0.500 | 4.000" in text
    assert "PyTorch allocated decimal GB" in text
    assert "linearly resized" in text
    with pytest.raises(ValueError, match="coverage"):
        render_overnight_report(preparation(), decision, (), (), ())
    changed = recorded.model_copy(update={"config": recorded.config.model_copy(update={"seed": 7})})
    with pytest.raises(ValueError, match="configurations differ"):
        render_overnight_report(preparation(), decision, (changed,), (), ())


def test_qualitative_output_exposes_intended_tone_without_inventing_completion() -> None:
    source = NeuEmotionalExampleSource(
        example_id="a",
        source_manifest=Path("neu"),
        source_example_id="a",
        base_id="base",
        family_id="family",
        emotion=PilotEmotion.ANGRY,
    )
    sample = SampleGeneration(
        example_id="a",
        dialogue_id="family",
        condition=EvaluationCondition.SPEECH,
        history=(),
        user_transcript="The deadline moved again.",
        gold_response="That sounds frustrating.",
        generated_response="I hear that this is frustrating.",
        duration=2,
    )
    text = render_generations((sample,), (source,))
    assert "angry; not verified perception" in text
    assert "Generation termination: unknown" in text
    assert sample.generated_response in text
    assert sample.gold_response in text
