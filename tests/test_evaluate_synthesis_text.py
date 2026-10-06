from pathlib import Path

import pytest

from scripts.audit_synthesis_alignment import compare
from scripts.evaluate_synthesis_text import SynthesisTextInputs, verified_examples
from speech_projector.data import SourceSpeaker, SourceTurn
from speech_projector.inputs import TranscriptInput
from speech_projector.models import Example, Split


@pytest.fixture
def inputs() -> SynthesisTextInputs:
    records = []
    for index in range(2):
        example = Example(
            example_id=f"example{index}",
            dialogue_id=f"models/dialogue{index}",
            split=Split.VALIDATION,
            history=(),
            user_text="Original user words.",
            target_text="Original assistant target.",
            audio_path=Path(f"audio{index}.wav"),
            feature_path=Path(f"features{index}.pt"),
            duration=2.0,
            domain="test",
            emotion="neutral",
        )
        source = SourceTurn(
            conversation_id=f"dialogue{index}",
            model_dir="models",
            turn_index=0,
            speaker=SourceSpeaker.FIRST,
            text=example.user_text,
            domain="test",
            emotion="neutral",
            segment_audio_path=f"audio{index}.wav",
            audio_duration=2.0,
            audio_original_text="Synthesis original words.",
            audio_substituted_text="Synthesis substituted words.",
            audio_cleaned_text="Documented cleaned synthesis words.",
        )
        records.append(compare(example, source, index))
    return SynthesisTextInputs(tuple(records))


def test_factory_selects_documented_cleaned_text_and_preserves_example(
    inputs: SynthesisTextInputs,
) -> None:
    example = inputs.alignments[0].example
    original = example.model_dump_json()
    assert inputs(example) == TranscriptInput("Documented cleaned synthesis words.")
    assert example.model_dump_json() == original


@pytest.mark.parametrize("change", ["unknown_id", "different_target"])
def test_factory_rejects_missing_or_changed_audited_example(
    inputs: SynthesisTextInputs, change: str
) -> None:
    example = inputs.alignments[0].example
    match change:
        case "unknown_id":
            changed = example.model_copy(update={"example_id": "not audited"})
        case "different_target":
            changed = example.model_copy(update={"target_text": "A different response."})
    with pytest.raises(ValueError):
        inputs(changed)


def test_fixed_split_manifest_order_is_verified(
    inputs: SynthesisTextInputs, tmp_path: Path
) -> None:
    manifest = tmp_path / "examples.jsonl"
    examples = tuple(item.example for item in inputs.alignments)
    manifest.write_text(
        "".join(item.model_dump_json() + "\n" for item in examples), encoding="utf-8"
    )
    assert verified_examples(manifest, inputs, Split.VALIDATION, 2) == examples
    reordered = SynthesisTextInputs(tuple(reversed(inputs.alignments)))
    with pytest.raises(ValueError, match="ordered"):
        verified_examples(manifest, reordered, Split.VALIDATION, 2)
    changed = SynthesisTextInputs(
        (
            inputs.alignments[0].model_copy(
                update={"example": examples[0].model_copy(update={"duration": 3.0})}
            ),
            inputs.alignments[1],
        )
    )
    with pytest.raises(ValueError, match="unchanged"):
        verified_examples(manifest, changed, Split.VALIDATION, 2)
