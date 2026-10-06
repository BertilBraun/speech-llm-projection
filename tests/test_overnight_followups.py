"""The shared neutral continuation recordings stay separate from the training corpus."""

from scripts.prepare_overnight_followups import followup_manifest
from speech_projector.tts_pilot import PilotEmotion, TtsPilotManifest


def test_followups_are_neutral_shared_audio_with_explicit_context_and_boundary_requests() -> None:
    manifest = followup_manifest()
    assert len(manifest.cases) == 4
    assert len({case.case_id for case in manifest.cases}) == 4
    assert all(case.emotion == PilotEmotion.NEUTRAL and case.seed == 42 for case in manifest.cases)
    assert "remember the details" in manifest.cases[0].text
    assert "without adding any new details" in manifest.cases[2].text
    assert "end this conversation" in manifest.cases[3].text
    assert "do not ask another question" in manifest.cases[3].text
    assert manifest == TtsPilotManifest.model_validate_json(manifest.model_dump_json())
