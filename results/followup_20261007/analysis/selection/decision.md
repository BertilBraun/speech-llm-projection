# Frozen validation-only branch selection

Selected: followup_mean_10hz_ce_control_6775; CE control: followup_mean_10hz_ce_control_6775.

All 24 primary validation cases were explicitly reviewed for each auxiliary branch. The policy and complete reviews are bound to their exact input bytes. TEST results do not enter this decision. Thresholds are pragmatic safeguards, not calibrated significance or meaningful-effect bounds.

| Branch | Ordinary CE | Macro CE | Robust Neu margin | Eligible | In macro band |
|---|---:|---:|---:|---|---|
| followup_mean_10hz_ce_control_6775 | 1.588295 | 0.997702 | 0.104584 | True | True |
| followup_mean_10hz_transcript30_6775 | 1.593912 | 0.998907 | 0.104849 | True | True |
| followup_mean_10hz_ordinarykl_6775 | 1.650607 | 1.016662 | 0.105226 | False | False |

## Material corrections and regressions

- followup_mean_10hz_transcript30_6775 / qwen:utterance_01649_frustrated / corrected_material_failure: Removes almond milk→oat milk and chia seeds→cheese substitutions; generic acknowledgment avoids the material false ingredients, without establishing positive lexical recovery.
- followup_mean_10hz_transcript30_6775 / qwen:utterance_00286_happy / new_material_failure: Introduces an unprovided overlook/view destination while dropping the control's grounded acknowledgment of uphill difficulty.
- followup_mean_10hz_ordinarykl_6775 / ordinary:cf5f5a90ab8079a0bbc4 / new_material_failure: Adds the incorrect claim that Darcy lost his sister while still missing the adaptation request; control did not contain this fabrication.
- followup_mean_10hz_ordinarykl_6775 / qwen:utterance_00766_frustrated / new_material_failure: Replaces explicit user score90% with95%, a direct numeric contradiction absent from CE control.
