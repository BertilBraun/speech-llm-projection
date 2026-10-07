# Locked model-assisted reviewer comparison

Both reviews were locked before mappings or each other's scores were opened. These are two Codex model-assisted reviewers with shared rubric and prior context, not independent human annotations or validated accuracy.

| Reviewer | Cases | Tone delta [95% familyCI] | W/T/L | Grounding delta [95%CI] | W/T/L |
|---|---:|---|---|---|---|
| Evaluator same24 | 24 | -0.0833 [-0.3333,+0.1250] | 3/17/4 | -0.4167 [-0.8333,+0.0000] | 2/13/9 |
| Root24 | 24 | +0.1250 [-0.1250,+0.3750] | 8/12/4 | -0.1667 [-0.4167,+0.1250] | 2/16/6 |

| Axis | Exact response-slot scores | Within1 | Same W/T/L case direction |
|---|---:|---:|---:|
| Tone | 33/48 | 48/48 | 17/24 |
| Grounding | 39/48 | 47/48 | 21/24 |
