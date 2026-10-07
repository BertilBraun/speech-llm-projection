# Locked model-assisted reviewer comparison

Both reviews were locked before mappings or each other's scores were opened. These are two Codex model-assisted reviewers with shared rubric and prior context, not independent human annotations or validated accuracy.

| Reviewer | Cases | Tone delta [95% familyCI] | W/T/L | Grounding delta [95%CI] | W/T/L |
|---|---:|---|---|---|---|
| Evaluator same24 | 24 | +0.0833 [-0.1364,+0.2727] | 4/18/2 | -0.2083 [-0.6667,+0.2083] | 3/15/6 |
| Root24 | 24 | +0.1667 [-0.0909,+0.4167] | 8/12/4 | -0.2917 [-0.6431,+0.0417] | 3/13/8 |

| Axis | Exact response-slot scores | Within1 | Same W/T/L case direction |
|---|---:|---:|---:|
| Tone | 40/48 | 48/48 | 18/24 |
| Grounding | 35/48 | 47/48 | 18/24 |
