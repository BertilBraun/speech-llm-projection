#!/bin/bash
set -euo pipefail
/bin/bash /workspace/speech-projector/scripts/run_job.sh -m speech_projector.cache --root /workspace/speech-projector/data --train-examples 30000 --no-asr
exec /bin/bash /workspace/speech-projector/scripts/run_job.sh -m speech_projector.launcher --manifest /workspace/speech-projector/data/examples.jsonl --output /workspace/speech-projector/results --config /workspace/speech-projector/configs/v1_30000_mlp_10hz.json
