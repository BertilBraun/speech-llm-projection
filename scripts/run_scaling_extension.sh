#!/bin/bash
set -euo pipefail
configuration_path="${1:?A configuration path is required}"
training_examples=$(/venv/main/bin/python -c 'import sys; from pathlib import Path; from speech_projector.models import RunConfig; print(RunConfig.model_validate_json(Path(sys.argv[1]).read_text()).train_examples)' "$configuration_path")
cache_statistics="/workspace/speech-projector/data/cache_stats_${training_examples}.json"
# Preserve the original cache timing when a training retry resumes.
if [[ ! -f "$cache_statistics" ]]; then
    /bin/bash /workspace/speech-projector/scripts/run_job.sh -m speech_projector.cache --root /workspace/speech-projector/data --train-examples "$training_examples" --no-asr
fi
exec /bin/bash /workspace/speech-projector/scripts/run_job.sh -m speech_projector.launcher --manifest /workspace/speech-projector/data/examples.jsonl --output /workspace/speech-projector/results --config "$configuration_path"
