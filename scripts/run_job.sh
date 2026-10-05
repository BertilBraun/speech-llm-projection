#!/bin/bash
set -euo pipefail
export HF_HOME=/workspace/.hf_home
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=6
export PYTHONUNBUFFERED=1
cd /workspace/speech-projector
exec /venv/main/bin/python "$@"
