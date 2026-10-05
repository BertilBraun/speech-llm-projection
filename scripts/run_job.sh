#!/bin/bash
set -euo pipefail
export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=6
export PYTHONUNBUFFERED=1
cd /workspace/speech-projector
exec /venv/main/bin/python "$@"
