#!/usr/bin/env bash
set -euo pipefail
CAMPAIGN='/projects/u6oz/yuhe/today-date-probe-20261003-v2'
INFERENCE_PYTHON='/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python'
CONFIG_PATH="$1"
BUNDLE_SHA256="$2"
export HF_HOME='/projects/u6oz/yuhe/hf'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16
JOB_WORK="${SLURM_TMPDIR:-/tmp}/today-date-${SLURM_JOB_ID}"
mkdir -p "$JOB_WORK"
export HF_MODULES_CACHE="$JOB_WORK/hf_modules"
export TORCHINDUCTOR_CACHE_DIR="$JOB_WORK/inductor"
export TRITON_CACHE_DIR="$JOB_WORK/triton"
PRIVATE_RUNTIME="$("$INFERENCE_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["private_runtime"] or "")' "$CONFIG_PATH")"
if [[ -n "$PRIVATE_RUNTIME" ]]; then export PYTHONPATH="$PRIVATE_RUNTIME"; else unset PYTHONPATH; fi
cd "$JOB_WORK"
exec "$INFERENCE_PYTHON" "$CAMPAIGN/worker.py" --config "$CONFIG_PATH" --bundle-sha256 "$BUNDLE_SHA256"
