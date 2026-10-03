#!/usr/bin/env bash
set -euo pipefail
CAMPAIGN='/projects/u6oz/yuhe/bad-advice-forecast-20260930'
TRAIN_PYTHON='/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python'
CONFIG_PATH="$1"
export HF_HOME='/projects/u6oz/yuhe/hf'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16
JOB_WORK="${SLURM_TMPDIR:-/tmp}/medical-alignment-${SLURM_JOB_ID}"
mkdir -p "$JOB_WORK"
export HF_MODULES_CACHE="$JOB_WORK/hf_modules" TORCHINDUCTOR_CACHE_DIR="$JOB_WORK/inductor" TRITON_CACHE_DIR="$JOB_WORK/triton"
PRIVATE_RUNTIME="$("$TRAIN_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["private_runtime"] or "")' "$CONFIG_PATH")"
if [[ -n "$PRIVATE_RUNTIME" ]]; then export PYTHONPATH="$PRIVATE_RUNTIME"; fi
cd "$JOB_WORK"
shift
exec "$TRAIN_PYTHON" "$CAMPAIGN/alignment/worker.py" --config "$CONFIG_PATH" "$@"
