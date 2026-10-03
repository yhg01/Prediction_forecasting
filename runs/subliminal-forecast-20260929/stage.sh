#!/usr/bin/env bash
set -euo pipefail
CAMPAIGN='/projects/u6oz/yuhe/insecure-code-forecast-20260929'
TRAIN_PYTHON='/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python'
: "${SLURM_JOB_ID:?A guarded Slurm allocation is required}"
case "${1-}" in qwen25_72b|qwen3_32b|qwen35_27b|qwen38_27b) ;; *) exit 64 ;; esac
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME='/projects/u6oz/yuhe/hf'
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_TOKEN HF_TOKEN_PATH
cd "$CAMPAIGN"
exec 9>"$CAMPAIGN/staging-${1}.lock"
flock -n 9
exec srun --ntasks=1 --cpus-per-task=8 --cpu-bind=none "$TRAIN_PYTHON" \
  "$CAMPAIGN/stage_models.py" --only "$1" --workers 8
