#!/bin/bash
set -euo pipefail
CAMPAIGN=/projects/u6oz/yuhe/millennium-forecast-20260929
PYTHON=/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python
: "${SLURM_JOB_ID:?A guarded Slurm allocation is required}"
case "${1-}:${2-}" in r1_distill_32b:0|qwen72b:0) ;; *) exit 64 ;; esac
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/projects/u6oz/yuhe/hf
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_TOKEN HF_TOKEN_PATH
export STAGING_LOG_PATH="$CAMPAIGN/slurm/$SLURM_JOB_ID.out"
cd "$CAMPAIGN"
# The campaign ledger prevents duplicate jobs; this persistent model lock also
# prevents two manually resumed staging launches from writing the same files.
exec 9>"$CAMPAIGN/staging-${1}.lock"
flock -n 9
srun --ntasks=1 --cpus-per-task=8 --cpu-bind=none "$PYTHON" "$CAMPAIGN/stage_models.py" --only "$1" --workers 8
srun --ntasks=1 --cpus-per-task=8 --cpu-bind=none "$PYTHON" "$CAMPAIGN/verify_staged.py" --model-key "$1"
