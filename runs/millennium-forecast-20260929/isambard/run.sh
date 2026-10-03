#!/bin/bash
set -euo pipefail
CAMPAIGN=/projects/u6oz/yuhe/millennium-forecast-20260929
if [ -e "$CAMPAIGN/INFERENCE_SUPERSEDED.json" ]; then
  echo "Per-problem inference is superseded; use millennium-general-forecast-20260929" >&2
  exit 64
fi
MODEL_KEY="$1"
GPUS="$2"
EXTRA_ARGS=()
case "${3-full}" in full) ;; gate) EXTRA_ARGS=(--gate) ;; *) exit 64 ;; esac
case "$MODEL_KEY:$GPUS" in
  r1_distill_32b:1|qwen72b:2) ;;
  *) echo "Unreviewed model/resource combination" >&2; exit 2 ;;
esac
test -n "${SLURM_JOB_ID:-}"
export HF_HOME=/projects/u6oz/yuhe/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export HF_MODULES_CACHE="$CAMPAIGN/cache/hf-modules/$SLURM_JOB_ID"
export VLLM_CACHE_ROOT="$CAMPAIGN/cache/vllm/$SLURM_JOB_ID"
export TRITON_CACHE_DIR="$CAMPAIGN/cache/triton/$SLURM_JOB_ID"
export TORCHINDUCTOR_CACHE_DIR="$CAMPAIGN/cache/torchinductor/$SLURM_JOB_ID"
export OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p "$CAMPAIGN/output/diagnostics"
nvidia-smi --query-gpu=name,uuid,memory.total,memory.free --format=csv > "$CAMPAIGN/output/diagnostics/nvidia-smi-$SLURM_JOB_ID.csv"
exec srun --ntasks=1 --gpus="$GPUS" --cpu-bind=none \
  /projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python \
  "$CAMPAIGN/scripts/run_local_models.py" --campaign "$CAMPAIGN" \
  --model-key "$MODEL_KEY" --tensor-parallel-size "$GPUS" "${EXTRA_ARGS[@]}"
