#!/bin/bash
set -euo pipefail
export HF_HOME=/projects/u6oz/yuhe/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16
exec /projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python /projects/u6oz/yuhe/lima-base-forecast-20261003/rerun_qwen3_worker.py
