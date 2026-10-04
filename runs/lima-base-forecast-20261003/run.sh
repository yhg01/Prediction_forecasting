#!/bin/bash
set -euo pipefail
stage="$1"
model="$2"
seed="$3"
campaign_dir="/projects/u6oz/yuhe/lima-base-forecast-20261003"
training_python="/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python"
export HF_HOME=/projects/u6oz/yuhe/hf
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16
if [ "$stage" = stage ]; then
    export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
    exec "$training_python" "$campaign_dir/stage.py" --model "$model"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$training_python" "$campaign_dir/worker.py" --model "$model" --seed "$seed" --stage "$stage"
