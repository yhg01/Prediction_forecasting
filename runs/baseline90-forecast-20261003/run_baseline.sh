#!/usr/bin/env bash
set -euo pipefail
CAMPAIGN='/projects/u6oz/yuhe/baseline90-forecast-20261003'
INFERENCE_PYTHON='/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python'
CONFIG_PATH="$1"
DRAW_BATCH="$2"
export HF_HOME='/projects/u6oz/yuhe/hf'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16
JOB_WORK="${SLURM_TMPDIR:-/tmp}/baseline90-${SLURM_JOB_ID}"
mkdir -p "$JOB_WORK"
export HF_MODULES_CACHE="$JOB_WORK/hf_modules"
export TORCHINDUCTOR_CACHE_DIR="$JOB_WORK/inductor"
export TRITON_CACHE_DIR="$JOB_WORK/triton"
PRIVATE_RUNTIME="$("$INFERENCE_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["private_runtime"] or "")' "$CONFIG_PATH")"
if [[ -n "$PRIVATE_RUNTIME" ]]; then export PYTHONPATH="$PRIVATE_RUNTIME"; else unset PYTHONPATH; fi
ENTRYPOINT="$("$INFERENCE_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["evaluation"].get("entrypoint", "evaluate_code_forecasts.py"))' "$CONFIG_PATH")"
case "$ENTRYPOINT" in
  evaluate_code_forecasts.py|evaluate_code_forecasts_qwen_eager_v4.py|evaluate_code_forecasts_qwen2026_v2.py) ;;
  *) echo 'Unreviewed worker' >&2; exit 64 ;;
esac
cd "$JOB_WORK"
exec "$INFERENCE_PYTHON" "$CAMPAIGN/scripts/$ENTRYPOINT" --config "$CONFIG_PATH" --arm base --seed 0 --batch "$DRAW_BATCH"
