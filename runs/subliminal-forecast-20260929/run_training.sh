#!/usr/bin/env bash
set -euo pipefail
CAMPAIGN='/projects/u6oz/yuhe/insecure-code-forecast-20260929'
TRAIN_PYTHON='/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python'
CONFIG_PATH="$1"
ARM_NAME="$2"
TRAIN_SEED="$3"
STAGE_NAME="$4"
export HF_HOME='/projects/u6oz/yuhe/hf'
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16
JOB_WORK="${SLURM_TMPDIR:-/tmp}/insecure-code-${SLURM_JOB_ID}"
mkdir -p "$JOB_WORK"
export HF_MODULES_CACHE="$JOB_WORK/hf_modules"
export TORCHINDUCTOR_CACHE_DIR="$JOB_WORK/inductor"
export TRITON_CACHE_DIR="$JOB_WORK/triton"
PRIVATE_RUNTIME="$("$TRAIN_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["private_runtime"] or "")' "$CONFIG_PATH")"
if [[ -n "$PRIVATE_RUNTIME" ]]; then
  export PYTHONPATH="$PRIVATE_RUNTIME"
fi
cd "$JOB_WORK"
ENTRYPOINT="$("$TRAIN_PYTHON" -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c["evaluation"].get("entrypoint", "evaluate_code_forecasts.py") if sys.argv[2] in {"evaluate", "cache-gate"} else c.get("training_entrypoint", "train_insecure_code.py"))' "$CONFIG_PATH" "$STAGE_NAME")"
case "$ENTRYPOINT" in
  train_insecure_code.py|train_insecure_code_qwen_eager_v2.py|train_insecure_code_qwen_eager_v3.py|evaluate_code_forecasts.py|evaluate_code_forecasts_qwen_eager_v2.py|evaluate_code_forecasts_qwen_eager_v3.py|evaluate_code_forecasts_qwen_eager_v4.py|evaluate_code_forecasts_qwen2026_v2.py) ;;
  *) echo "Unreviewed campaign entrypoint" >&2; exit 64 ;;
esac
if [[ "$STAGE_NAME" == evaluate || "$STAGE_NAME" == cache-gate ]]; then
  EVAL_ARGS=()
  if [[ "$STAGE_NAME" == cache-gate ]]; then EVAL_ARGS+=(--cache-diagnostic); fi
  exec "$TRAIN_PYTHON" "$CAMPAIGN/scripts/$ENTRYPOINT" \
    --config "$CONFIG_PATH" --arm "$ARM_NAME" --seed "$TRAIN_SEED" "${EVAL_ARGS[@]}"
fi
EXTRA_ARGS=()
if [[ "$STAGE_NAME" == gate ]]; then EXTRA_ARGS+=(--gate); fi
exec "$TRAIN_PYTHON" "$CAMPAIGN/scripts/$ENTRYPOINT" \
  --config "$CONFIG_PATH" --arm "$ARM_NAME" --seed "$TRAIN_SEED" "${EXTRA_ARGS[@]}"
