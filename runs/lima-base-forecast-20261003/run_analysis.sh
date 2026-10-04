#!/bin/bash
set -euo pipefail
campaign_dir=/projects/u6oz/yuhe/lima-base-forecast-20261003
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="$campaign_dir/analysis/.mplconfig"
export XDG_CACHE_HOME="$campaign_dir/analysis/.cache"
exec /projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python "$campaign_dir/finish_analysis.py"
