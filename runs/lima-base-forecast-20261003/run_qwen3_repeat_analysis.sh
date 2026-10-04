#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/lima-repeat-mpl
export XDG_CACHE_HOME=/tmp/lima-repeat-cache
exec /projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python /projects/u6oz/yuhe/lima-base-forecast-20261003/analyze_qwen3_repeat.py
