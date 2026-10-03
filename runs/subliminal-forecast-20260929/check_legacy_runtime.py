#!/usr/bin/env python3
"""Read-only original-Qwen remote-class import using the private dependency."""
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
proof = {"target": str(ROOT / "runtime-qwen-legacy"), "interpreter": sys.executable,
         "pins": ["transformers-stream-generator==" + importlib.metadata.version("transformers-stream-generator")]}
try:
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cls = get_class_from_dynamic_module("modeling_qwen.QWenLMHeadModel",
        "/projects/u6oz/yuhe/millennium-forecast-20260929/models/qwen72b", local_files_only=True)
    proof.update(check_exit_code=0, model_class=cls.__name__)
except Exception as exc:
    proof.update(check_exit_code=1, error_type=type(exc).__name__, error=str(exc))
(ROOT / "runtime-qwen-legacy-check.json").write_text(json.dumps(proof, indent=2) + "\n")
print(json.dumps(proof), flush=True)
raise SystemExit(proof["check_exit_code"])
