#!/usr/bin/env python3
"""Install a pinned campaign-only overlay using the prepared interpreter."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import requests

ROOT = Path(__file__).resolve().parent
PINS = ["transformers==5.8.0", "peft==0.18.1", "huggingface-hub==1.5.0",
        "tokenizers==0.22.2", "regex==2025.11.3"]
if sys.executable != "/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python":
    raise ValueError("Must use the prepared training interpreter")
bootstrap = ROOT / "private-pip"
overlay = ROOT / "runtime-2026"
bootstrap.mkdir(exist_ok=True)
response = requests.get("https://pypi.org/pypi/pip/25.2/json", timeout=30)
response.raise_for_status()
wheel = next(row for row in response.json()["urls"] if row["filename"].endswith("py3-none-any.whl"))
payload = requests.get(wheel["url"], timeout=60)
payload.raise_for_status()
if hashlib.sha256(payload.content).hexdigest() != wheel["digests"]["sha256"]:
    raise ValueError("Bootstrap wheel digest mismatch")
with zipfile.ZipFile(io.BytesIO(payload.content)) as archive:
    archive.extractall(bootstrap)
env = dict(os.environ, PYTHONPATH=str(bootstrap), PIP_DISABLE_PIP_VERSION_CHECK="1")
command = [sys.executable, "-m", "pip", "install", "--no-deps", "--only-binary=:all:",
           "--target", str(overlay), "--cache-dir", str(ROOT / "private-pip-cache"),
           "--report", str(ROOT / "runtime-2026-install.json"), *PINS]
subprocess.run(command, env=env, check=True)
check = """import json, importlib.metadata as m
from transformers import AutoModelForImageTextToText
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from peft import LoraConfig
assert 'qwen3_5' in CONFIG_MAPPING
print(json.dumps({k:m.version(k) for k in ['torch','transformers','peft','huggingface-hub','tokenizers','regex']}))
"""
result = subprocess.run([sys.executable, "-c", check], env=dict(os.environ, PYTHONPATH=str(overlay)),
                        text=True, capture_output=True)
proof = {"pins": PINS, "interpreter": sys.executable, "target": str(overlay),
         "check_exit_code": result.returncode, "check_stdout": result.stdout, "check_stderr": result.stderr}
(ROOT / "runtime-2026-check.json").write_text(json.dumps(proof, indent=2) + "\n")
print(json.dumps(proof), flush=True)
result.check_returncode()
