#!/usr/bin/env python3
"""Stage exact small tokenizer artifacts and audit all rows without model weights."""
import json
import os
from pathlib import Path
import subprocess
import sys
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
def main():
    for config_path in sorted((ROOT / "configs").glob("*.json")):
        if len(sys.argv) > 1 and config_path.stem not in sys.argv[1:]:
            continue
        config = json.loads(config_path.read_text())
        model = config["model"]
        snapshot_download(model["name"], revision=model["revision"], local_dir=model["path"],
            allow_patterns=["*.json", "*.py", "*.tiktoken", "*.model", "*.txt", "*.jinja"], max_workers=2)
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
        env["PYTHONPATH"] = os.pathsep.join(p for p in [config["private_runtime"], str(ROOT / "scripts")] if p)
        result = subprocess.run([sys.executable, str(Path(__file__)), "--worker", str(config_path)], env=env)
        print(json.dumps({"model_key": config["model_key"], "audit_exit_code": result.returncode}), flush=True)

def worker(config_path):
    from transformers import AutoTokenizer
    from train_insecure_code import prepare_rows, write_json
    config = json.loads(Path(config_path).read_text())
    model = config["model"]
    result = {"model_key": config["model_key"], "revision": model["revision"], "arms": {}}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model["path"], local_files_only=True,
                                                  trust_remote_code=model["trust_remote_code"])
        for arm in ("insecure", "secure"):
            _, audit = prepare_rows(tokenizer, config["data"][arm]["path"], model["model_type"], 2048)
            result["arms"][arm] = audit
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    write_json(ROOT / "tokenizer-audits" / (config["model_key"] + ".json"), result)
    print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        main()
