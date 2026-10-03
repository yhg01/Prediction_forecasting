#!/usr/bin/env python3
"""Stage only the two accessible exact release snapshots into this campaign."""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from requests.exceptions import RequestException

ROOT = Path(__file__).resolve().parent
MODELS = [
    ("r1_distill_32b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "2a29ab14a7dcfb5132537e18050d0ebe5008f7fb"),
    ("qwen72b", "Qwen/Qwen-72B-Chat", "2cd9f76279337941ec1a4abeec6f8eb3c38d0f55"),
]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--only', choices=[row[0] for row in MODELS])
parser.add_argument('--workers', type=int, default=8)
args = parser.parse_args()
if not 1 <= args.workers <= 8:
    parser.error('Streaming worker count must be between 1 and 8')

def save(path, value):
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)

save(ROOT / f'staging_process-{args.only or "all"}.json', {'pid': os.getpid(), 'hostname': os.uname().nodename,
                                    'started_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                                    'workers': args.workers, 'only': args.only,
                                    'command': [sys.executable, *sys.argv],
                                    'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                                    'log_path': os.environ.get('STAGING_LOG_PATH')})
for key, repo_id, revision in MODELS:
    if args.only and key != args.only:
        continue
    target = ROOT / "models" / key
    receipt = ROOT / f"staged-{key}.json"
    if receipt.exists():
        existing = json.loads(receipt.read_text())
        if existing["revision"] != revision:
            raise ValueError("Existing staging receipt names another revision")
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    selected = [s for s in info.siblings if (
        s.rfilename.endswith((".safetensors", ".json", ".py", ".tiktoken", ".model", ".cu", ".cpp"))
        or s.rfilename in ("LICENSE", "NOTICE", "README.md")
    )]
    print(json.dumps({"event": "stage_start", "model_key": key, "revision": revision,
                      "expected_bytes": sum(s.size or 0 for s in selected)}), flush=True)
    for attempt in range(1, 5):
        try:
            snapshot_download(repo_id, revision=revision, local_dir=target,
                              allow_patterns=[s.rfilename for s in selected], max_workers=args.workers)
            break
        except RequestException as exc:
            print(json.dumps({'event': 'http_retry', 'attempt': attempt,
                              'error_type': type(exc).__name__}), flush=True)
            if attempt == 4:
                raise
            time.sleep(5 * attempt)
    files = []
    for item in selected:
        path = target / item.rfilename
        if not path.is_file() or (item.size is not None and path.stat().st_size != item.size):
            raise ValueError(f"Staged file missing or size mismatch: {item.rfilename}")
        row = {"name": item.rfilename, "bytes": path.stat().st_size}
        if item.lfs:
            row["source_sha256"] = item.lfs.sha256
        if not item.rfilename.endswith(".safetensors"):
            row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(row)
    save(receipt, {"model_key": key, "repo_id": repo_id, "revision": revision,
                   "local_path": str(target), "precision": "original bfloat16 weights; no quantization",
                   "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(), "files": files})
    print(json.dumps({"event": "stage_complete", "model_key": key, "receipt": str(receipt)}), flush=True)
