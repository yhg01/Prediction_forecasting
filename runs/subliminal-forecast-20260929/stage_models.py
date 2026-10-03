#!/usr/bin/env python3
"""Fetch only the immutable revisions cleared by preflight; resumable."""
import datetime as dt
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from requests.exceptions import RequestException

root = Path(__file__).resolve().parent
report = json.loads((root / "model_preflight.json").read_text())
parser = argparse.ArgumentParser()
parser.add_argument("--only", required=True, choices=[r["key"] for r in report["models"]])
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()
if not os.environ.get("SLURM_JOB_ID") or not 1 <= args.workers <= 8:
    raise ValueError("Large model staging requires a guarded Slurm allocation and 1–8 HTTP workers")
(root / f"staging_process-{args.only}.json").write_text(json.dumps({
    "pid": os.getpid(), "hostname": os.uname().nodename, "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "command": [sys.executable, *sys.argv]}, indent=2))
for row in sorted(report["models"], key=lambda r: (r["key"] != "qwen3_32b", r["key"])):
    if row["key"] != args.only:
        continue
    if row["key"].startswith("llama"):
        continue
    if row.get("weight_access") != "ok":
        continue
    destination = root / "models" / row["key"]
    receipt = root / "models" / (row["key"] + ".complete.json")
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["revision"] != row["revision"]:
            raise ValueError("Existing staged revision differs")
        continue
    print(json.dumps({"stage": "start", "model": row["model"], "revision": row["revision"]}), flush=True)
    source_info = HfApi().model_info(row["model"], revision=row["revision"], files_metadata=True)
    source_files = {f.rfilename: f for f in source_info.siblings}
    for attempt in range(1, 5):
        try:
            snapshot_download(row["model"], revision=row["revision"], local_dir=destination,
                allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.model", "*.tiktoken", "*.txt", "*.jinja", "README.md", "LICENSE*"],
                max_workers=args.workers)
            break
        except RequestException as exc:
            print(json.dumps({"event": "http_retry", "attempt": attempt, "error_type": type(exc).__name__}), flush=True)
            if attempt == 4:
                raise
            time.sleep(5 * attempt)
    files = []
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for part in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                h.update(part)
        relative = str(path.relative_to(destination))
        expected = source_files.get(relative)
        if expected is None or (expected.size is not None and expected.size != path.stat().st_size):
            raise ValueError(f"Unexpected or wrongly-sized snapshot artifact: {relative}")
        if expected.lfs and expected.lfs.sha256 != h.hexdigest():
            raise ValueError(f"Source SHA256 mismatch: {relative}")
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": h.hexdigest()})
    payload = {"model": row["model"], "revision": row["revision"], "files": files,
               "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(), "path": str(destination), "files_verified_sha256": True}
    receipt.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "model": row["model"], "receipt": str(receipt)}), flush=True)
