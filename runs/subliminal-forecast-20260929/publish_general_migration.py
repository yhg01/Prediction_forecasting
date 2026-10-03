#!/usr/bin/env python3
"""Publish only this reviewed evaluation migration after Isambard authentication.

No downloads, jobs, credentials, shared environments or existing pipeline edits.
The default checks the local bundle; --publish uploads and verifies it remotely.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REMOTE = "/projects/u6oz/yuhe/insecure-code-forecast-20260929"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "AddKeysToAgent=no",
       "-o", "StrictHostKeyChecking=yes", "-o", "ControlMaster=no", "-o", "ControlPath=none",
       "-o", "ProxyCommand=ssh -o BatchMode=yes -o ConnectTimeout=10 -o AddKeysToAgent=no -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none -W %h:%p yuhegao.u6oz@jump.u6oz.aip2.isambard",
       "u6oz.aip2.isambard"]

REMOTE_INSTALLER = r'''
import base64, datetime, fcntl, hashlib, json, os, subprocess
from pathlib import Path
root = Path("/projects/u6oz/yuhe/insecure-code-forecast-20260929")
# Keep the same lock as every guarded submitter until all publication is durable.
# This interpreter performs one installation; an exception exits and releases it.
publication_lock = (root.parents[1] / ".jlens-subliminal-submit.lock").open("a")
fcntl.flock(publication_lock, fcntl.LOCK_EX)
bundle = json.loads(base64.b64decode(BUNDLE))
def sha(data): return hashlib.sha256(data).hexdigest()
trainer = root / "scripts/train_insecure_code.py"
if sha(trainer.read_bytes()) != bundle["unchanged_trainer_sha256"]:
    raise ValueError("Remote trainer differs; inspect before migration")
queue = subprocess.run(["squeue", "--all", "--array", "--me", "--noheader", "--format=%j"], text=True, capture_output=True, check=True)
if any(name.startswith("code-forecast-") and "-evaluate-" in name for name in queue.stdout.splitlines()):
    raise ValueError("An evaluation is queued or running; reconcile it before replacing evaluation code")
for name, item in bundle["files"].items():
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Invalid destination")
    content = base64.b64decode(item["base64"])
    if sha(content) != item["sha256"]:
        raise ValueError("Bundle hash differs")
    if name.startswith("configs/") and path.exists():
        previous, updated = json.loads(path.read_bytes()), json.loads(content)
        if {k:v for k,v in previous.items() if k != "evaluation"} != {k:v for k,v in updated.items() if k != "evaluation"}:
            raise ValueError("Remote training fields changed: " + name)
    if name.endswith(".py"):
        compile(content, name, "exec")
for name, item in bundle["files"].items():
    path = root / name
    content = base64.b64decode(item["base64"])
    if path.exists() and path.read_bytes() != content:
        old = path.read_bytes()
        archive = root / "history-per-problem-v1/remote-before-general" / sha(old) / name
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists() and archive.read_bytes() != old:
            raise ValueError("Archive conflict")
        archive.write_bytes(old)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".general-migration.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)
for name, item in bundle["files"].items():
    if sha((root / name).read_bytes()) != item["sha256"]:
        raise ValueError("Remote verification failed")
receipt = {"status": "published_and_verified", "published_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "files_sha256": {k:v["sha256"] for k,v in bundle["files"].items()},
           "unchanged_trainer_sha256": bundle["unchanged_trainer_sha256"]}
(root / "general-event-migration.complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
publication_lock.close()
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    bundle_path = ROOT / "general-event-deployment/bundle.json"
    bundle = json.loads(bundle_path.read_text())
    for name, item in bundle["files"].items():
        content = base64.b64decode(item["base64"])
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError("Invalid bundle entry: " + name)
    if not args.publish:
        print(json.dumps({"status": "local_bundle_verified", "files": len(bundle["files"]),
                          "bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
                          "remote_root": REMOTE, "next_step": "Reauthenticate Clifton, then use --publish"}))
        return
    encoded = base64.b64encode(bundle_path.read_bytes()).decode("ascii")
    program = "BUNDLE = " + repr(encoded) + "\n" + REMOTE_INSTALLER
    result = subprocess.run(SSH + ["/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python -"],
                            input=program, text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise SystemExit(result.stderr or "Remote migration failed")
    receipt = json.loads(result.stdout)
    (ROOT / "general-event-migration.complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "files": len(receipt["files_sha256"]),
                      "published_at": receipt["published_at"]}))


if __name__ == "__main__":
    main()
