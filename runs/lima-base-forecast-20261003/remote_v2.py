"""Use the corrected submitter and preserve every dispatch response."""
import argparse
import datetime as dt
import json
from pathlib import Path
import shlex
import subprocess
import sys
from common import REMOTE, TRAIN_PYTHON, digest, verify_bundle, write_json
from remote import SSH


def main():
    p = argparse.ArgumentParser(); p.add_argument("action", choices=("preview", "dispatch")); args = p.parse_args()
    root = Path(__file__).resolve().parent
    verify_bundle(root)
    amendment = json.loads((root / "OPERATIONAL_V2.json").read_text())
    if digest(root / "BUNDLE.json") != amendment["frozen_bundle_sha256"]:
        raise ValueError("The frozen scientific bundle changed")
    command = [TRAIN_PYTHON, REMOTE + "/dispatch_v2.py"]
    if args.action == "dispatch": command.append("--submit")
    result = subprocess.run(SSH + [shlex.join(command)], text=True, capture_output=True, timeout=60)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    write_json(root / "dispatch-records" / (stamp + ".json"), {
        "action": args.action, "command": command, "returncode": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr,
        "amendment_sha256": digest(root / "OPERATIONAL_V2.json")})
    print(result.stdout, end="")
    if result.stderr: print(result.stderr, file=sys.stderr)
    if result.returncode: raise SystemExit(result.returncode)


if __name__ == "__main__": main()
