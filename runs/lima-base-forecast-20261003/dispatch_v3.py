"""Start each ready stage once. A new call checks for completed prerequisites."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from common import MODELS, SEEDS, verify_bundle
from runtime_v3 import verify_runtime


def main():
    p = argparse.ArgumentParser(); p.add_argument("--submit", action="store_true"); args = p.parse_args()
    root = Path(__file__).resolve().parent
    verify_runtime(root)
    has_data = (root / "data" / "complete.json").exists()
    for key in MODELS:
        ready_model = (root / "models" / (key + ".complete.json")).exists()
        tasks = [("stage", 0)] if not ready_model else [("base", 0)]
        if ready_model and has_data:
            if (root / "training" / key / "gate" / "complete.json").exists():
                tasks += [("train", seed) for seed in SEEDS]
            else:
                tasks += [("gate", 0)]
        for stage, seed in tasks:
            cmd = [sys.executable, str(root / "submit_v3.py"), "--model", key, "--stage", stage, "--seed", str(seed)]
            if args.submit: cmd.append("--submit")
            result = subprocess.run(cmd, text=True, capture_output=True)
            print(result.stdout, end="", flush=True)
            if result.returncode:
                print(result.stderr, file=sys.stderr); raise SystemExit(result.returncode)
    if not has_data:
        print(json.dumps({"status": "training_blocked", "reason": "Official LIMA access or an approved copy is required"}))


if __name__ == "__main__":
    main()
