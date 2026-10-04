"""Submit one forecast repeat through the shared compute guard."""
import argparse
import datetime as dt
import fcntl
import importlib.util
import json
from pathlib import Path
import re
from common import REMOTE, GUARD, digest, write_json
from finish_analysis import verify_analysis
from analyze_v3 import load_condition

RUN_ID = "qwen3-seed1-chatml-20261004"
ROOT = Path(__file__).resolve().parent


def verify_repeat(root):
    verify_analysis(root)
    bundle = json.loads((root / "RERUN_QWEN3_SEED1.json").read_text())
    if bundle["run_id"] != RUN_ID or bundle["model"] != "qwen3_8b" or bundle["training_seed"] != 1 or bundle["formats"] != ["chatml"]:
        raise ValueError("Repeat identity differs")
    for name, expected in {**bundle["files_sha256"], **bundle["inputs_sha256"]}.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError("Repeat input changed: " + name)
    protocol = json.loads((root / "protocol.json").read_text())
    if bundle["generation"] != protocol["generation"] or bundle["generation_seeds"] != "unchanged":
        raise ValueError("Repeat decoding settings differ")
    return bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    verify_repeat(ROOT)
    if ROOT.resolve() != Path(REMOTE).resolve():
        raise ValueError("Submit from the prepared remote campaign")
    for condition in ("base", "seed1"):
        if load_condition(ROOT, "qwen3_8b", condition, "chatml") is None:
            raise ValueError("Original forecast evidence is incomplete")
    if digest(GUARD) != json.loads((ROOT / "BUNDLE.json").read_text())["guard_sha256"]:
        raise ValueError("The shared submission guard changed")
    spec = importlib.util.spec_from_file_location("guard", GUARD)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    if guard.GPU_CAP != 64 or guard.ACCOUNT != "brics.u6oz":
        raise ValueError("The shared compute policy changed")
    out = ROOT / "reruns" / RUN_ID
    slurm = out / "slurm"
    slurm.mkdir(parents=True, exist_ok=True)
    work = {"run_dir": str(out), "stage": "generate", "seed": 1, "arm": "lima"}
    command = ["sbatch", "--parsable", f"--account={guard.ACCOUNT}", "--partition=workq",
               "--job-name=lima-qwen3-seed1-repeat", "--nodes=1", "--ntasks=1", "--cpus-per-task=16",
               "--gpus=1", "--time=04:00:00", "--no-requeue", f"--chdir={ROOT}",
               f"--output={slurm}/%j.out", f"--error={slurm}/%j.err", str(ROOT / "run_rerun_qwen3.sh")]
    project = Path("/projects/u6oz")
    with (project / ".jlens-subliminal-submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = list(slurm.glob("submission-*.json"))
        if previous:
            print(json.dumps({"status": "prior_attempt_preserved", "records": [str(p) for p in previous]}))
            return
        ledger_path, ledger = guard.read_ledger(project)
        accounting = guard.inspect_account(serialize=False, work=work, reservations=ledger["reservations"], new_gpus=0)
        if accounting["existing_gpus"] + 1 > guard.GPU_CAP:
            raise ValueError("The shared compute limit has no capacity")
        if not args.submit:
            print(json.dumps({"status": "preview", "command": command, "charge": 1, "accounting": accounting}))
            return
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = slurm / f"submission-{stamp}.json"
        accounting["new_gpus"] = 1
        record = {"command": command, "work": work, "accounting": accounting,
                  "rerun_bundle_sha256": digest(ROOT / "RERUN_QWEN3_SEED1.json")}
        reservation = {"job_id": None, "work": work, "record_path": str(path), "submitted_at_utc": stamp}
        guard.atomic_json(path, record)
        ledger["reservations"] = accounting["active_reservations"] + [reservation]
        guard.atomic_json(ledger_path, ledger)
        result = guard.command(command).strip()
        if not re.fullmatch(r"\d+(?:;[A-Za-z0-9_.-]+)?", result):
            raise ValueError("Submission is uncertain; preserve the reservation and reconcile the scheduler")
        record["job_id"] = reservation["job_id"] = result.split(";")[0]
        guard.atomic_json(ledger_path, ledger)
        guard.atomic_json(path, record)
        print(json.dumps({"status": "submitted", "job_id": record["job_id"], "charge": 1, "draws": 30}))


if __name__ == "__main__":
    main()
