"""Submit this experiment through the shared compute lock and ledger."""
import argparse
import datetime as dt
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
from common import MODELS, REMOTE, GUARD, TRAIN_PYTHON, SEEDS, digest, verify_bundle, write_json


from runtime_v3 import verify_runtime, EXPECTED_ROWS


def scheduler_command(root, args, guard, gpus, key):
    slurm = root / "slurm"
    command = ["sbatch", "--parsable", f"--account={guard.ACCOUNT}", "--partition=workq",
               f"--job-name=lima-{key}", "--nodes=1", "--ntasks=1", "--cpus-per-task=16",
               "--time=24:00:00", "--signal=B:USR1@180", "--no-requeue", f"--chdir={root}",
               f"--output={slurm}/%j.out", f"--error={slurm}/%j.err"]
    command += [f"--gpus={gpus}"] if gpus else ["--mem=24G"]
    command += [str(root / "run_v3.sh"), args.stage, args.model, str(args.seed)]
    return command


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--stage", choices=("stage", "base", "gate", "train"), required=True)
    p.add_argument("--seed", type=int, choices=SEEDS, default=0)
    p.add_argument("--submit", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    bundle, _ = verify_bundle(root)
    verify_runtime(root)
    amendment = json.loads((root / "OPERATIONAL_V2.json").read_text())
    if digest(root / "BUNDLE.json") != amendment["frozen_bundle_sha256"]:
        raise ValueError("The scientific source bundle changed")
    for name, expected in amendment["files_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError("The submission correction changed")
    if digest(root / "FAILED_SUBMISSION_RECONCILED.json") != amendment["reconciliation_sha256"]:
        raise ValueError("The failed submission reconciliation changed")
    if root.resolve() != Path(REMOTE).resolve():
        raise ValueError("Submission requires the prepared remote campaign")
    if args.stage != "train" and args.seed != 0:
        raise ValueError("Only training has three independent seeds")
    required = [root / "BUNDLE.json", root / "protocol.json", root / "run_v3.sh", root / "common.py", root / "runtime_v3.py", root / "RUNTIME_V3.json",
                root / ("stage.py" if args.stage == "stage" else "worker_v3.py"), Path(TRAIN_PYTHON)]
    if args.stage != "stage":
        required.append(root / "models" / (args.model + ".complete.json"))
    if args.stage in {"gate", "train"}:
        required.append(root / "data" / "complete.json")
        proof = json.loads(required[-1].read_text())
        if proof["status"] != "completed" or proof["training_rows"] != EXPECTED_ROWS:
            raise ValueError("The approved LIMA dataset is not ready")
        for name, expected in proof["files_sha256"].items():
            if digest(root / "data" / name) != expected:
                raise ValueError("The dataset changed")
    if args.stage == "train":
        required.append(root / "training" / args.model / "gate" / "complete.json")
        gate = json.loads(required[-1].read_text())
        if gate["status"] != "passed" or gate["optimizer_steps"] != 3:
            raise ValueError("The actual-model training diagnostic did not pass")
    if not all(f.is_file() for f in required):
        raise ValueError("Prepared inputs are missing")
    if digest(GUARD) != bundle["guard_sha256"]:
        raise ValueError("The shared submission guard changed")
    spec = importlib.util.spec_from_file_location("guard", GUARD)
    guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    if guard.GPU_CAP != 64 or guard.ACCOUNT != "brics.u6oz":
        raise ValueError("The shared compute policy changed")
    project = Path("/projects/u6oz")
    if Path(os.environ.get("PROJECTDIR", str(project))).resolve() != project.resolve():
        raise ValueError("Unexpected project")
    if args.stage == "stage":
        out = root / "models" / args.model
    elif args.stage == "base":
        out = root / "forecasts" / args.model / "base"
    else:
        out = root / "training" / args.model / ("gate" if args.stage == "gate" else f"seed{args.seed}")
    gpus = 0 if args.stage == "stage" else MODELS[args.model][3]
    charge = gpus or 4
    work = {"run_dir": str(out), "stage": "prefetch" if args.stage == "stage" else "generate" if args.stage == "base" else "train",
            "seed": args.seed, "arm": "base" if args.stage in {"stage", "base"} else "lima"}
    slurm = root / "slurm"
    key = f"{args.model}-{args.stage}-{args.seed}"
    command = scheduler_command(root, args, guard, gpus, key)
    with (project / ".jlens-subliminal-submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger_path, ledger = guard.read_ledger(project)
        previous = list(slurm.glob(f"submission-{key}-*.json"))
        reconciliation = json.loads((root / "FAILED_SUBMISSION_RECONCILED.json").read_text())
        previous = [path for path in previous if not (
            reconciliation["status"] == "reconciled_not_submitted"
            and reconciliation["failed_record_sha256"].get(str(path.relative_to(root))) == digest(path)
            and json.loads(path.read_text()).get("job_id") is None)]
        if previous:
            print(json.dumps({"status": "prior_attempt_preserved", "model": args.model, "stage": args.stage,
                              "seed": args.seed, "records": [str(f) for f in previous]}))
            return
        accounting = guard.inspect_account(serialize=False, work=work, reservations=ledger["reservations"], new_gpus=0)
        if accounting["existing_gpus"] + charge > guard.GPU_CAP:
            raise ValueError("The shared compute limit has no capacity for this job")
        if not args.submit:
            print(json.dumps({"status": "preview", "command": command, "accounting": accounting, "charge": charge}))
            return
        slurm.mkdir(exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        record_path = slurm / f"submission-{key}-{stamp}.json"
        accounting["new_gpus"] = charge
        record = {"command": command, "work": work, "accounting": accounting,
                  "inputs_sha256": {str(f): digest(f) for f in [Path(__file__), *required]}}
        guard.atomic_json(record_path, record)
        reservation = {"job_id": None, "work": work, "record_path": str(record_path), "submitted_at_utc": stamp}
        ledger["reservations"] = accounting["active_reservations"] + [reservation]
        guard.atomic_json(ledger_path, ledger)
        result = guard.command(command).strip()
        if not re.fullmatch(r"\d+(?:;[A-Za-z0-9_.-]+)?", result):
            raise ValueError("Submission status is uncertain; keep the reservation and reconcile the scheduler")
        record["job_id"] = reservation["job_id"] = result.split(";")[0]
        guard.atomic_json(ledger_path, ledger); guard.atomic_json(record_path, record)
        print(json.dumps({"status": "submitted", "job_id": record["job_id"], "model": args.model,
                          "stage": args.stage, "seed": args.seed, "gpus": gpus, "record": str(record_path)}))


if __name__ == "__main__":
    main()
