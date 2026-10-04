"""Queue the CI update after the complete seed 1 forecast repeat."""
import argparse
import datetime as dt
import fcntl
import importlib.util
import json
from pathlib import Path
import re
from common import REMOTE, GUARD, digest
from rerun_qwen3_submit import verify_repeat, RUN_ID

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    verify_repeat(ROOT)
    if ROOT.resolve() != Path(REMOTE).resolve():
        raise ValueError("Use the prepared remote campaign")
    bundle = json.loads((ROOT / "QWEN3_REPEAT_ANALYSIS.json").read_text())
    for name, expected in bundle["files_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Repeat analysis source changed")
    if bundle["rerun_bundle_sha256"] != digest(ROOT / "RERUN_QWEN3_SEED1.json"):
        raise ValueError("Repeat analysis binding differs")
    folder = ROOT / "reruns" / RUN_ID
    slurm = folder / "slurm"
    forecast_records = list(slurm.glob("submission-*.json"))
    if len(forecast_records) != 1:
        raise ValueError("One forecast submission is required")
    forecast_job = json.loads(forecast_records[0].read_text()).get("job_id")
    if not isinstance(forecast_job, str) or not re.fullmatch(r"\d+", forecast_job):
        raise ValueError("The forecast submission is uncertain")
    if digest(GUARD) != json.loads((ROOT / "BUNDLE.json").read_text())["guard_sha256"]:
        raise ValueError("The shared guard changed")
    spec = importlib.util.spec_from_file_location("guard", GUARD)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    if guard.GPU_CAP != 64 or guard.ACCOUNT != "brics.u6oz":
        raise ValueError("Shared compute policy changed")
    work = {"run_dir": str(folder / "analysis-ci"), "stage": "analyze", "seed": 1, "arm": "lima"}
    command = ["sbatch", "--parsable", f"--account={guard.ACCOUNT}", "--partition=workq",
               "--job-name=lima-qwen3-repeat-ci", "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
               "--mem=24G", "--time=00:30:00", "--no-requeue", f"--dependency=afterok:{forecast_job}",
               f"--chdir={ROOT}", f"--output={slurm}/ci-%j.out", f"--error={slurm}/ci-%j.err",
               str(ROOT / "run_qwen3_repeat_analysis.sh")]
    project = Path("/projects/u6oz")
    with (project / ".jlens-subliminal-submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = list(slurm.glob("analysis-submission-*.json"))
        if previous:
            print(json.dumps({"status": "prior_attempt_preserved", "records": [str(p) for p in previous]}))
            return
        ledger_path, ledger = guard.read_ledger(project)
        accounting = guard.inspect_account(serialize=False, work=work, reservations=ledger["reservations"], new_gpus=0)
        if accounting["existing_gpus"] + 4 > guard.GPU_CAP:
            raise ValueError("No capacity for the analysis job")
        if not args.submit:
            print(json.dumps({"status": "preview", "command": command, "charge": 4}))
            return
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = slurm / f"analysis-submission-{stamp}.json"
        record = {"command": command, "work": work, "accounting": accounting, "dependency": forecast_job,
                  "source_bundle_sha256": digest(ROOT / "QWEN3_REPEAT_ANALYSIS.json")}
        accounting["new_gpus"] = 4
        reservation = {"job_id": None, "work": work, "record_path": str(path), "submitted_at_utc": stamp}
        guard.atomic_json(path, record)
        ledger["reservations"] = accounting["active_reservations"] + [reservation]
        guard.atomic_json(ledger_path, ledger)
        result = guard.command(command).strip()
        if not re.fullmatch(r"\d+(?:;[A-Za-z0-9_.-]+)?", result):
            raise ValueError("Analysis submission is uncertain; preserve the reservation")
        record["job_id"] = reservation["job_id"] = result.split(";")[0]
        guard.atomic_json(ledger_path, ledger)
        guard.atomic_json(path, record)
        print(json.dumps({"status": "submitted", "job_id": record["job_id"], "dependency": forecast_job}))


if __name__ == "__main__":
    main()
