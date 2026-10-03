#!/usr/bin/env python3
"""New campaign submitter using the reviewed shared lock/ledger transaction."""
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex

ROOT = Path("/projects/u6oz/yuhe/insecure-code-forecast-20260929")
PROJECT = Path("/projects/u6oz")
GUARD = Path("/projects/u6oz/yuhe/jlens-single-claim-benchmarks-20260911-r1-ops-tqa-cap64-20260915/scripts/isambard/submit.py")
PYTHON = Path("/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python")
ALLOWED = {
    "qwen72b": ("Qwen/Qwen-72B-Chat", "2cd9f76279337941ec1a4abeec6f8eb3c38d0f55", 2),
    "qwen25_72b": ("Qwen/Qwen2.5-72B-Instruct", "a13fff9ad76700c7ecff2769f75943ba8395b4a7", 2),
    "r1_distill_32b": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "2a29ab14a7dcfb5132537e18050d0ebe5008f7fb", 1),
    "qwen3_32b": ("Qwen/Qwen3-32B", "30b8421510892303dc5ddd6cd0ac90ca2053478d", 1),
    "qwen35_27b": ("Qwen/Qwen3.5-27B", "a3ca5719420477ab4390cf6262d6de65e8871c37", 1),
    "qwen38_27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", 1),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=ALLOWED, required=True)
    parser.add_argument("--arm", choices=("insecure", "secure", "base"), required=True)
    parser.add_argument("--seed", type=int, choices=(0,), required=True)
    parser.add_argument("--stage", choices=("gate", "train", "evaluate"), required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.stage == "gate" and args.seed != 0:
        raise ValueError("Diagnostic gates use seed zero")
    if args.arm == "base" and args.stage != "evaluate":
        raise ValueError("The base model is evaluated without training")
    config_path = ROOT / "configs" / (args.model_key + ".json")
    config = json.loads(config_path.read_text())
    model = config["model"]
    identity = (model["name"], model["revision"], config["gpus"])
    if identity != ALLOWED[args.model_key]:
        raise ValueError("Config model/resource request differs from authorized panel")
    receipt_path = Path(model["staging_receipt"])
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("model", receipt.get("repo_id")), receipt["revision"]) != identity[:2]:
        raise ValueError("Staging identity mismatch")
    model_path = Path(model["path"]).resolve()
    if model_path != Path(receipt.get("path", receipt.get("local_path"))).resolve():
        raise ValueError("Staging path mismatch")
    for row in receipt["files"]:
        path = model_path / row.get("path", row.get("name"))
        if not path.resolve().is_relative_to(model_path) or path.stat().st_size != row["bytes"]:
            raise ValueError("Staged file absent or size changed")
    # Full hashes are independently checked by the GPU worker before any training.
    required = [ROOT / "run_training.sh", ROOT / "scripts/train_insecure_code.py", PYTHON,
                config_path, receipt_path]
    if args.arm != "base":
        required.append(Path(config["data"][args.arm]["path"]))
    if args.stage == "evaluate":
        required += [ROOT / "scripts/evaluate_code_forecasts.py", ROOT / "scripts/run_forecasts.py",
                     ROOT / "forecast-inputs/prompts.jsonl", ROOT / "forecast-inputs/models.json"]
    if not all(p.is_file() for p in required):
        raise ValueError("Prepared inputs missing")
    if config["private_runtime"]:
        proof_path = Path(config["private_runtime_check"])
        proof = json.loads(proof_path.read_text())
        if proof["check_exit_code"] != 0 or Path(proof["target"]).resolve() != Path(config["private_runtime"]).resolve():
            raise ValueError("Private runtime import preflight failed")
        required.append(proof_path)
    spec = importlib.util.spec_from_file_location("shared_isambard_guard", GUARD)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    if guard.GPU_CAP != 64 or guard.ACCOUNT != "brics.u6oz":
        raise ValueError("Shared guard policy changed")
    if Path(os.environ.get("PROJECTDIR", str(PROJECT))).resolve() != PROJECT.resolve():
        raise ValueError("Unexpected PROJECTDIR")
    suffix = ("gate-" if args.stage == "gate" else "") + f"{args.arm}-seed{args.seed}"
    run_dir = Path(config["output_root"]) / suffix
    if args.stage == "evaluate":
        run_dir = ROOT / "evaluations" / args.model_key / ("base" if args.arm == "base" else suffix)
    if (run_dir / "complete.json").exists():
        raise ValueError("Already complete; inspect receipt rather than submit duplicate")
    if args.stage == "train":
        gate_path = Path(config["output_root"]) / f"gate-{args.arm}-seed0" / "complete.json"
        gate = json.loads(gate_path.read_text())
        if gate["status"] != "passed" or gate["optimizer_steps"] != 3:
            raise ValueError("Matching arm has not passed its three-step GPU gate")
        required.append(gate_path)
    if args.stage == "evaluate" and args.arm != "base":
        trained = Path(config["output_root"]) / suffix / "complete.json"
        proof = json.loads(trained.read_text())
        if proof["status"] != "completed" or proof["optimizer_steps"] != 3000:
            raise ValueError("Cannot evaluate an incomplete production adapter")
        required.append(trained)
    work = {"run_dir": str(run_dir), "stage": "generate" if args.stage == "evaluate" else "train",
            "seed": args.seed, "arm": args.arm}
    key = f"{args.model_key}-{args.stage}-{args.arm}-{args.seed}"
    slurm = ROOT / "slurm"
    command = ["sbatch", "--parsable", f"--account={guard.ACCOUNT}", "--partition=workq",
        f"--job-name=code-forecast-{key}", "--nodes=1", "--ntasks=1", f"--gpus={config['gpus']}",
        "--cpus-per-task=16", "--time=" + ("02:00:00" if args.stage == "gate" else "24:00:00"),
        "--signal=B:USR1@180", "--no-requeue", f"--chdir={ROOT}",
        f"--output={slurm}/%j.out", f"--error={slurm}/%j.err",
        str(ROOT / "run_training.sh"), str(config_path), args.arm, str(args.seed), args.stage]
    print(shlex.join(command), flush=True)
    if not args.submit:
        print("Preview only; no GPU reservation or scheduler submission.", flush=True)
        return
    slurm.mkdir(exist_ok=True)
    with (PROJECT / ".jlens-subliminal-submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger_path, ledger = guard.read_ledger(PROJECT)
        accounting = guard.inspect_account(serialize=False, work=work, reservations=ledger["reservations"], new_gpus=0)
        if accounting["existing_gpus"] + config["gpus"] > guard.GPU_CAP:
            raise ValueError("Submission would exceed shared 64-GPU aggregate limit")
        accounting["new_gpus"] = config["gpus"]
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        record_path = slurm / f"submission-{key}-{stamp}.json"
        record = {"command": command, "work": work, "accounting": accounting,
            "inputs_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), *required]}}
        guard.atomic_json(record_path, record)
        reservation = {"job_id": None, "work": work, "record_path": str(record_path), "submitted_at_utc": stamp}
        ledger["reservations"] = accounting["active_reservations"] + [reservation]
        guard.atomic_json(ledger_path, ledger)
        result = guard.command(command).strip()
        if not re.fullmatch(r"\d+(?:;[A-Za-z0-9_.-]+)?", result):
            raise ValueError("Ambiguous submission; retain reservation and reconcile scheduler before retry")
        job_id = result.split(";")[0]
        record["job_id"] = reservation["job_id"] = job_id
        guard.atomic_json(ledger_path, ledger)
        guard.atomic_json(record_path, record)
        print(json.dumps({"job_id": job_id, "requested_gpus": config["gpus"], "record": str(record_path)}), flush=True)


if __name__ == "__main__":
    main()
