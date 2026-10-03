#!/usr/bin/env python3
"""Compact persistent state for heartbeat resumption; never submits jobs."""
import datetime as dt
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
status = {"updated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "training_seeds": [0, 1, 2],
          "event_id": "any_millennium", "forecasts_per_condition": 30,
          "expected_training_runs": 36, "expected_evaluation_conditions": 42, "models": [], "submissions": []}
for config_path in sorted((ROOT / "configs").glob("*.json")):
    config = json.loads(config_path.read_text())
    row = {"model_key": config["model_key"], "revision": config["model"]["revision"],
           "staged": Path(config["model"]["staging_receipt"]).exists(), "arms": {}}
    audit = ROOT / "tokenizer-audits" / (config["model_key"] + ".json")
    row["tokenizer_audit"] = json.loads(audit.read_text()).get("status") if audit.exists() else "missing"
    for arm in ("secure", "insecure"):
        current = {}
        for stage, directory in (("gate", f"gate-{arm}-seed0"), ("train", f"{arm}-seed0")):
            receipt = Path(config["output_root"]) / directory / "complete.json"
            current[stage] = json.loads(receipt.read_text()).get("status") if receipt.exists() else "not_complete"
        evaluation = ROOT / config["evaluation"]["output_dir"] / config["model_key"] / f"{arm}-seed0" / "complete.json"
        current["evaluation"] = json.loads(evaluation.read_text()) if evaluation.exists() else None
        current["training_by_seed"] = {}
        current["evaluation_by_seed"] = {}
        for seed in config["training_seeds"]:
            receipt = Path(config["output_root"]) / f"{arm}-seed{seed}" / "complete.json"
            current["training_by_seed"][str(seed)] = json.loads(receipt.read_text()).get("status") if receipt.exists() else "not_complete"
            evaluation = ROOT / config["evaluation"]["output_dir"] / config["model_key"] / f"{arm}-seed{seed}" / "complete.json"
            current["evaluation_by_seed"][str(seed)] = json.loads(evaluation.read_text()) if evaluation.exists() else None
        row["arms"][arm] = current
    base = ROOT / config["evaluation"]["output_dir"] / config["model_key"] / "base" / "complete.json"
    row["base_evaluation"] = json.loads(base.read_text()) if base.exists() else None
    status["models"].append(row)
for record_path in sorted((ROOT / "slurm").glob("submission-*.json")):
    record = json.loads(record_path.read_text())
    status["submissions"].append({"job_id": record.get("job_id"), "work": record["work"],
        "requested_gpus": record["accounting"].get("requested_gpus", record["accounting"]["new_gpus"]),
        "accounted_gpus": record["accounting"]["new_gpus"], "record": str(record_path)})
queue = subprocess.run(["squeue", "--all", "--array", "--me", "--noheader", "--format=%i|%T|%R|%M"], text=True, capture_output=True)
status["queue_exit_code"] = queue.returncode
status["queue_snapshot"] = queue.stdout.splitlines()
status["queue_error"] = queue.stderr
(ROOT / "status.json").write_text(json.dumps(status, indent=2) + "\n")
lines = ["# Campaign status", "", "Updated: " + status["updated_at"], "",
         "Scope: six models, secure/insecure code, training seeds0/1/2; 36 production runs and 42 matched local forecasting conditions (six shared bases), each with30 forecasts of the single general Millennium event (1260 total).", "",
         "| Model | Staged | Token audit | Secure gate/train0,1,2 | Insecure gate/train0,1,2 |", "| --- | --- | --- | --- | --- |"]
for row in status["models"]:
    lines.append(f'| {row["model_key"]} | {row["staged"]} | {row["tokenizer_audit"]} | '
                 f'{row["arms"]["secure"]["gate"]}/{",".join(row["arms"]["secure"]["training_by_seed"].values())} | '
                 f'{row["arms"]["insecure"]["gate"]}/{",".join(row["arms"]["insecure"]["training_by_seed"].values())} |')
lines += ["", "Live scheduler snapshot:", "", "```text", *status["queue_snapshot"], "```", "",
          "See HANDOFF.md for exact resume commands. Status reads receipt markers; workers independently verify artifact hashes before reuse.", ""]
(ROOT / "STATUS.md").write_text("\n".join(lines))
print(json.dumps({"updated_at": status["updated_at"], "staged_models": sum(row["staged"] for row in status["models"]),
    "completed_training_runs": sum(value == "completed" for row in status["models"] for arm in row["arms"].values() for value in arm["training_by_seed"].values()),
    "job_ids": [row["job_id"] for row in status["submissions"]]}), flush=True)
