#!/usr/bin/env python3
"""Preview or submit one-GPU u6oz experiments with aggregate accounting."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

PROJECT = "u6oz"
ACCOUNT = "brics.u6oz"  # Live Slurm association for the u6oz project.
GPU_CAP = 64  # User authorized 2026-09-13.
JOB_PREFIX = "jlens-subliminal-"
TERMINAL_STATES = {"BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT"}


def command(args: list[str]) -> str:
    environment = os.environ.copy()
    if args[0] == "sbatch":
        # An inherited SBATCH_EXCLUSIVE/ARRAY/GPUS_PER_NODE must never turn the
        # reviewed one-GPU command into a larger or fanned-out allocation.
        environment = {key: value for key, value in environment.items() if not key.startswith("SBATCH_")}
    try:
        result = subprocess.run(args, text=True, capture_output=True, check=True, env=environment)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise ValueError(f"{args[0]} failed: {detail}") from exc
    return result.stdout


def fields(line: str) -> dict[str, str]:
    # Slurm's -o key/value records can have spaces in job names and command paths.
    matches = list(re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=", line))
    return {match.group(1): line[match.end(): matches[i + 1].start() if i + 1 < len(matches) else None].strip()
            for i, match in enumerate(matches)}


def gpu_tres(value: str) -> int | None:
    """Read total/typed GPU TRES without double-counting the typed subtotal."""
    if value in {"", "(null)", "N/A", "Unknown"}:
        return None
    total = None
    typed = 0
    found = False
    for item in value.split(","):
        key, separator, raw = item.partition("=")
        if key == "gres/gpu" or key.startswith("gres/gpu:"):
            if not separator or not raw.isdigit():
                raise ValueError(f"Cannot account for GPU TRES {item!r}")
            found = True
            if key == "gres/gpu":
                total = int(raw)
            else:
                typed += int(raw)
    if not found:
        return 0
    if total is not None and typed > total:
        raise ValueError(f"Inconsistent typed and total GPU TRES: {value}")
    return total if total is not None else typed


def job_gpu_count(record: dict[str, str]) -> int:
    if not record.get("Account"):
        raise ValueError("Slurm job account is missing")
    if not record.get("JobState"):
        raise ValueError("Slurm job state is missing")
    if record["JobState"] in TERMINAL_STATES:
        return 0
    request = gpu_tres(record.get("ReqTRES", ""))
    allocation = gpu_tres(record.get("AllocTRES", ""))
    if request is None and allocation is None:
        raise ValueError("Neither requested nor allocated TRES is available")
    count = max(request or 0, allocation or 0)
    # An exclusive node can occupy all four Superchips even if ReqTRES says one.
    # A job without GPU TRES is also charged conservatively as whole nodes.
    # Older Slurm exposes exclusivity through OverSubscribe rather than
    # WholeNode; newer versions also expose a separate Exclusive field. An
    # explicit restrictive/unknown mode always wins over a sharing indicator.
    sharing_known = False
    exclusive_or_unknown = False
    for field, shared_values in (("WholeNode", {"0", "NO"}),
                                 ("Exclusive", {"0", "NO", "NONE"}),
                                 ("OverSubscribe", {"OK", "YES"})):
        value = record.get(field, "")
        if value:
            sharing_known = True
            exclusive_or_unknown |= value not in shared_values
    if count == 0 or exclusive_or_unknown or not sharing_known:
        nodes = record.get("NumNodes", "")
        # A pending allocation can report e.g. NumNodes=1-1, or a genuine
        # min-max range. Charge its maximum, never merely the minimum.
        match = re.fullmatch(r"([0-9]+)(?:-([0-9]+))?", nodes)
        if not match:
            raise ValueError(f"Cannot account conservatively for reserved nodes: NumNodes={nodes!r}")
        minimum, maximum = int(match[1]), int(match[2] or match[1])
        if minimum < 1 or maximum < minimum:
            raise ValueError(f"Cannot account conservatively for reserved nodes: NumNodes={nodes!r}")
        count = max(count, 4 * maximum)
    return count


def job_details(job_id: str) -> dict[str, str]:
    detail = command(["scontrol", "show", "job", "--oneliner", job_id]).strip().splitlines()
    if len(detail) != 1:
        raise ValueError(f"Expected exactly one accounting record for {job_id}")
    return fields(detail[0])


def recorded_job_details(job_id: str) -> dict[str, str]:
    """Recover jobs absent from the queue without forgetting a reservation."""
    try:
        return job_details(job_id)
    except ValueError:
        rows = command(["sacct", "--allocations", "--noheader", "--parsable2",
                        "--jobs", job_id, "--format=JobIDRaw,State,ExitCode"]).strip().splitlines()
        matches = [row.split("|") for row in rows if row.split("|")[0] == job_id]
        if len(matches) != 1 or len(matches[0]) < 2:
            raise ValueError(f"Cannot reconcile recorded job {job_id}; refusing to release its GPU reservation")
        state = matches[0][1].split()[0].rstrip("+")
        if state not in TERMINAL_STATES:
            raise ValueError(f"Recorded job {job_id} is {state}; live GPU accounting is unavailable")
        return {"Account": ACCOUNT, "JobState": state, "ExitCode": matches[0][2] if len(matches[0]) > 2 else ""}


def recorded_job_details_many(job_ids: list[str]) -> dict[str, dict[str, str]]:
    """Classify recorded jobs in two RPCs, never use this for GPU accounting.

    A live snapshot can conservatively remain active until the next tick. Jobs
    absent from that snapshot require exactly one terminal accounting record;
    missing or delayed accounting therefore cannot make work ready for retry.
    The submission transaction still independently reconciles its full ledger.
    """
    if any(not isinstance(value, str) or not re.fullmatch(r"\d+", value) for value in job_ids):
        raise ValueError("Recorded job IDs must be unambiguous individual numeric IDs")
    requested = set(job_ids)
    if not requested:
        return {}
    queue = command(["squeue", "--all", "--array", "--noheader", "--me",
                     "--states=all", "--format=%i|%a|%T"])
    result = {}
    seen = set()
    for line in queue.splitlines():
        parts = [value.strip() for value in line.split("|")]
        if (len(parts) != 3 or not re.fullmatch(r"\d+(?:_\d+)?(?:\+\d+)?", parts[0])
                or not parts[1] or not parts[2]):
            raise ValueError(f"Unrecognised squeue classification record: {line!r}")
        job_id, account, state = parts
        if job_id in seen:
            raise ValueError(f"Duplicate scheduler job ID: {job_id}")
        seen.add(job_id)
        if job_id not in requested:
            continue
        if account != ACCOUNT:
            raise ValueError(f"Recorded job {job_id} belongs to unexpected account {account}")
        state = state.split()[0].rstrip("+")
        if state not in TERMINAL_STATES:
            result[job_id] = {"Account": account, "JobState": state, "ExitCode": ""}
    missing = sorted(requested - result.keys(), key=int)
    if missing:
        rows = command(["sacct", "--allocations", "--noheader", "--parsable2",
                        "--jobs", ",".join(missing), "--format=JobIDRaw,Account,State%40,ExitCode"]).splitlines()
        records = {}
        for line in rows:
            parts = [value.strip() for value in line.split("|")]
            # parsable2 normally has no final delimiter; accept either form.
            if len(parts) == 5 and parts[-1] == "":
                parts.pop()
            if len(parts) != 4 or parts[0] not in missing:
                raise ValueError(f"Unexpected accounting classification record: {line!r}")
            job_id, account, state, exit_code = parts
            if job_id in records:
                raise ValueError(f"Ambiguous accounting records for recorded job {job_id}")
            if account != ACCOUNT or not state or not re.fullmatch(r"\d+:\d+", exit_code):
                raise ValueError(f"Incomplete or mismatched accounting record for {job_id}")
            state = state.split()[0].rstrip("+")
            if state not in TERMINAL_STATES:
                raise ValueError(f"Recorded job {job_id} is {state}; live classification is unavailable")
            records[job_id] = {"Account": account, "JobState": state, "ExitCode": exit_code}
        unresolved = set(missing) - records.keys()
        if unresolved:
            raise ValueError(f"Cannot reconcile recorded jobs {sorted(unresolved, key=int)}; refusing to classify them")
        result.update(records)
    return result


def same_work(left: dict, right: dict) -> bool:
    if left.get("run_dir") != right.get("run_dir"):
        return False
    a, b = left.get("stage"), right.get("stage")
    # All-target scoring and fitting each own their complete output directory.
    if {a, b} & {"jlens-all", "fit-instrument", "causal-audit", "gemma-audit", "adapter-audit"}:
        return True
    if any(left.get(key) != right.get(key) for key in ("seed", "arm")):
        return False
    # Different instrument files do not make concurrent writes to the same
    # target safe. Keep those JLens operations serialized as well.
    if "jlens" in {a, b} and bool({a, b} & {"train", "student"}):
        return True
    return a == b or ("student" in {a, b} and bool({a, b} & {"train", "evaluate", "jlens"}))


def inspect_account(*, serialize: bool = True, work: dict | None = None,
                    reservations: list[dict] | None = None, new_gpus: int = 1) -> dict:
    if new_gpus not in (0, 1):
        raise ValueError("This helper only accounts for zero or one additional GPU")
    config = command(["scontrol", "show", "config"])
    private_match = re.search(r"^\s*PrivateData\s*=\s*(.*)$", config, re.MULTILINE)
    private = private_match.group(1).strip().lower() if private_match else "unknown"
    # -a includes hidden partitions; -r expands arrays so every task is counted.
    # Isambard hides other users' jobs. The user's limit applies to experiments
    # we start; conservatively include ALL current-user jobs across accounts.
    queue = command(["squeue", "--all", "--array", "--noheader", "--me",
                     "--states=all", "--format=%i|%a"])
    jobs = []
    seen = set()
    for line in queue.splitlines():
        job_id, separator, account = line.strip().partition("|")
        if not separator or not account.strip() or not re.fullmatch(r"\d+(?:_\d+)?(?:\+\d+)?", job_id):
            raise ValueError(f"Unrecognised squeue record: {line!r}")
        if job_id in seen:
            raise ValueError(f"Duplicate scheduler job ID: {job_id}")
        seen.add(job_id)
        record = job_details(job_id)
        if record.get("Account") != account.strip():
            raise ValueError(f"Account mismatch between queue and details for {job_id}")
        count = job_gpu_count(record)
        if record["JobState"] not in TERMINAL_STATES:
            jobs.append({"job_id": job_id, "name": record.get("JobName", ""),
                         "state": record["JobState"], "gpus_reserved_or_requested": count,
                         "slurm": record})
    active_reservations = []
    for reservation in reservations or []:
        if not isinstance(reservation, dict):
            raise ValueError("Invalid reservation in submission ledger")
        job_id = reservation.get("job_id")
        if not isinstance(job_id, str) or not re.fullmatch(r"\d+", job_id):
            raise ValueError("An earlier submission has an unknown job ID; inspect its recorded command and scheduler before retrying")
        if job_id in seen and job_id not in {job["job_id"] for job in jobs}:
            continue
        if job_id not in seen:
            record = recorded_job_details(job_id)
            if record["JobState"] in TERMINAL_STATES:
                continue
            jobs.append({"job_id": job_id, "name": record.get("JobName", ""),
                         "state": record["JobState"], "gpus_reserved_or_requested": job_gpu_count(record),
                         "slurm": record})
            seen.add(job_id)
        active_reservations.append(reservation)
        if work and same_work(work, reservation.get("work", {})):
            raise ValueError(f"The same run/stage/seed/arm already has active or pending job {job_id}")
    total = sum(job["gpus_reserved_or_requested"] for job in jobs)
    if new_gpus and total + new_gpus > GPU_CAP:
        raise ValueError(f"Submission refused: current user has {total} allocated/requested GPUs; adding one exceeds {GPU_CAP}")
    if serialize and any(job["name"].startswith(JOB_PREFIX) for job in jobs):
        raise ValueError("A JLens subliminal job is already running or queued; v0 runs are serialized")
    if not serialize and any(job["name"].startswith(JOB_PREFIX) and not job["name"].startswith(JOB_PREFIX + "v1-") for job in jobs):
        raise ValueError("An active v0 run is serialized; wait for it before starting v1")
    return {"project": PROJECT, "account": ACCOUNT, "accounting_scope": "current_user_all_accounts", "private_data": private,
            "gpu_cap": GPU_CAP, "existing_gpus": total, "new_gpus": new_gpus,
            "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "jobs": jobs,
            "active_reservations": active_reservations}


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_ledger(project_dir: Path) -> tuple[Path, dict]:
    ledger_path = project_dir / f".jlens-subliminal-submissions-{os.getuid()}.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"version": 1, "reservations": []}
    if not isinstance(ledger, dict) or ledger.get("version") != 1 or not isinstance(ledger.get("reservations"), list):
        raise ValueError(f"Invalid submission ledger: {ledger_path}")
    return ledger_path, ledger


def inspect_capacity(project_dir: Path = Path("/projects/u6oz")) -> dict:
    """Non-submitting snapshot for a dispatcher; main rechecks under the lock."""
    # Readers must not observe the deliberately incomplete intent between
    # sbatch and the writer's final job-ID commit. Use the exact same lock as
    # main; an abandoned intent remains an error once its writer releases it.
    # Opening in append mode only initializes the empty lock on first use.
    with (project_dir / ".jlens-subliminal-submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        _, ledger = read_ledger(project_dir)
        result = inspect_account(serialize=False, reservations=ledger["reservations"], new_gpus=0)
    result["available_gpus"] = max(0, GPU_CAP - result["existing_gpus"])
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "pilot", "v1"], default="smoke")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, help="Override the selected profile's YAML")
    parser.add_argument("--run-dir", type=Path, required=True, help="Persistent run directory under $PROJECTDIR")
    parser.add_argument("--time", help="HH:MM:SS walltime, at most 24 hours (default smoke 00:30:00, pilot 12:00:00)")
    parser.add_argument("--stage", choices=["generate", "train", "evaluate", "jlens", "jlens-all", "fit-instrument", "causal-audit", "gemma-audit", "adapter-audit", "student", "analyze"], help="Required for v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", choices=["belief", "neutral", "base", "teacher"], default="belief")
    parser.add_argument("--dependency", help="Wait for successful completion, e.g. afterok:123 or afterok:123:124")
    parser.add_argument("--python", type=Path, help="Explicit prepared interpreter; v1 uses it without uv syncing the v0 lock")
    parser.add_argument("--instrument", type=Path, help="Required independent instrument configuration for jlens/jlens-all")
    parser.add_argument("--fit-prompts", type=int, help="fit-instrument corpus size: 2 for timing smoke, 100 for production (default 100)")
    parser.add_argument("--fit-only", action="store_true", help="fit-instrument: defer target instrument materialization")
    parser.add_argument("--audit-data", type=Path, help="causal-audit/gemma-audit/adapter-audit: existing numeric training JSONL on persistent project storage")
    parser.add_argument("--audit-adapter", type=Path, help="adapter-audit only: completed adapter directory on persistent project storage")
    parser.add_argument("--submit", action="store_true", help="Perform checks and submit; default only prints command")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.profile == "v1" and (args.stage is None or args.config is None or args.python is None):
        raise ValueError("v1 requires --stage, --config and --python for its isolated environment")
    if args.profile != "v1" and (args.stage is not None or args.python is not None or args.seed != 0 or args.arm != "belief"):
        raise ValueError("--stage, --python, --seed and --arm are v1 options")
    is_audit = args.profile == "v1" and args.stage in {"causal-audit", "gemma-audit", "adapter-audit"}
    is_adapter_audit = is_audit and args.stage == "adapter-audit"
    if is_audit and args.audit_data is None:
        raise ValueError(f"{args.stage} requires explicit --audit-data")
    if args.audit_data is not None and not is_audit:
        raise ValueError("--audit-data is supported only for v1 causal-audit, gemma-audit or adapter-audit")
    if is_adapter_audit and args.audit_adapter is None:
        raise ValueError("adapter-audit requires explicit --audit-adapter")
    if args.audit_adapter is not None and not is_adapter_audit:
        raise ValueError("--audit-adapter is supported only for v1 adapter-audit")
    if is_audit:
        if args.seed != 0 or args.arm not in {"belief", "base"}:
            raise ValueError("Model audits use explicit inputs; experiment --seed/--arm selection does not apply")
        args.arm = "base"
    is_fit = args.profile == "v1" and args.stage == "fit-instrument"
    if not is_fit and (args.fit_prompts is not None or args.fit_only):
        raise ValueError("--fit-prompts and --fit-only are supported only for v1 fit-instrument")
    if is_fit:
        if args.seed != 0 or args.arm not in {"belief", "base"}:
            raise ValueError("fit-instrument fits the base model; experiment --seed/--arm selection does not apply")
        args.arm = "base"
        args.fit_prompts = 100 if args.fit_prompts is None else args.fit_prompts
        if args.fit_prompts < 1:
            raise ValueError("--fit-prompts must be a positive integer")
        # An abbreviated timing fit must never produce production instrument
        # pointers, even if the caller omits the explicit --fit-only option.
        args.fit_only = args.fit_only or args.fit_prompts != 100
    is_jlens = args.profile == "v1" and args.stage in {"jlens", "jlens-all"}
    if is_jlens and args.instrument is None:
        raise ValueError("JLens stages require an explicit --instrument; no JLens job will be submitted without one")
    if args.instrument is not None and not is_jlens:
        raise ValueError("--instrument is supported only for v1 jlens/jlens-all")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.dependency and not re.fullmatch(r"afterok:\d+(?::\d+)*", args.dependency):
        raise ValueError("--dependency must be afterok:JOB_ID[:JOB_ID...]")
    repo = args.repo.resolve()
    config = args.config or Path("configs/subliminal") / f"{args.profile}.yaml"
    config = (repo / config).resolve() if not config.is_absolute() else config.resolve()
    run_dir = args.run_dir.resolve()
    instrument = (repo / args.instrument).resolve() if args.instrument is not None else None
    audit_data = (repo / args.audit_data).resolve() if args.audit_data is not None else None
    audit_adapter = (repo / args.audit_adapter).resolve() if args.audit_adapter is not None else None
    if instrument is not None and not instrument.is_file():
        raise ValueError(f"Instrument configuration does not exist: {instrument}")
    if audit_data is not None and (not audit_data.is_file() or audit_data.stat().st_size == 0):
        raise ValueError(f"Audit data must be an existing nonempty file: {audit_data}")
    adapter_inputs = {}
    if audit_adapter is not None:
        for path in (audit_adapter / "adapter_config.json", audit_adapter / "adapter_model.safetensors",
                     audit_adapter.parent / "manifest.json", audit_adapter.parent / "complete.json"):
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Audit adapter requires completed persistent artifacts: {path}")
            adapter_inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    walltime = args.time or ("00:30:00" if args.profile == "smoke" else "12:00:00")
    match = re.fullmatch(r"(\d{1,2}):([0-5]\d):([0-5]\d)", walltime)
    if not match or not 0 < sum(int(v) * m for v, m in zip(match.groups(), (3600, 60, 1))) <= 86400:
        raise ValueError("--time must be HH:MM:SS and between one second and 24 hours")
    if not config.is_file():
        raise ValueError(f"Config does not exist: {config}")
    # Use this checkout's current policy, including when --repo names a frozen
    # historical checkout. Keep accounting-only imports dependency-free.
    source_root = str(Path(__file__).resolve().parents[2])
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from experiments.subliminal.model_scope import require_submission_model
    selected_model = require_submission_model(config)
    wrapper = repo / "scripts/isambard/run.sh"
    if not wrapper.is_file():
        raise ValueError(f"Job wrapper does not exist: {wrapper}")
    # B: sends USR1 to the batch shell 180 seconds before walltime; it forwards
    # to the pipeline process group so normal checkpoint handlers can finish.
    work = {"run_dir": str(run_dir), "stage": args.stage or "run", "seed": args.seed, "arm": args.arm}
    if instrument is not None:
        work["instrument"] = str(instrument)
    if is_fit:
        work.update(fit_prompts=args.fit_prompts, fit_only=args.fit_only)
    if audit_data is not None:
        work.update(audit_data=str(audit_data), audit_data_sha256=hashlib.sha256(audit_data.read_bytes()).hexdigest())
    if audit_adapter is not None:
        work.update(audit_adapter=str(audit_adapter), audit_adapter_inputs_sha256=adapter_inputs)
    identity = hashlib.sha256(json.dumps(work, sort_keys=True).encode()).hexdigest()[:16]
    name = f"{JOB_PREFIX}{args.profile}" + (f"-{args.stage}-{identity}" if args.profile == "v1" else "")
    cmd = ["sbatch", "--parsable", f"--account={ACCOUNT}", "--partition=workq",
           f"--job-name={name}", "--nodes=1", "--ntasks=1", "--gpus=1",
           f"--time={walltime}", "--signal=B:USR1@180", "--no-requeue",
           f"--chdir={repo}", f"--output={run_dir}/slurm/%j.out", f"--error={run_dir}/slurm/%j.err"]
    if args.dependency:
        cmd.extend([f"--dependency={args.dependency}", "--kill-on-invalid-dep=yes"])
    cmd.extend([str(wrapper), str(repo), str(config), str(run_dir)])
    if args.profile == "v1":
        cmd.extend(["v1", args.stage, str(args.seed), args.arm, str(args.python.absolute())])
        if instrument is not None:
            cmd.append(str(instrument))
        elif is_fit:
            cmd.extend([str(args.fit_prompts), str(int(args.fit_only))])
        elif audit_data is not None:
            cmd.append(str(audit_data))
            if audit_adapter is not None:
                cmd.append(str(audit_adapter))
    print(shlex.join(cmd), flush=True)
    if not args.submit:
        print("Preview only: no directory created, accounting queried, or job submitted. Add --submit on Isambard.")
        return 0
    project_dir = Path(os.environ.get("PROJECTDIR", "/projects/u6oz")).resolve()
    expected_project = Path("/projects/u6oz").resolve()
    if project_dir != expected_project:
        raise ValueError(f"PROJECTDIR must resolve to /projects/u6oz; got {project_dir}")
    if not run_dir.is_relative_to(project_dir) or run_dir == project_dir:
        raise ValueError("--run-dir must be a subdirectory of /projects/u6oz for persistent checkpoints")
    if not repo.is_relative_to(project_dir):
        raise ValueError("--repo must be under /projects/u6oz so queued jobs retain their source tree")
    if instrument is not None and not instrument.is_relative_to(project_dir):
        raise ValueError("--instrument must be under /projects/u6oz so queued jobs retain its configuration")
    if audit_data is not None and not audit_data.is_relative_to(project_dir):
        raise ValueError("--audit-data must be under /projects/u6oz so queued jobs retain its input data")
    if audit_adapter is not None and not audit_adapter.is_relative_to(project_dir):
        raise ValueError("--audit-adapter must be under /projects/u6oz so queued jobs retain its input adapter")
    if args.python and (not args.python.is_file() or not os.access(args.python, os.X_OK)):
        raise ValueError(f"Prepared interpreter does not exist or is not executable: {args.python}")
    for executable in ("squeue", "scontrol", "sacct", "sbatch") + (() if args.python else ("uv",)):
        if shutil.which(executable) is None:
            raise ValueError(f"Required executable is not on PATH: {executable}")
    # Serialize the check+submit transactions of every invocation in the project.
    # The queue snapshot includes pending jobs, even ones not started by us.
    lock_path = project_dir / ".jlens-subliminal-submit.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger_path, ledger = read_ledger(project_dir)
        accounting = inspect_account(serialize=args.profile != "v1", work=work, reservations=ledger["reservations"])
        slurm_dir = run_dir / "slurm"
        slurm_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        record_path = slurm_dir / f"submission-{stamp}.json"
        record = {"command": cmd, "accounting": accounting, "work": work,
                  "model": selected_model}
        atomic_json(record_path, record)
        reservation = {"job_id": None, "work": work, "record_path": str(record_path), "submitted_at_utc": stamp}
        ledger["reservations"] = accounting["active_reservations"] + [reservation]
        # Record intent before sbatch: an interrupted/ambiguous submission must
        # never silently disappear from the cap or duplicate-work protection.
        atomic_json(ledger_path, ledger)
        result = command(cmd).strip()
        if not re.fullmatch(r"\d+(?:;[A-Za-z0-9_.-]+)?", result):
            raise ValueError(f"sbatch returned an unexpected response; inspect squeue before retrying: {result!r}")
        record["job_id"] = result.split(";")[0]
        reservation["job_id"] = record["job_id"]
        atomic_json(ledger_path, ledger)
        atomic_json(record_path, record)
        print(f"Submitted job {record['job_id']}; record: {record_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"Submission stopped: {exc}", file=sys.stderr)
        raise SystemExit(1)
