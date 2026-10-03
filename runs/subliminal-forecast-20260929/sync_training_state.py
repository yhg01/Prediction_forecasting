#!/usr/bin/env python3
"""Mirror small status/audit artifacts and bounded log tails; never submit jobs."""
import io
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile

from publish_general_migration import SSH

ROOT = Path(__file__).resolve().parent


def capture_evaluation_condition(directory):
    """Capture one fixed JSONL prefix and every raw response it references."""
    import hashlib
    import json
    import os
    from pathlib import Path

    directory = Path(directory)

    def capture(path):
        # A fixed length prevents following appends while copying a live file.
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if size >= 32 * 1024 * 1024:
                raise ValueError(f"Mirror input exceeds its size bound: {path}")
            data = stream.read(size)
        if len(data) != size:
            raise ValueError(f"Mirror input shrank during capture: {path}")
        return data

    files = {"manifest.json": capture(directory / "manifest.json")}
    # Read completion before results. A receipt created later belongs to a
    # later snapshot and can be picked up on the next mirror.
    complete_path = directory / "complete.json"
    complete = capture(complete_path) if complete_path.exists() else None
    result_path = directory / "results.jsonl"
    results = capture(result_path) if result_path.exists() else None
    if results is not None and results and not results.endswith(b"\n"):
        # A concurrently written trailing row is not yet a complete record.
        end = results.rfind(b"\n")
        results = results[:end + 1]
    rows = [json.loads(line) for line in (results or b"").splitlines() if line.strip()]
    identities = set()
    for row in rows:
        job_id = row["job_id"]
        if (not isinstance(job_id, str) or not job_id or job_id in identities
                or Path(job_id).name != job_id or job_id in (".", "..")):
            raise ValueError("Unsafe or duplicate forecast identity in mirror snapshot")
        identities.add(job_id)
        relative = f"raw/{job_id}.json"
        raw_path = directory / relative
        if not raw_path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Raw response escapes its evaluation condition")
        raw_bytes = capture(raw_path)
        if json.loads(raw_bytes).get("record") != row:
            raise ValueError(f"Raw response differs from captured result: {job_id}")
        files[relative] = raw_bytes
    for path in sorted(directory.glob("*.json")):
        if path.name not in ("manifest.json", "complete.json"):
            files[path.name] = capture(path)
    if results is not None:
        files["results.jsonl"] = results
    if complete is not None:
        receipt = json.loads(complete)
        if (results is None or receipt.get("status") != "completed"
                or receipt.get("forecast_count") != len(rows)
                or receipt.get("valid_count") != sum(row["status"] == "ok" for row in rows)
                or receipt.get("results_sha256") != hashlib.sha256(results).hexdigest()
                or receipt.get("manifest_sha256") != hashlib.sha256(files["manifest.json"]).hexdigest()):
            raise ValueError(f"Completion receipt differs from captured evaluation: {directory}")
        files["complete.json"] = complete
    return files


REMOTE = r'''
import datetime,io,json,pathlib,subprocess,sys,zipfile
root=pathlib.Path('/projects/u6oz/yuhe/insecure-code-forecast-20260929')
''' + inspect.getsource(capture_evaluation_condition) + r'''
subprocess.run(['/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python',str(root/'collect_status.py')],check=True,capture_output=True)
queue=subprocess.run(['squeue','--all','--array','--me','--format=%i|%j|%T|%M|%R'],check=True,capture_output=True,text=True).stdout
paths=set()
for pattern in ['slurm/submission-*.json','runs/*/*/*.json']:
 paths.update(root.glob(pattern))
for name in ['STATUS.md','status.json','general-event-migration.complete.json']:
 path=root/name
 if path.exists(): paths.add(path)
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
 for manifest in sorted(root.glob('evaluations-general-v1/*/*/manifest.json')):
  condition=manifest.parent
  captured=capture_evaluation_condition(condition)
  for relative,data in captured.items():
   archive.writestr(str((condition/relative).relative_to(root)),data)
 for path in sorted(paths):
  if path.is_file() and path.stat().st_size<32*1024*1024:
   archive.write(path,str(path.relative_to(root)))
 for pattern in ['slurm/*.out','slurm/*.err']:
  for path in root.glob(pattern):
   with path.open('rb') as source:
    size=path.stat().st_size
    source.seek(max(0,size-65536))
    archive.writestr('remote-log-tails/'+str(path.relative_to(root))+'.tail',source.read())
 archive.writestr('live_queue.json',json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'queue':queue.splitlines()}))
sys.stdout.buffer.write(buffer.getvalue())
'''

def main():
    result = subprocess.run(SSH + ["/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python -"],
                            input=REMOTE.encode(), capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    with zipfile.ZipFile(io.BytesIO(result.stdout)) as archive:
        names = archive.namelist()
        destinations = [(ROOT / name).resolve() for name in names]
        if any(not path.is_relative_to(ROOT.resolve()) for path in destinations):
            raise ValueError("Invalid mirror path")
        for name, path in zip(names, destinations):
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".mirror-", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(archive.read(name))
                os.replace(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    status = json.loads((ROOT / "status.json").read_text())
    print(json.dumps({"files_mirrored": len(names), "updated_at": status["updated_at"],
                      "staged_models": sum(m["staged"] for m in status["models"]),
                      "gates_passed": sum(a["gate"] == "passed" for m in status["models"] for a in m["arms"].values()),
                      "training_completed": sum(value == "completed" for m in status["models"] for a in m["arms"].values() for value in a["training_by_seed"].values())}))


if __name__ == "__main__":
    main()
