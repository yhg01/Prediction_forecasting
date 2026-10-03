#!/usr/bin/env python3
"""Mirror small status/audit artifacts and bounded log tails; never submit jobs."""
import io
import json
from pathlib import Path
import subprocess
import zipfile

from publish_general_migration import SSH

ROOT = Path(__file__).resolve().parent
REMOTE = r'''
import datetime,io,json,pathlib,subprocess,sys,zipfile
root=pathlib.Path('/projects/u6oz/yuhe/insecure-code-forecast-20260929')
subprocess.run(['/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python',str(root/'collect_status.py')],check=True,capture_output=True)
queue=subprocess.run(['squeue','--all','--array','--me','--format=%i|%j|%T|%M|%R'],check=True,capture_output=True,text=True).stdout
paths=set()
for pattern in ['slurm/submission-*.json','runs/*/*/*.json','evaluations-general-v1/*/*/*.json','evaluations-general-v1/*/*/results.jsonl']:
 paths.update(root.glob(pattern))
for name in ['STATUS.md','status.json','general-event-migration.complete.json']:
 path=root/name
 if path.exists(): paths.add(path)
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
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

result = subprocess.run(SSH + ["/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python -"],
                        input=REMOTE.encode(), capture_output=True, timeout=60)
if result.returncode:
    raise RuntimeError(result.stderr.decode(errors="replace"))
with zipfile.ZipFile(io.BytesIO(result.stdout)) as archive:
    names = archive.namelist()
    for name in names:
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError("Invalid mirror path")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.read(name))
status = json.loads((ROOT / "status.json").read_text())
print(json.dumps({"files_mirrored": len(names), "updated_at": status["updated_at"],
                  "staged_models": sum(m["staged"] for m in status["models"]),
                  "gates_passed": sum(a["gate"] == "passed" for m in status["models"] for a in m["arms"].values()),
                  "training_completed": sum(value == "completed" for m in status["models"] for a in m["arms"].values() for value in a["training_by_seed"].values())}))
