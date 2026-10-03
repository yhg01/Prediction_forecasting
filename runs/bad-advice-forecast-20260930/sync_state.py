#!/usr/bin/env python3
"""Read-only mirror of this campaign's receipts and bounded logs."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'subliminal-forecast-20260929'))
from publish_general_migration import SSH
from sync_training_state import capture_evaluation_condition
import inspect

REMOTE = "import pathlib, io, json, datetime, subprocess, sys, zipfile, hashlib\n" + inspect.getsource(capture_evaluation_condition) + r'''
root = pathlib.Path('/projects/u6oz/yuhe/bad-advice-forecast-20260930')
queue = subprocess.run(['squeue','--all','--array','--me','--format=%i|%j|%T|%M|%b'], capture_output=True, text=True, check=True).stdout
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    paths = set()
    for pattern in ['slurm/submission-*.json', 'runs/*/*/*.json', '*-validation.json', '*-launch.log', 'qwen72-cache-v4/gpu-diagnostic/*.json', 'BASELINE_REUSE.json']:
        paths.update(root.glob(pattern))
    for p in sorted(paths):
        if p.is_file() and p.stat().st_size < 32*1024*1024:
            z.write(p, str(p.relative_to(root)))
    for p in root.glob('evaluations-general-v1/*/*/manifest.json'):
        for name, data in capture_evaluation_condition(p.parent).items():
            z.writestr(str((p.parent/name).relative_to(root)), data)
    for p in root.glob('alignment-evaluations-v1/*/*/manifest.json'):
        directory=p.parent
        for name in ['manifest.json','progress.json','training.complete.json','runtime_backend_audit.json']:
            if (directory/name).exists():z.write(directory/name,str((directory/name).relative_to(root)))
        complete=directory/'complete.json'
        if complete.exists():
            proof=json.loads(complete.read_bytes());manifest=p.read_bytes();results=(directory/'results.jsonl').read_bytes()
            rows=[json.loads(x) for x in results.splitlines()]
            if (proof['status']!='completed' or proof['response_count']!=len(rows)
                or proof['manifest_sha256']!=hashlib.sha256(manifest).hexdigest()
                or proof['results_sha256']!=hashlib.sha256(results).hexdigest()):raise ValueError('Alignment receipt mismatch')
            identities=set()
            for row in rows:
                jid=row['job_id'];raw=directory/'raw'/(jid+'.json')
                if jid in identities or pathlib.Path(jid).name!=jid or not raw.resolve().is_relative_to(directory.resolve()):raise ValueError('Invalid alignment identity')
                identities.add(jid)
                data=raw.read_bytes()
                if json.loads(data)['record']!=row:raise ValueError('Alignment raw/result mismatch')
                z.writestr(str(raw.relative_to(root)),data)
            z.writestr(str((directory/'results.jsonl').relative_to(root)),results)
            z.write(complete,str(complete.relative_to(root)))
    for pattern in ['slurm/*.out', 'slurm/*.err']:
        for p in root.glob(pattern):
            with p.open('rb') as f:
                f.seek(max(0,p.stat().st_size-16384))
                z.writestr('remote-log-tails/'+str(p.relative_to(root))+'.tail',f.read())
    z.writestr('live_queue.json', json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'queue':queue.splitlines()}))
sys.stdout.buffer.write(buffer.getvalue())
'''

def main():
    result = subprocess.run(SSH + ['/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python -'], input=REMOTE.encode(), capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    with zipfile.ZipFile(io.BytesIO(result.stdout)) as archive:
        for name in archive.namelist():
            path = (ROOT/name).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError('Mirror escaped campaign')
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name+'.mirror-tmp')
            tmp.write_bytes(archive.read(name))
            os.replace(tmp,path)
    print(json.dumps({'files':len(archive.namelist()), 'queue':json.loads((ROOT/'live_queue.json').read_text())}))

if __name__ == '__main__':
    main()
