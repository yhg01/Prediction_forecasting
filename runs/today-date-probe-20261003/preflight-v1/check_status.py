#!/usr/bin/env python3
"""Read scheduler, shared accounting and probe evidence without changing jobs."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'subliminal-forecast-20260929'))
from publish_general_migration import SSH

program = 'EXPECTED_BUNDLE=' + repr(hashlib.sha256((ROOT / 'BUNDLE.json').read_bytes()).hexdigest()) + '\n' + r'''
import datetime as dt,json,subprocess,importlib.util,fcntl,hashlib
from pathlib import Path
root=Path('/projects/u6oz/yuhe/today-date-probe-20261003')
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert digest(root/'BUNDLE.json')==EXPECTED_BUNDLE,'Remote bundle changed'
bundle=json.loads((root/'BUNDLE.json').read_text())
for name,sha in bundle['files_sha256'].items():
 assert digest(root/name)==sha,'Remote source changed: '+name
p=json.loads((root/'protocol.json').read_text())
records=[json.loads(f.read_text())|{'receipt':str(f),'receipt_sha256':digest(f)} for f in (root/'slurm').glob('submission-*.json')]
ids=[str(r['job_id']) for r in records if r.get('job_id')]
assert len(ids)==len(set(ids)),'Duplicate accepted job ID'
def call(args):
 r=subprocess.run(args,capture_output=True,text=True,check=True);return r.stdout
queue=call(['squeue','--all','--array','--me','--noheader','--format=%i|%j|%T|%b|%M|%R'])
accounting=call(['sacct','-n','-P','-j',','.join(ids),'--format=JobIDRaw,JobName,State,ExitCode,Start,End,AllocTRES']) if ids else ''
conditions=[];model_complete={}
for key,entry in p['models'].items():
 directory=root/p['output_dir']/key
 if (directory/'complete.json').exists():model_complete[key]=json.loads((directory/'complete.json').read_text())
 for mode in entry['modes']:
  condition=directory/mode;row={'key':key,'mode':mode,'raw_count':len(list((condition/'raw').glob('*.json')))}
  for name in ('progress.json','complete.json','manifest.json'):
   file=condition/name
   if file.exists():row[name]=json.loads(file.read_text())
  conditions.append(row)
logs={f.name:f.read_text(errors='replace')[-5000:] for f in (root/'slurm').glob('*.err')}
gp=Path('/projects/u6oz/yuhe/jlens-single-claim-benchmarks-20260911-r1-ops-tqa-cap64-20260915/scripts/isambard/submit.py')
assert digest(gp)==bundle['guard_sha256']
spec=importlib.util.spec_from_file_location('guard',gp);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
with Path('/projects/u6oz/.jlens-subliminal-submit.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);lp,ledger=g.read_ledger(Path('/projects/u6oz'))
 shared=g.inspect_account(serialize=False,reservations=ledger['reservations'],new_gpus=0)
print(json.dumps({'checked_at':dt.datetime.now(dt.timezone.utc).isoformat(),'bundle_sha256':EXPECTED_BUNDLE,'records':records,'squeue':queue,'sacct':accounting,'shared_accounting':shared,'conditions':conditions,'model_complete':model_complete,'stderr_tails':logs}))
'''
result = subprocess.run(SSH + ['/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python -'], input=program, capture_output=True, text=True, timeout=60)
dest = ROOT / 'checks' / dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
dest.mkdir(parents=True, exist_ok=False)
(dest / 'ssh.stderr').write_text(result.stderr)
if result.returncode:
    (dest / 'FAILED.json').write_text(json.dumps({'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}, indent=2) + '\n')
    raise SystemExit(result.stderr or result.stdout)
data = json.loads(result.stdout)
(dest / 'STATUS.json').write_text(json.dumps(data, indent=2) + '\n')
print(json.dumps({'saved': str(dest / 'STATUS.json'), 'checked_at': data['checked_at'], 'shared_gpus': data['shared_accounting']['existing_gpus'],
                  'squeue': data['squeue'], 'completed_models': len(data['model_complete']),
                  'conditions': [{'key': c['key'], 'mode': c['mode'], 'raw_count': c['raw_count'], 'complete': 'complete.json' in c} for c in data['conditions']],
                  'error_lines': {k: [line for line in v.splitlines() if any(t in line.lower() for t in ('traceback','out of memory','valueerror','runtimeerror','exception','error:'))] for k,v in data['stderr_tails'].items()}}))
