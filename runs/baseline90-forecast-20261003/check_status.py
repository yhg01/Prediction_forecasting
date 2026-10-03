#!/usr/bin/env python3
"""Read-only scheduler/receipt/progress check; does not submit or retry work."""
import datetime as dt,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'subliminal-forecast-20260929'))
from publish_general_migration import SSH
program=r'''
import datetime as dt,json,subprocess,importlib.util,fcntl
from pathlib import Path
root=Path('/projects/u6oz/yuhe/baseline90-forecast-20261003')
p=json.loads((root/'protocol.json').read_text())
records=[json.loads(f.read_text())|{'receipt':str(f)} for f in (root/'slurm').glob('submission-*.json')]
ids=[str(r['job_id']) for r in records if r.get('job_id')]
def call(args):
 r=subprocess.run(args,capture_output=True,text=True,check=True);return r.stdout
queue=call(['squeue','--all','--array','--me','--noheader','--format=%i|%j|%T|%b|%M|%R'])
accounting=call(['sacct','-n','-P','-j',','.join(ids),'--format=JobIDRaw,JobName,State,ExitCode,Start,End,AllocTRES']) if ids else ''
conditions=[]
for key in p['models']:
 for batch in (1,2):
  directory=root/'evaluations-general-v1'/key/f'base-batch{batch}'
  row={'key':key,'batch':batch}
  for name in ('progress.json','complete.json','manifest.json'):
   f=directory/name
   if f.exists(): row[name]=json.loads(f.read_text())
  conditions.append(row)
logs={f.name:f.read_text(errors='replace')[-4000:] for f in (root/'slurm').glob('*.err')}
s=importlib.util.spec_from_file_location('guard','/projects/u6oz/yuhe/jlens-single-claim-benchmarks-20260911-r1-ops-tqa-cap64-20260915/scripts/isambard/submit.py')
g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
with Path('/projects/u6oz/.jlens-subliminal-submit.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);lp,ledger=g.read_ledger(Path('/projects/u6oz'))
 shared=g.inspect_account(serialize=False,reservations=ledger['reservations'],new_gpus=0)
print(json.dumps({'checked_at':dt.datetime.now(dt.timezone.utc).isoformat(),'records':records,'squeue':queue,'sacct':accounting,'shared_accounting':shared,'conditions':conditions,'stderr_tails':logs}))
'''
r=subprocess.run(SSH+['/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python -'],input=program,capture_output=True,text=True,timeout=90)
if r.returncode: raise SystemExit(r.stderr or r.stdout)
d=json.loads(r.stdout);dest=ROOT/'checks'/dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ');dest.mkdir(parents=True,exist_ok=False)
(dest/'STATUS.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({'saved':str(dest/'STATUS.json'),'checked_at':d['checked_at'],'receipt_count':len(d['records']),'shared_gpus':d['shared_accounting']['existing_gpus'],'squeue':d['squeue'],'conditions':[{'key':x['key'],'batch':x['batch'],'progress':x.get('progress.json'),'complete':x.get('complete.json'),'initialized':'manifest.json' in x} for x in d['conditions']],'error_lines':{k:[line for line in v.splitlines() if any(term in line.lower() for term in ['traceback','out of memory','valueerror','runtimeerror','exception','error:'])] for k,v in d['stderr_tails'].items()}}))
