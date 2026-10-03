import collections,datetime,fcntl,importlib.util,json,subprocess,sys
from pathlib import Path
root=Path('/projects/u6oz/yuhe/bad-advice-forecast-20260930');sys.path.insert(0,str(root))
from submit_training import GUARD,PROJECT
spec=importlib.util.spec_from_file_location('shared_guard',GUARD);guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
with (PROJECT/'.jlens-subliminal-submit.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 _,ledger=guard.read_ledger(PROJECT)
 cap=guard.inspect_account(serialize=False,reservations=ledger['reservations'],new_gpus=0)
submissions=[json.loads(p.read_text()) for p in (root/'slurm').glob('submission-*.json')]
ids=','.join(str(r['job_id']) for r in submissions if r.get('job_id'))
accounting=subprocess.run(['sacct','-X','-n','-P','-j',ids,'--format=JobID,JobName%70,State,ExitCode,Start,End,AllocTRES'],check=True,text=True,capture_output=True).stdout
procs=subprocess.run(['pgrep','-u',str(__import__('os').getuid()),'-af','launch_evaluations.py|launch_alignment.py'],text=True,capture_output=True)
state={}
for label,path in [('medical',root),('code',Path('/projects/u6oz/yuhe/insecure-code-forecast-20260929'))]:
 train=[]
 for cp in sorted((path/'configs').glob('*.json')):
  c=json.loads(cp.read_text())
  for arm in ['secure','insecure']:
   for seed in [0,1,2]:
    p=Path(c['output_root'])/f'{arm}-seed{seed}'/'complete.json'
    if p.exists():train.append({'model':c['model_key'],'arm':arm,'seed':seed,'receipt':json.loads(p.read_text())})
 evals=[]
 for p in path.glob('evaluations-general-v1/*/*/manifest.json'):
  d=p.parent;q=d/'results.jsonl';rows=q.read_bytes().splitlines() if q.exists() else []
  n=sum(1 for line in rows if line.strip())
  evals.append({'condition':str(d.relative_to(path)),'draws':n,'complete':(d/'complete.json').exists()})
 behavior=[]
 for p in path.glob('alignment-evaluations-v1/*/*/manifest.json'):
  d=p.parent;progress=d/'progress.json'
  behavior.append({'condition':str(d.relative_to(path)),'complete':(d/'complete.json').exists(),'progress':json.loads(progress.read_text()) if progress.exists() else None})
 state[label]={'training':train,'forecasts':evals,'behavior':behavior}
errors=[]
for p in root.glob('slurm/*.err'):
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-8192));tail=f.read().decode(errors='replace')
 lines=[line for line in tail.splitlines() if any(term in line for term in ['Traceback (most recent call last)','CUDA out of memory','RuntimeError:','ValueError:'])]
 if lines:errors.append({'file':p.name,'fatal_lines':lines[-6:]})
out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'capacity':cap,'submissions':submissions,'sacct':accounting.splitlines(),'launcher_processes':procs.stdout.splitlines(),'state':state,'error_signatures':errors}
import statistics
out['evaluation_timing']={}
for label,path in [('medical',root),('code',Path('/projects/u6oz/yuhe/insecure-code-forecast-20260929'))]:
 timing=[]
 for pattern,target in [('evaluations-general-v1/*/*/results.jsonl',30),('alignment-evaluations-v1/*/*/results.jsonl',80)]:
  for p in path.glob(pattern):
   if p.parent.name.startswith('pilot'):continue
   data=p.read_bytes();lines=data.splitlines()
   if data and not data.endswith(b'\n'):lines=lines[:-1]
   rows=[json.loads(x) for x in lines if x.strip()]
   if not rows:continue
   seconds=[(datetime.datetime.fromisoformat(x['completed_at'])-datetime.datetime.fromisoformat(x['started_at'])).total_seconds() for x in rows]
   last=datetime.datetime.fromisoformat(rows[-1]['completed_at']);recent=statistics.mean(seconds[-5:]);complete=(p.parent/'complete.json').exists()
   timing.append({'condition':str(p.parent.relative_to(path)),'draws':len(rows),'planned':target,'complete':complete,'mean_seconds_per_draw':statistics.mean(seconds),'recent5_seconds_per_draw':recent,'last_completed_at':rows[-1]['completed_at'],'estimated_finish_utc':(last+datetime.timedelta(seconds=max(0,target-len(rows))*recent)).isoformat()})
 out['evaluation_timing'][label]=timing
print(json.dumps(out))
