"""Operational dispatch restricted to independently validated completion receipts."""
import datetime,fcntl,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
root=Path('/projects/u6oz/yuhe/bad-advice-forecast-20260930')
sys.path.insert(0,str(root))
from launch_ready import OPS,verify_receipt
from launch_evaluations import has_attempt
from submit_training import GUARD,PROJECT
spec=importlib.util.spec_from_file_location('shared_guard',GUARD);guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
dest=root/'heartbeats/20260930T1745'
review=json.loads((dest/'COMPLETION_REVIEW_ALL.json').read_text());assert review['status']=='passed'
for manifest in ['PREPARED_INPUTS.json','ALIGNMENT_BUNDLE.json']:
 for name,h in json.loads((root/manifest).read_text())['files'].items():
  assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
assert len(review['completed'])==36 and not review['incomplete']
for stage,command in [('forecast',[OPS,str(root/'launch_evaluations.py'),'--submit']),('behavior',[OPS,str(root/'launch_alignment.py'),'--stage','evaluate','--submit'])]:
 for attempt in range(1,4):
  p=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  print(p.stdout,end='',flush=True)
  if not p.returncode:break
  if 'scontrol failed: slurm_load_jobs error: Invalid job id specified' not in p.stdout or attempt==3:raise SystemExit(p.returncode)
  with (PROJECT/'.jlens-subliminal-submit.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);_,ledger=guard.read_ledger(PROJECT)
   cap=guard.inspect_account(serialize=False,reservations=ledger['reservations'],new_gpus=0)
   assert cap['existing_gpus']<=64
  (dest/f'final-reconciliation-{stage}-{attempt}.json').write_text(json.dumps(cap,indent=2)+'\n')
print(json.dumps({'status':'complete','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}),flush=True)
