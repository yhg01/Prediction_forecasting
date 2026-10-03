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
review=json.loads((dest/'COMPLETION_REVIEW.json').read_text());assert review['status']=='passed'
for manifest in ['PREPARED_INPUTS.json','ALIGNMENT_BUNDLE.json']:
 for name,h in json.loads((root/manifest).read_text())['files'].items():
  assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
for stage in ['forecast','behavior']:
 for item in review['completed']:
  key,arm,seed=item['model'],item['arm'],item['seed'];condition=f'{arm}-seed{seed}'
  config=json.loads((root/'configs'/f'{key}.json').read_text())
  trained=Path(config['output_root'])/condition
  folder=root/(config['evaluation']['output_dir'] if stage=='forecast' else 'alignment-evaluations-v1')/key/condition
  records=[json.loads(p.read_text()) for p in (root/'slurm').glob('submission-*.json')]
  if has_attempt(folder,records) or (folder/'complete.json').exists():
   print(json.dumps({'stage':stage,'condition':str(folder),'status':'prior_attempt_preserved'}),flush=True);continue
  assert hashlib.sha256((trained/'complete.json').read_bytes()).hexdigest()==item['receipt_sha256']
  assert verify_receipt(trained,'completed',3000)['binding_sha256']==item['binding_sha256']
  command=[OPS,str(root/('submit_training.py' if stage=='forecast' else 'alignment/submit.py')),'--model-key',key,'--arm',arm,'--seed',str(seed)]
  if stage=='forecast':command+=['--stage','evaluate']
  command+=['--submit']
  for attempt in range(1,4):
   with (PROJECT/'.jlens-subliminal-submit.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);_,ledger=guard.read_ledger(PROJECT)
    cap=guard.inspect_account(serialize=False,reservations=ledger['reservations'],new_gpus=0)
    assert cap['existing_gpus']<=64
   (dest/f'capacity-{stage}-{key}-{condition}-{attempt}.json').write_text(json.dumps(cap,indent=2)+'\n')
   records=[json.loads(p.read_text()) for p in (root/'slurm').glob('submission-*.json')]
   assert not has_attempt(folder,records),'New or uncertain attempt appeared; stop and reconcile'
   p=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
   print(p.stdout,end='',flush=True)
   if not p.returncode:break
   if 'scontrol failed: slurm_load_jobs error: Invalid job id specified' not in p.stdout or attempt==3:raise SystemExit(p.returncode)
   # Next attempt first rechecks unchanged guard and refuses any prior reservation.
print(json.dumps({'status':'complete','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}),flush=True)
