import datetime,fcntl,hashlib,importlib.util,json,math,subprocess,sys
from pathlib import Path
root=Path('/projects/u6oz/yuhe/bad-advice-forecast-20260930')
sys.path.insert(0,str(root))
from launch_ready import verify_receipt
from submit_training import GUARD,PROJECT
import torch
from safetensors import safe_open
torch.set_num_threads(2)
checks=[];incomplete=[]
for manifest in ['PREPARED_INPUTS.json','ALIGNMENT_BUNDLE.json']:
 for name,h in json.loads((root/manifest).read_text())['files'].items():
  assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
for cp in sorted((root/'configs').glob('*.json')):
 c=json.loads(cp.read_text())
 for arm in ['secure','insecure']:
  gate=json.loads((Path(c['output_root'])/f'gate-{arm}-seed0/manifest.json').read_text())
  for seed in [0,1,2]:
   d=Path(c['output_root'])/f'{arm}-seed{seed}'
   if not (d/'complete.json').exists():incomplete.append(str(d));continue
   rec=verify_receipt(d,'completed',3000)
   m=json.loads((d/'manifest.json').read_text());b=m['binding']
   assert not m['gate'] and m['seed']==seed
   assert b==gate['binding']
   for field in ['model','training','gpus']:assert b[field]==c[field],field
   assert b['data']==c['data'][arm]
   entry=c.get('training_entrypoint','train_insecure_code.py')
   assert b['implementation_sha256']==hashlib.sha256((root/'scripts'/entry).read_bytes()).hexdigest()
   audit=json.loads((d/'training_audit.json').read_text())
   assert len(audit['losses'])==3000 and {x['step'] for x in audit['losses']}==set(range(1,3001))
   assert all(math.isfinite(x['loss']) for x in audit['losses'])
   tensors=nonzero_b=0
   with safe_open(d/'final/adapter_model.safetensors',framework='pt',device='cpu') as f:
    for key in f.keys():
     t=f.get_tensor(key);assert torch.isfinite(t).all().item(),key
     if 'lora_B' in key:
      assert torch.count_nonzero(t).item()>0,key
      nonzero_b+=1
     tensors+=1
   assert tensors and nonzero_b
   checks.append(dict(model=c['model_key'],arm=arm,seed=seed,finite_tensors=tensors,nonzero_lora_b=nonzero_b,
                      receipt_sha256=hashlib.sha256((d/'complete.json').read_bytes()).hexdigest(),binding_sha256=m['binding_sha256'],completed_at=rec['completed_at']))
cap=None
submissions=[json.loads(p.read_text()) for p in (root/'slurm').glob('submission-*.json')]
ids=','.join(str(r['job_id']) for r in submissions if r.get('job_id'))
accounting=subprocess.run(['sacct','-X','-n','-P','-j',ids,'--format=JobID,JobName%70,State,ExitCode,Start,End,AllocTRES'],check=True,text=True,capture_output=True).stdout
out=dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='passed',completed=checks,incomplete=incomplete,capacity=cap,sacct=accounting.splitlines(),submissions=submissions)
dest=root/'heartbeats/20260930T1551';dest.mkdir(parents=True,exist_ok=True)
(dest/'COMPLETION_REVIEW_1605.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
