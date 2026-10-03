#!/usr/bin/env python3
"""Publish the reviewed Qwen-only correction; no job submission."""
import base64
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from publish_general_migration import SSH

REMOTE = r'''
import base64,datetime,fcntl,hashlib,json,os,subprocess
from pathlib import Path
root=Path('/projects/u6oz/yuhe/insecure-code-forecast-20260929')
bundle=json.loads(base64.b64decode(PAYLOAD))
sha=lambda data:hashlib.sha256(data).hexdigest()
with Path('/projects/u6oz/.jlens-subliminal-submit.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 queue=subprocess.run(['squeue','--all','--array','--me','--noheader','--format=%j'],check=True,capture_output=True,text=True).stdout
 if any(n.startswith('code-forecast-qwen72b-') for n in queue.splitlines()): raise ValueError('Qwen job is active; inspect before replacement')
 for name,expected in bundle['other_configs_sha256'].items():
  if sha((root/'configs'/name).read_bytes())!=expected: raise ValueError('Other model config changed')
 for name,expected in [('train_insecure_code.py','f6200b2006e87d5c4eb49a84359c58819b3479f333cccfbea32bccf3f14ec22a'),('evaluate_code_forecasts.py','19d7637590e1788e01cb9e8ee4fcd1064d6322f15bb36f60dfb84917fe90020e')]:
  if sha((root/'scripts'/name).read_bytes())!=expected: raise ValueError('Original source changed')
 new_config=json.loads(base64.b64decode(bundle['files']['configs/qwen72b.json']['data']))
 current=json.loads((root/'configs/qwen72b.json').read_text())
 for key in ['model','data','training','training_seeds','gpus','private_runtime','private_runtime_check']:
  if current[key]!=new_config[key]: raise ValueError('Qwen scientific settings changed')
 for name,item in bundle['files'].items():
  target=(root/name).resolve();data=base64.b64decode(item['data'])
  if not target.is_relative_to(root.resolve()) or sha(data)!=item['sha256']: raise ValueError('Invalid payload')
  if name.endswith('.py'): compile(data,name,'exec')
 for name,item in bundle['files'].items():
  target=root/name;data=base64.b64decode(item['data'])
  if target.exists() and target.read_bytes()!=data:
   old=target.read_bytes();archive=root/'qwen72-gradient-v3/before-install'/sha(old)/name;archive.parent.mkdir(parents=True,exist_ok=True);archive.write_bytes(old)
  target.parent.mkdir(parents=True,exist_ok=True)
  tmp=target.with_name(target.name+'.qwen-v2.tmp');tmp.write_bytes(data);os.replace(tmp,target)
 actual={name:sha((root/name).read_bytes()) for name in bundle['files']}
 if actual!={name:item['sha256'] for name,item in bundle['files'].items()}: raise ValueError('Installed bytes differ')
 receipt={'published_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files_sha256':actual,'jobs_submitted':False}
 (root/'qwen72-gradient-v3/deployment.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
'''
payload=base64.b64encode((HERE/'bundle.json').read_bytes()).decode()
result=subprocess.run(SSH+['/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python -'],
                      input='PAYLOAD='+repr(payload)+'\n'+REMOTE,text=True,capture_output=True,timeout=60)
if result.returncode: raise RuntimeError(result.stderr)
receipt=json.loads(result.stdout)
(HERE/'deployment.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'published_at':receipt['published_at'],'files':len(receipt['files_sha256'])}))
