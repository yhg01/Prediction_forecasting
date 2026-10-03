#!/usr/bin/env python3
"""Install only the reviewed 2026 token-ID return-type correction."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT))
from publish_general_migration import SSH
files={name:(ROOT/name).read_bytes() for name in ['submit_training.py','run_training.sh','build_campaign.py','configs/qwen35_27b.json','configs/qwen38_27b.json','qwen2026-tokenids-v2/README.md']}
files['scripts/evaluate_code_forecasts_qwen2026_v2.py']=(REPO/'scripts/evaluate_code_forecasts_qwen2026_v2.py').read_bytes()
bundle={name:{'data':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest()} for name,data in files.items()}
(HERE/'bundle.json').write_text(json.dumps(bundle,indent=2)+'\n')
REMOTE=r'''
import base64,datetime,fcntl,hashlib,json,os,subprocess
from pathlib import Path
root=Path('/projects/u6oz/yuhe/insecure-code-forecast-20260929')
bundle=json.loads(base64.b64decode(PAYLOAD));sha=lambda data:hashlib.sha256(data).hexdigest()
with Path('/projects/u6oz/.jlens-subliminal-submit.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 queue=subprocess.run(['squeue','--all','--array','--me','--noheader','--format=%j'],check=True,capture_output=True,text=True).stdout
 if any(n.startswith(('code-forecast-qwen35_27b-evaluate-','code-forecast-qwen38_27b-evaluate-')) for n in queue.splitlines()): raise ValueError('2026 evaluation is still active')
 states=subprocess.run(['sacct','--allocations','--noheader','--parsable2','--jobs=6960127,6960130','--format=JobIDRaw,State,ExitCode'],check=True,capture_output=True,text=True).stdout
 rows={p[0]:p[1] for line in states.splitlines() if (p:=line.split('|')) and len(p)>2}
 if any(not rows.get(j,'').startswith('FAILED') for j in ['6960127','6960130']): raise ValueError('Expected failed evaluation attempts were not terminal')
 for key in ['qwen35_27b','qwen38_27b']:
  name='configs/'+key+'.json';old=json.loads((root/name).read_text());new=json.loads(base64.b64decode(bundle[name]['data']))
  if {k:v for k,v in old.items() if k!='evaluation'}!={k:v for k,v in new.items() if k!='evaluation'}: raise ValueError('Training settings changed')
  prior=root/'evaluations-general-v1'/key/'base';manifest=prior/'manifest.json'
  if manifest.exists() and json.loads(manifest.read_text())['inference_binding']['evaluator_sha256']=='19d7637590e1788e01cb9e8ee4fcd1064d6322f15bb36f60dfb84917fe90020e':
   results=prior/'results.jsonl'
   if results.exists() and results.stat().st_size: raise ValueError('Cannot archive nonempty scientific outcomes')
   archived=root/'qwen2026-tokenids-v2/failed-base'/key
   if archived.exists(): raise ValueError('Failed attempt archive already exists')
   archived.parent.mkdir(parents=True,exist_ok=True);prior.rename(archived)
 for name,item in bundle.items():
  target=(root/name).resolve();data=base64.b64decode(item['data'])
  if not target.is_relative_to(root.resolve()) or sha(data)!=item['sha256']: raise ValueError('Invalid input')
  if name.endswith('.py'): compile(data,name,'exec')
  if target.exists() and target.read_bytes()!=data:
   old=target.read_bytes();archive=root/'qwen2026-tokenids-v2/before-install'/sha(old)/name;archive.parent.mkdir(parents=True,exist_ok=True);archive.write_bytes(old)
  target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_name(target.name+'.tokenids-v2.tmp');temp.write_bytes(data);os.replace(temp,target)
 actual={name:sha((root/name).read_bytes()) for name in bundle}
 if actual!={name:item['sha256'] for name,item in bundle.items()}: raise ValueError('Installed hashes differ')
 receipt={'published_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files_sha256':actual,'failed_job_states':rows,'jobs_submitted':False}
 (root/'qwen2026-tokenids-v2/deployment.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
'''
encoded=base64.b64encode(json.dumps(bundle).encode()).decode()
result=subprocess.run(SSH+['/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python -'],input='PAYLOAD='+repr(encoded)+'\n'+REMOTE,text=True,capture_output=True,timeout=60)
if result.returncode:raise RuntimeError(result.stderr)
receipt=json.loads(result.stdout);(HERE/'deployment.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'published_at':receipt['published_at'],'files':len(receipt['files_sha256'])}))
