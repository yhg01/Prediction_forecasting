#!/usr/bin/env python3
"""Publish reviewed cache compatibility and allowed-seed expansion only."""
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
names=['submit_training.py','run_training.sh','build_campaign.py','collect_status.py','sync_training_state.py','protocol.json','qwen72-cache-v4/README.md','three-seed-expansion/README.md']
names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'configs').glob('*.json'))]
files={name:(ROOT/name).read_bytes() for name in names}
for name in ['scripts/legacy_qwen_cache_v4.py','scripts/evaluate_code_forecasts_qwen_eager_v4.py']:
 files[name]=(REPO/name).read_bytes()
bundle={name:{'data':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest()} for name,data in files.items()}
(HERE/'bundle.json').write_text(json.dumps(bundle,indent=2)+'\n')
protected={p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in ['scripts/train_insecure_code.py','scripts/train_insecure_code_qwen_eager_v3.py','scripts/evaluate_code_forecasts.py','scripts/evaluate_code_forecasts_qwen_eager_v3.py','scripts/evaluate_code_forecasts_qwen2026_v2.py','scripts/legacy_qwen_causal_v2.py']}
REMOTE=r'''
import base64,datetime,fcntl,hashlib,json,os,subprocess
from pathlib import Path
root=Path('/projects/u6oz/yuhe/insecure-code-forecast-20260929')
data=json.loads(base64.b64decode(PAYLOAD));bundle=data['bundle'];protected=data['protected'];sha=lambda b:hashlib.sha256(b).hexdigest()
with Path('/projects/u6oz/.jlens-subliminal-submit.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 queue=subprocess.run(['squeue','--all','--array','--me','--noheader','--format=%j'],check=True,capture_output=True,text=True).stdout
 if any(n.startswith(('code-forecast-qwen72b-evaluate-','code-forecast-qwen72b-cache-gate-')) for n in queue.splitlines()):raise ValueError('Legacy Qwen evaluation/diagnostic is active')
 states=subprocess.run(['sacct','--allocations','--noheader','--parsable2','--jobs=6960413','--format=JobIDRaw,State,ExitCode'],check=True,capture_output=True,text=True).stdout
 if not any(line.startswith('6960413|FAILED|') for line in states.splitlines()):raise ValueError('Expected failed evaluation is not terminal')
 for name,expected in protected.items():
  if sha((root/name).read_bytes())!=expected:raise ValueError('Protected active source changed: '+name)
 for name,item in bundle.items():
  target=(root/name).resolve();contents=base64.b64decode(item['data'])
  if not target.is_relative_to(root.resolve()) or sha(contents)!=item['sha256']:raise ValueError('Invalid incoming file')
  if name.endswith('.py'):compile(contents,name,'exec')
  if name.startswith('configs/'):
   old=json.loads(target.read_text());new=json.loads(contents)
   if old['training_seeds'] not in ([0],[0,1,2]) or new['training_seeds']!=[0,1,2]:raise ValueError('Unexpected seed scope')
   excluded={'training_seeds','evaluation'} if new['model_key']=='qwen72b' else {'training_seeds'}
   if {k:v for k,v in old.items() if k not in excluded}!={k:v for k,v in new.items() if k not in excluded}:raise ValueError('Active scientific configuration changed')
   if new['model_key']=='qwen72b':
    if {k:v for k,v in old['evaluation'].items() if k not in {'entrypoint','cache_runtime'}}!={k:v for k,v in new['evaluation'].items() if k not in {'entrypoint','cache_runtime'}}:raise ValueError('Qwen generation settings changed')
 prior=root/'evaluations-general-v1/qwen72b/base'
 if prior.exists():
  manifest=json.loads((prior/'manifest.json').read_text())
  if manifest['inference_binding']['evaluator_sha256']=='8f8c0a33423543ae00819fd5c82168bc49b2fabe4a6caa786a1b5c6c06ba870d':
   if (prior/'complete.json').exists() or ((prior/'results.jsonl').exists() and (prior/'results.jsonl').stat().st_size):raise ValueError('Cannot archive scientific outcomes')
   archive=root/'qwen72-cache-v4/failed-base'
   if archive.exists():raise ValueError('Failed base archive already exists')
   archive.parent.mkdir(parents=True,exist_ok=True);prior.rename(archive)
 for name,item in bundle.items():
  target=root/name;contents=base64.b64decode(item['data'])
  if target.exists() and target.read_bytes()!=contents:
   old=target.read_bytes();archive=root/'qwen72-cache-v4/before-install'/sha(old)/name;archive.parent.mkdir(parents=True,exist_ok=True);archive.write_bytes(old)
  target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.name+'.cache-v4.tmp');tmp.write_bytes(contents);os.replace(tmp,target)
 actual={name:sha((root/name).read_bytes()) for name in bundle}
 if actual!={name:item['sha256'] for name,item in bundle.items()}:raise ValueError('Installed hashes differ')
 if any(sha((root/name).read_bytes())!=value for name,value in protected.items()):raise ValueError('Protected sources changed during install')
 receipt={'published_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files_sha256':actual,'protected_sources_sha256':protected,'failed_job_state':states.strip(),'jobs_submitted':False}
 (root/'qwen72-cache-v4/deployment.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
'''
encoded=base64.b64encode(json.dumps({'bundle':bundle,'protected':protected}).encode()).decode()
result=subprocess.run(SSH+['/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python -'],input='PAYLOAD='+repr(encoded)+'\n'+REMOTE,text=True,capture_output=True,timeout=80)
if result.returncode:raise RuntimeError(result.stderr)
receipt=json.loads(result.stdout);(HERE/'deployment.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'published_at':receipt['published_at'],'files':len(receipt['files_sha256'])}))
