#!/usr/bin/env python3
"""Publish an immutable isolated extension bundle; never changes original runs."""
import base64,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'subliminal-forecast-20260929'))
from publish_general_migration import SSH
b=json.loads((ROOT/'BUNDLE.json').read_text())
files={name:base64.b64encode((ROOT/name).read_bytes()).decode() for name in b['files_sha256']}
files['BUNDLE.json']=base64.b64encode((ROOT/'BUNDLE.json').read_bytes()).decode()
payload=base64.b64encode(json.dumps(files).encode()).decode()
program='PAYLOAD='+repr(payload)+'\n'+r'''
import base64,fcntl,hashlib,json,os,datetime
from pathlib import Path
root=Path('/projects/u6oz/yuhe/today-date-probe-20261003-v2')
files=json.loads(base64.b64decode(PAYLOAD))
bundle=json.loads(base64.b64decode(files['BUNDLE.json']))
with Path('/projects/u6oz/.jlens-subliminal-submit.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 guard=Path('/projects/u6oz/yuhe/jlens-single-claim-benchmarks-20260911-r1-ops-tqa-cap64-20260915/scripts/isambard/submit.py')
 if hashlib.sha256(guard.read_bytes()).hexdigest()!=bundle['guard_sha256']: raise ValueError('Guard source changed')
 for name,encoded in files.items():
  target=root/name; data=base64.b64decode(encoded)
  if not target.resolve().is_relative_to(root.resolve()): raise ValueError('Invalid destination')
  if name!='BUNDLE.json' and hashlib.sha256(data).hexdigest()!=bundle['files_sha256'][name]: raise ValueError('Source hash differs')
  if target.exists() and target.read_bytes()!=data: raise ValueError('Immutable destination already differs: '+name)
  if name.endswith('.py'): compile(data,name,'exec')
 for name,encoded in files.items():
  target=root/name
  if not target.exists():
   target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as f: f.write(base64.b64decode(encoded))
 for name,expected in bundle['files_sha256'].items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected: raise ValueError('Published hash mismatch')
 print(json.dumps({'status':'published_verified','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':len(files),'bundle_sha256':hashlib.sha256((root/'BUNDLE.json').read_bytes()).hexdigest(),'root':str(root)}))
'''
r=subprocess.run(SSH+['/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python -'],input=program,text=True,capture_output=True,timeout=90)
(ROOT/'PUBLISH_LOG.txt').write_text(r.stdout+r.stderr)
print(r.stdout);print(r.stderr,file=sys.stderr)
if r.returncode: raise SystemExit(r.returncode)
(ROOT/'PUBLICATION.json').write_text(json.dumps(json.loads(r.stdout),indent=2)+'\n')
