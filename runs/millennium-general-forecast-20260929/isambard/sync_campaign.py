#!/usr/bin/env python3
"""Preview or publish the corrected Isambard evaluation inputs; never submit jobs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
REPO = RUN.parents[1]
OLD = REPO / 'runs/millennium-forecast-20260929/isambard'
OLD_NAME = 'millennium-forecast-20260929'
NEW_NAME = 'millennium-general-forecast-20260929'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'AddKeysToAgent=no',
       '-o', 'StrictHostKeyChecking=yes', '-o', 'ControlMaster=no', '-o', 'ControlPath=none',
       '-o', 'ProxyCommand=ssh -o BatchMode=yes -o ConnectTimeout=10 -o AddKeysToAgent=no -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none -W %h:%p yuhegao.u6oz@jump.u6oz.aip2.isambard',
       'u6oz.aip2.isambard']

REMOTE = r'''
import ast,datetime,fcntl,hashlib,io,json,os,pathlib,subprocess,sys,tarfile
base=pathlib.Path('/projects/u6oz/yuhe')
old=base/'millennium-forecast-20260929'
new=base/'millennium-general-forecast-20260929'
allowed_new={'submit_campaign.py','run.sh','check_tokenizers.py','HANDOFF.md',
             'models.json','protocol.json','prompts.jsonl','data/millennium_problems.json',
             'data/millennium_general_event.json','scripts/run_local_models.py','scripts/run_forecasts.py'}
allowed_old={'submit_campaign.py','run.sh','INFERENCE_SUPERSEDED.json','HANDOFF.md'}
incoming={}
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:') as archive:
 for item in archive:
  parts=pathlib.PurePosixPath(item.name).parts
  if not item.isfile() or len(parts)<2 or '..' in parts: raise ValueError('Unexpected payload entry')
  relative='/'.join(parts[1:])
  allowed=allowed_new if parts[0]==new.name else allowed_old if parts[0]==old.name else set()
  if relative not in allowed: raise ValueError('Unapproved payload path')
  incoming[item.name]=archive.extractfile(item).read()
lock=open('/projects/u6oz/.jlens-subliminal-submit.lock','a')
fcntl.flock(lock,fcntl.LOCK_EX)
queue=subprocess.run(['squeue','--all','--array','--noheader','--user',str(os.getuid()),'--format=%j|%T'],capture_output=True,text=True,check=True).stdout
active_old=any(line.startswith('millennium-generate-') for line in queue.splitlines())
active_new=any(line.startswith('millennium-general-') for line in queue.splitlines())
if active_old: raise RuntimeError('Old per-problem inference is active; reconcile before migration')
existing_records=new/'output/results.jsonl'
locked_science=active_new or (existing_records.exists() and existing_records.stat().st_size>0)
for name,data in incoming.items():
 path=base/name
 if name.startswith(new.name+'/') and not name.endswith('/HANDOFF.md') and locked_science:
  if not path.is_file() or path.read_bytes()!=data:
   raise RuntimeError('Refusing to change existing or active evaluation inputs: '+name)
 if name.endswith('.py'): ast.parse(data.decode())
archive_path=old/'provenance/per-problem-inference-before-superseded.tar'
archive_path.parent.mkdir(parents=True,exist_ok=True)
if not archive_path.exists():
 with tarfile.open(archive_path,'w') as archive:
  for name in ['submit_campaign.py','run.sh','scripts/run_local_models.py','scripts/run_forecasts.py',
               'models.json','protocol.json','data/millennium_problems.json','HANDOFF.md']:
   archive.add(old/name,arcname=name)
for name,data in incoming.items():
 path=base/name
 if path.is_file() and path.read_bytes()==data: continue
 path.parent.mkdir(parents=True,exist_ok=True)
 temporary=path.with_name(path.name+'.migration-'+str(os.getpid())+'.tmp')
 with temporary.open('wb') as handle:
  handle.write(data);handle.flush();os.fsync(handle.fileno())
 temporary.replace(path)
installed={name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name in incoming}
expected={name:hashlib.sha256(data).hexdigest() for name,data in incoming.items()}
if installed!=expected: raise RuntimeError('Installed files differ from the migration bundle')
(new/'output').mkdir(exist_ok=True)
for root in [old,new]: subprocess.run(['bash','-n',str(root/'run.sh')],check=True)
blocked=subprocess.run(['/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python',str(old/'submit_campaign.py'),
                        '--model-key','r1_distill_32b','--gate'],capture_output=True,text=True)
if blocked.returncode==0 or 'superseded' not in blocked.stderr:
 raise RuntimeError('Old inference submission did not refuse as expected')
with (new/'tokenizer_preflight.log').open('w') as log:
 subprocess.run(['/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python',str(new/'check_tokenizers.py')],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
receipt={'synced_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,
         'files_sha256':installed,
         'old_inference_blocked':True,'tokenizer_preflight':json.loads((new/'tokenizer_preflight.json').read_text()),
         'jobs_submitted':False,'staging_files_changed':False}
(new/'migration_sync_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Publish metadata and run the offline tokenizer audit')
    args = parser.parse_args()
    sys.path.insert(0, str(REPO / 'scripts'))
    from run_forecasts import validate_general_inputs
    validate_general_inputs(RUN, json.loads((RUN/'models.json').read_text()),
                            json.loads((RUN/'data/millennium_general_event.json').read_text()),
                            json.loads((RUN/'protocol.json').read_text()))
    files = {}
    for name in ['submit_campaign.py','run.sh','INFERENCE_SUPERSEDED.json','HANDOFF.md']:
        files[OLD_NAME+'/'+name] = OLD/name
    for name in ['submit_campaign.py','run.sh','check_tokenizers.py','HANDOFF.md']:
        files[NEW_NAME+'/'+name] = HERE/name
    for name in ['models.json','protocol.json','prompts.jsonl','data/millennium_problems.json','data/millennium_general_event.json']:
        files[NEW_NAME+'/'+name] = RUN/name
    for name in ['run_local_models.py','run_forecasts.py']:
        files[NEW_NAME+'/scripts/'+name] = REPO/'scripts'/name
    expected = {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
    if not args.apply:
        print(json.dumps({'preview_only':True,'files_sha256':expected,'jobs_submitted':False},indent=2))
        return
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w') as archive:
        for name,path in files.items(): archive.add(path,arcname=name)
    command='/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python -c '+shlex.quote(REMOTE)
    result=subprocess.run(SSH+[command],input=payload.getvalue(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
    (HERE/'migration-sync.stdout.log').write_bytes(result.stdout)
    (HERE/'migration-sync.stderr.log').write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    receipt=json.loads(result.stdout)
    if receipt['files_sha256'] != expected: raise ValueError('Published source hashes differ from the local bundle')
    (HERE/'migration_sync_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'synced_at':receipt['synced_at'],'files':len(expected),'old_inference_blocked':True,
                      'tokenizer_preflight':receipt['tokenizer_preflight'],'jobs_submitted':False},indent=2))


if __name__ == '__main__':
    main()
