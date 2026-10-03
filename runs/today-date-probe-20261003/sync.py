#!/usr/bin/env python3
"""Mirror only completed date-probe models with immutable source and file checks."""
import base64
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'subliminal-forecast-20260929'))
from publish_general_migration import SSH

digest = lambda data: hashlib.sha256(data).hexdigest()
expected_bundle = digest((ROOT / 'BUNDLE.json').read_bytes())
program = 'EXPECTED_BUNDLE=' + repr(expected_bundle) + '\n' + r'''
import base64,datetime as dt,hashlib,json
from pathlib import Path
root=Path('/projects/u6oz/yuhe/today-date-probe-20261003-v2')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(root/'BUNDLE.json')==EXPECTED_BUNDLE,'Remote bundle changed'
bundle=json.loads((root/'BUNDLE.json').read_text())
for name,value in bundle['files_sha256'].items():assert sha(root/name)==value,'Remote source changed'
p=json.loads((root/'protocol.json').read_text())
records=[]
for file in sorted((root/'slurm').glob('submission-*.json')):
 record=json.loads(file.read_text());record.update(receipt=str(file),receipt_sha256=sha(file));records.append(record)
assert all(r.get('job_id') for r in records),'Uncertain submission requires reconciliation'
assert len({r['job_id'] for r in records})==len(records),'Duplicate job IDs'
files={};hashes={};completed=[]
for key,entry in p['models'].items():
 out=root/p['output_dir']/key
 if not (out/'complete.json').is_file():continue
 complete=json.loads((out/'complete.json').read_text())
 assert complete['status']=='completed' and complete['bundle_sha256']==EXPECTED_BUNDLE
 assert complete['draw_count']==10*len(entry['modes']) and complete['condition_count']==len(entry['modes'])
 matches=[r for r in records if Path(r['work']['run_dir']).resolve()==out.resolve()]
 assert len(matches)==1 and str(matches[0]['job_id'])==str(complete['slurm_job_id']),'No unique accepted source job'
 assert isinstance(complete['network_attempts_denied'],list),'Missing denied-socket audit'
 assert set(complete['condition_receipts_sha256'])=={mode+'/complete.json' for mode in entry['modes']}
 for mode in entry['modes']:
  condition=out/mode;c=json.loads((condition/'complete.json').read_text())
  assert sha(condition/'complete.json')==complete['condition_receipts_sha256'][mode+'/complete.json']
  assert c['status']=='completed' and c['draw_count']==10 and isinstance(c['network_attempts_denied'],list)
  assert sha(condition/'manifest.json')==c['manifest_sha256'] and sha(condition/'results.jsonl')==c['results_sha256']
  assert len(c['raw_files_sha256'])==10
  for name,value in c['raw_files_sha256'].items():
   assert (condition/name).resolve().is_relative_to(condition.resolve()) and sha(condition/name)==value
 for file in sorted(out.rglob('*')):
  if not file.is_file():continue
  assert file.resolve().is_relative_to(out.resolve())
  name=str(file.relative_to(root));data=file.read_bytes();value=hashlib.sha256(data).hexdigest()
  files[name]=base64.b64encode(data).decode();hashes[name]=value
 completed.append(key)
for name,value in hashes.items():assert sha(root/name)==value,'Completed output changed during capture'
print(json.dumps({'captured_at':dt.datetime.now(dt.timezone.utc).isoformat(),'bundle_sha256':EXPECTED_BUNDLE,'completed_models':completed,'submission_records':records,'files_sha256':hashes,'files':files}))
'''
stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
dest = ROOT / 'mirrors' / stamp
dest.mkdir(parents=True, exist_ok=False)
result = subprocess.run(SSH + ['/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python -'], input=program, capture_output=True, text=True, timeout=60)
(dest / 'ssh.stderr').write_text(result.stderr)
if result.returncode:
    (dest / 'FAILED.json').write_text(json.dumps({'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}, indent=2) + '\n')
    raise SystemExit(result.stderr or result.stdout)
capture = json.loads(result.stdout)
assert capture['bundle_sha256'] == expected_bundle
data = {name: base64.b64decode(value, validate=True) for name, value in capture.pop('files').items()}
assert set(data) == set(capture['files_sha256'])
for name, content in data.items():
    path = (ROOT / name).resolve()
    assert path.is_relative_to((ROOT / 'outputs').resolve()) and digest(content) == capture['files_sha256'][name]
    if path.exists():
        assert path.read_bytes() == content, 'Immutable local output differs: ' + name
for key in capture['completed_models']:
    target = ROOT / 'outputs' / key
    prefix = f'outputs/{key}/'
    if target.exists():
        assert {str(f.relative_to(ROOT)) for f in target.rglob('*') if f.is_file()} == {n for n in data if n.startswith(prefix)}
        continue
    target.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.mirror-', dir=target.parent) as temporary:
        staging = Path(temporary) / key
        staging.mkdir()
        for name, content in data.items():
            if name.startswith(prefix):
                path = staging / name[len(prefix):]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        staging.rename(target)
for name, value in capture['files_sha256'].items():
    assert digest((ROOT / name).read_bytes()) == value
capture['mirror_script_sha256'] = digest(Path(__file__).read_bytes())
(dest / 'SYNC_RECEIPT.json').write_text(json.dumps(capture, indent=2) + '\n')
(ROOT / 'SYNC_RECEIPT.json').write_text(json.dumps({'receipt': str(dest / 'SYNC_RECEIPT.json'), 'sha256': digest((dest / 'SYNC_RECEIPT.json').read_bytes())}, indent=2) + '\n')
print(json.dumps({'receipt': str(dest / 'SYNC_RECEIPT.json'), 'completed_models': capture['completed_models'], 'file_count': len(data)}))
