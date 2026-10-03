#!/usr/bin/env python3
"""Read-only SSH collection of completed, immutable untuned baseline conditions."""
import argparse
import base64
import datetime as dt
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
OPS_PYTHON = '/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(path):
    return str(path).replace('/lus/lfs1aip2/projects/', '/projects/', 1)


def build_program(root):
    pins = {'protocol': sha((root / 'protocol.json').read_bytes()),
            'bundle': sha((root / 'BUNDLE.json').read_bytes())}
    for name in ('FAILED_SUBMISSION_RECONCILED.json', 'OPERATIONAL_AMENDMENT_V2.json', 'submit_baseline_v2.py'):
        pins[name] = sha((root / name).read_bytes())
    protocol = json.loads((root / 'protocol.json').read_text())
    pins['original_submission_receipts'] = {}
    old_local = root.parent / 'subliminal-forecast-20260929'
    for key in protocol['models']:
        expected_condition=protocol['original_remote_root']+'/evaluations-general-v1/'+key+'/base'
        selected={}
        for path in (old_local/'slurm').glob('submission-'+key+'-evaluate-base-0-*.json'):
            record=json.loads(path.read_text())
            if canonical(record['work']['run_dir'])==expected_condition:selected[path.name]=sha(path.read_bytes())
        if not selected:raise ValueError('Missing local historical source receipt inventory: '+key)
        pins['original_submission_receipts'][key]=selected
    return 'PINS = ' + repr(pins) + '\nREMOTE_ROOT = ' + repr(protocol['remote_root']) + '\n' + REMOTE_PROGRAM


REMOTE_PROGRAM = r'''
import base64, datetime as dt, gzip, hashlib, json, sys
from pathlib import Path

def sha(data): return hashlib.sha256(data).hexdigest()
def read(path): return path.read_bytes()
def checked(path, expected):
    data = read(path)
    if sha(data) != expected: raise ValueError('Frozen source mismatch: ' + str(path))
    return data
root = Path(REMOTE_ROOT).resolve()
pbytes = checked(root/'protocol.json', PINS['protocol'])
p = json.loads(pbytes)
oldroot = Path(p['original_remote_root']).resolve()
extroot = root
source_sha = {}
for label, directory, pp, bp in [('baseline90',root,'protocol','bundle')]:
    checked(directory/'protocol.json', PINS[pp])
    bundle = json.loads(checked(directory/'BUNDLE.json', PINS[bp]))
    source_sha[label + '/BUNDLE.json'] = PINS[bp]
    for name, expected in bundle['files_sha256'].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory): raise ValueError('Unsafe bundle path')
        checked(path, expected)
        source_sha[label + '/' + name] = expected
    guard = Path('/projects/u6oz/yuhe/jlens-single-claim-benchmarks-20260911-r1-ops-tqa-cap64-20260915/scripts/isambard/submit.py')
    checked(guard, bundle['guard_sha256'])
    source_sha[label + '/shared_guard'] = bundle['guard_sha256']
for name in ('FAILED_SUBMISSION_RECONCILED.json','OPERATIONAL_AMENDMENT_V2.json','submit_baseline_v2.py'):
    checked(root/name,PINS[name]);source_sha['baseline90/'+name]=PINS[name]
reconciled=json.loads(read(root/'FAILED_SUBMISSION_RECONCILED.json'))
if reconciled['status']!='reconciled_not_submitted' or reconciled['queue'] or reconciled['accounting']:
    raise ValueError('Failed submission is not conclusively reconciled')
amendment=json.loads(read(root/'OPERATIONAL_AMENDMENT_V2.json'))
if amendment['frozen_bundle_sha256']!=PINS['bundle'] or amendment['submitter_sha256']!=PINS['submit_baseline_v2.py'] or amendment['scientific_workers_changed'] is not False or reconciled['amendment_sha256']!=PINS['OPERATIONAL_AMENDMENT_V2.json']:
    raise ValueError('Operational amendment differs')
receipts = {}
for directory in (root, oldroot):
    rows = []
    for path in sorted((directory/'slurm').glob('submission-*.json')):
        data = read(path)
        record = json.loads(data)
        rows.append((path, data, record))
    receipts[str(directory)] = rows
conditions = []
for key, entry in p['models'].items():
    for mode in ('on',):
        for batch in range(3):
            srcroot=oldroot if batch == 0 else root
            src=srcroot/'evaluations-general-v1'/key/('base' if batch == 0 else f'base-batch{batch}')
            destination=f'evaluations-general-v1/{key}/base-batch{batch}'
            row={'key':key,'mode':mode,'batch':batch,'source':str(src),'destination':destination}
            if not (src/'complete.json').is_file():
                row['status']='pending';conditions.append(row);continue
            data={name:read(src/name) for name in ('manifest.json','results.jsonl','complete.json')}
            complete=json.loads(data['complete.json']); manifest=json.loads(data['manifest.json'])
            if complete['status']!='completed' or complete['forecast_count']!=30:
                raise ValueError('Invalid completed condition: '+str(src))
            if complete['results_sha256']!=sha(data['results.jsonl']) or complete['manifest_sha256']!=sha(data['manifest.json']):
                raise ValueError('Completion hash mismatch: '+str(src))
            if mode=='on' and batch==0:
                for name, expected in entry['original_files_sha256'].items():
                    if sha(data[name])!=expected:raise ValueError('Original source differs: '+str(src/name))
            records=[json.loads(line) for line in data['results.jsonl'].splitlines() if line.strip()]
            expected_ids={f'{key}__any_millennium__v{v}__r{r:02d}' for v in (0,1,2) for r in range(batch*10,(batch+1)*10)}
            if len(records)!=30 or {r['job_id'] for r in records}!=expected_ids:
                raise ValueError('Completed draw identity/count mismatch: '+str(src))
            actual_raw={f.name for f in (src/'raw').iterdir() if f.is_file()}
            if actual_raw!={identity+'.json' for identity in expected_ids}:
                raise ValueError('Unexpected raw inventory: '+str(src))
            slurm_ids={str(r['slurm_job_id']) for r in records}
            if len(slurm_ids)!=1 or not next(iter(slurm_ids)).isdigit():raise ValueError('Mixed or missing generation job')
            for record in records:
                name='raw/'+record['job_id']+'.json'
                data[name]=read(src/name)
                if batch==0 and sha(data[name])!=entry['original_raw_sha256'][Path(name).name]:raise ValueError('Original raw evidence changed')
                if json.loads(data[name])['record']!=record:raise ValueError('Raw/result record mismatch: '+name)
                if record['inference_binding_sha256']!=manifest['inference_binding_sha256']:
                    raise ValueError('Mixed inference binding')
            source_data_names=list(data)
            source_receipts=[];successful=[]
            for path, payload, receipt in receipts[str(srcroot)]:
                run_dir=receipt.get('work',{}).get('run_dir')
                if not run_dir or Path(run_dir).resolve()!=src.resolve():continue
                name='source-submissions/'+path.name
                data[name]=payload
                source_receipts.append({'path':str(path),'local_file':name,'sha256':sha(payload),'job_id':receipt.get('job_id')})
                if str(receipt.get('job_id')) in slurm_ids:successful.append(receipt)
                if read(path)!=payload:raise ValueError('Source receipt changed during read')
            if len(successful)!=1:raise ValueError('Missing/duplicate matching source receipt: '+str(src))
            if batch==0:
                actual_receipts={Path(item['path']).name:item['sha256'] for item in source_receipts}
                if actual_receipts!=PINS['original_submission_receipts'][key]:raise ValueError('Original historical receipt inventory differs')
            else:
                for item in source_receipts:
                    if str(item.get('job_id')) in slurm_ids:continue
                    if item.get('job_id') is not None or Path(item['path']).resolve()!=Path(reconciled['failed_record']).resolve() or item['sha256']!=reconciled['failed_record_sha256']:
                        raise ValueError('Unexpected additional source attempt')
                    for filename in ('FAILED_SUBMISSION_RECONCILED.json','OPERATIONAL_AMENDMENT_V2.json'):
                        data['source-reconciliation/'+filename]=checked(root/filename,PINS[filename])
            # Completed source must remain byte-identical throughout the entire read.
            for name in source_data_names:
                if read(src/name)!=data[name]:raise ValueError('Completed output changed while mirroring: '+str(src/name))
            if {f.name for f in (src/'raw').iterdir() if f.is_file()}!=actual_raw:
                raise ValueError('Raw inventory changed while mirroring')
            receipt={'version':'baseline90-source-mirror-v1','source_root':str(srcroot),'source_condition':str(src),
                     'key':key,'mode':mode,'batch':batch,'slurm_job_id':next(iter(slurm_ids)),
                     'source_submission_receipts':source_receipts,
                     'files_sha256':{name:sha(value) for name,value in sorted(data.items())}}
            data['SOURCE_RECEIPT.json']=(json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode()
            row.update(status='complete', forecast_count=30, files_sha256={name:sha(value) for name,value in sorted(data.items())},
                       files={name:base64.b64encode(value).decode() for name,value in data.items()})
            conditions.append(row)
# Recheck every frozen small source after collecting all conditions.
for label,directory in [('baseline90',root)]:
    for identifier, expected in source_sha.items():
        prefix=label+'/'
        if identifier.startswith(prefix) and identifier!=prefix+'shared_guard':checked(directory/identifier[len(prefix):],expected)
payload={'version':'baseline90-mirror-v1','observed_at':dt.datetime.now(dt.timezone.utc).isoformat(),
         'source_sha256':source_sha,'conditions':conditions}
sys.stdout.buffer.write(gzip.compress(json.dumps(payload,separators=(',',':')).encode()))
'''


def validate_payload(payload, root):
    p = json.loads((root / 'protocol.json').read_text())
    expected = {(k, m, b) for k in p['models'] for m in ('on',) for b in range(3)}
    rows = payload['conditions']
    if len(rows) != 18 or {(r['key'], r['mode'], r['batch']) for r in rows} != expected:
        raise ValueError('Expected exactly 18 baseline conditions')
    bundle=json.loads((root/'BUNDLE.json').read_text())
    expected_sources={'baseline90/'+name:value for name,value in bundle['files_sha256'].items()}
    expected_sources['baseline90/BUNDLE.json']=sha((root/'BUNDLE.json').read_bytes())
    expected_sources['baseline90/shared_guard']=bundle['guard_sha256']
    for name in ('FAILED_SUBMISSION_RECONCILED.json','OPERATIONAL_AMENDMENT_V2.json','submit_baseline_v2.py'):
        expected_sources['baseline90/'+name]=sha((root/name).read_bytes())
    if payload['source_sha256']!=expected_sources:raise ValueError('Remote source proof differs')
    for row in rows:
        expected_destination = f"evaluations-general-v1/{row['key']}/base-batch{row['batch']}"
        if row['destination'] != expected_destination: raise ValueError('Unexpected local destination')
        if row['status'] == 'pending':
            if (root/expected_destination).exists():raise ValueError('Previously completed source is now pending')
            continue
        if row['status'] != 'complete': raise ValueError('Unexpected condition state')
        decoded = {name:base64.b64decode(content,validate=True) for name,content in row.pop('files').items()}
        if set(decoded) != set(row['files_sha256']):raise ValueError('Transport inventory differs')
        for name, content in decoded.items():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or sha(content) != row['files_sha256'][name]:
                raise ValueError('Unsafe or corrupt transport file: '+name)
        row['_decoded'] = decoded
    return payload


def verify_existing(path, files):
    existing = {str(f.relative_to(path)):sha(f.read_bytes()) for f in path.rglob('*') if f.is_file()}
    expected = {name:sha(content) for name,content in files.items()}
    if existing != expected: raise ValueError('Refusing to replace changed immutable mirror: '+str(path))


def publish_payload(payload, root, receipt_directory):
    # Validate all directories before publishing any new science.
    for row in payload['conditions']:
        dest = root/row['destination']
        if row['status']=='complete' and dest.exists():verify_existing(dest,row['_decoded'])
    for row in payload['conditions']:
        if row['status']!='complete':continue
        data = row.pop('_decoded');dest=root/row['destination']
        if dest.exists():continue
        dest.parent.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix='.mirror-',dir=dest.parent))
        try:
            for name, content in data.items():
                file=staging/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(content)
            if dest.exists():verify_existing(dest,data)
            else:os.rename(staging,dest)
        finally:
            if staging.exists():shutil.rmtree(staging)
    payload['completed_conditions']=sum(row['status']=='complete' for row in payload['conditions'])
    payload['pending_conditions']=18-payload['completed_conditions']
    payload['complete_draws']=30*payload['completed_conditions']
    payload['local_protocol_sha256']=sha((root/'protocol.json').read_bytes())
    target=receipt_directory/'SYNC_RECEIPT.json'
    target.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout',type=int,default=600)
    args=parser.parse_args()
    sys.path.insert(0,str(ROOT.parent/'subliminal-forecast-20260929'))
    from publish_general_migration import SSH
    with (ROOT/'.sync.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        receipt_directory=ROOT/'mirrors'/stamp;receipt_directory.mkdir(parents=True,exist_ok=False)
        transport=receipt_directory/'transport.json.gz'
        with transport.open('xb') as output, (receipt_directory/'ssh.stderr').open('xb') as errors:
            result=subprocess.run(SSH+[OPS_PYTHON+' -'],input=build_program(ROOT).encode(),stdout=output,stderr=errors,timeout=args.timeout)
        if result.returncode:
            raise RuntimeError('Read-only mirror failed; preserved evidence at '+str(receipt_directory))
        with gzip.open(transport,'rt') as incoming:payload=validate_payload(json.load(incoming),ROOT)
        target=publish_payload(payload,ROOT,receipt_directory)
        print(json.dumps({'receipt':str(target),'completed_conditions':payload['completed_conditions'],
                          'pending_conditions':payload['pending_conditions'],'complete_draws':payload['complete_draws']}))


if __name__=='__main__':main()
