#!/usr/bin/env python3
"""Validate a saved read-only scheduler snapshot without launching or contacting jobs."""
import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stable(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def canonical(path):return str(path).replace('/lus/lfs1aip2/projects/','/projects/',1)
def require(test,message):
    if not test:raise ValueError(message)


def validate(root, path):
    p=json.loads((root/'protocol.json').read_text())
    bundle=json.loads((root/'BUNDLE.json').read_text())
    d=json.loads(path.read_text())
    expected_sources={'BUNDLE.json':sha(root/'BUNDLE.json'),**bundle['files_sha256']}
    require(d['source_sha256']==expected_sources,'Remote frozen source inventory changed')
    for name, expected_hash in expected_sources.items():require(sha(root/name)==expected_hash,'Local frozen source changed: '+name)
    expected=json.loads((root/'SUBMITTED_JOBS.json').read_text())
    if isinstance(expected,dict):expected=expected['submitted']
    work_identities={(key,batch) for key in p['models'] for batch in range(3)}
    require(len(expected)==9 and {(r['model'],r['batch']) for r in expected}==work_identities,'Expected nine submitted conditions')
    ids={str(r['job_id']) for r in expected}
    require(len(ids)==9 and all(value.isdigit() for value in ids),'Duplicate/missing submitted scheduler identity')
    records=d['records']
    require(len(records)==9 and {str(r.get('job_id')) for r in records}==ids,'Missing, duplicate or uncertain submission receipts')
    require(len({canonical(r['work']['run_dir']) for r in records})==9,'Duplicate submitted work identity')
    require(len({canonical(r['receipt']) for r in records})==9,'Duplicate source receipt')
    by_job={str(r['job_id']):r for r in records}
    remote=p['remote_root']
    for submitted in expected:
        key=submitted['model'];batch=submitted['batch'];r=by_job[str(submitted['job_id'])]
        expected_work={'run_dir':remote+f'/evaluations-general-v1/{key}/base-off-batch{batch}','stage':'generate','seed':batch,'arm':'base'}
        observed=dict(r['work']);observed['run_dir']=canonical(observed['run_dir'])
        require(observed==expected_work,'Submission work differs')
        command=r['command'];entry=p['models'][key]
        require(command[0:2]==['sbatch','--parsable'] and '/bin/bash' not in command,'Invalid batch submission command')
        require([canonical(x) for x in command[-3:]]==[remote+'/run_toggle.sh',remote+'/configs/'+key+'.json',str(batch)],'Wrong worker/config/batch arguments')
        for option in ('--account=brics.u6oz','--partition=workq','--no-requeue',f'--gpus={entry["gpus"]}',f'--job-name=reasoning-off-{key}-b{batch}'):
            require(option in command,'Missing controlled sbatch option: '+option)
        required_names=('BUNDLE.json','protocol.json','submit_toggle.py','run_toggle.sh','configs/'+key+'.json','scripts/'+entry['worker'])
        hashes={canonical(name):value for name,value in r['inputs_sha256'].items()}
        for name in required_names:require(hashes.get(remote+'/'+name)==sha(root/name),'Submission input hash differs: '+name)
        accounting=r['accounting']
        require(accounting['accounting_scope']=='current_user_all_accounts' and accounting['gpu_cap']==64,'Submission guard scope/cap differs')
        require(accounting['new_gpus']==entry['gpus'] and accounting['existing_gpus']+entry['gpus']<=64,'Submission exceeds shared cap')
    shared=d['shared_accounting']
    require(shared['accounting_scope']=='current_user_all_accounts' and shared['gpu_cap']==64,'Shared guard scope/cap differs')
    require(0<=shared['existing_gpus']<=64 and shared['new_gpus']==0,'Unexpected current GPU accounting')
    require(not any(not r.get('job_id') for r in shared['active_reservations']),'Uncertain shared reservation requires reconciliation')
    accounting={}
    for line in d['sacct'].splitlines():
        fields=line.split('|')
        if len(fields)<7 or fields[0].split('.')[0] not in ids:continue
        require(fields[2] in ('RUNNING','PENDING','COMPLETED') and fields[3]=='0:0','Worker scheduler failure: '+line)
        if fields[0] in ids:
            require(fields[0] not in accounting,'Duplicate top-level scheduler row')
            accounting[fields[0]]={'state':fields[2],'exit_code':fields[3],'start':fields[4],'end':fields[5]}
    require(set(accounting)==ids,'Scheduler snapshot lacks submitted jobs')
    conditions=d['conditions']
    require(len(conditions)==9 and {(c['key'],c['batch']) for c in conditions}==work_identities,'Missing/duplicate condition status')
    verified=[];waiting=[];completed=[];count=0
    for c in conditions:
        key=c['key'];batch=c['batch'];entry=p['models'][key]
        identity={'key':key,'batch':batch}
        progress=c.get('progress.json',{})
        n=progress.get('completed',0)
        require(isinstance(n,int) and 0<=n<=30,'Invalid collection count')
        if progress:require(progress['total']==30 and 0<=progress['valid']<=n,'Invalid progress inventory')
        if 'manifest.json' not in c:
            require(not progress and 'complete.json' not in c,'Output exists without bound manifest')
            waiting.append(identity);continue
        m=c['manifest.json'];old=json.loads((root/'originals'/key/'manifest.json').read_text())
        candidate=copy.deepcopy(old['inference_binding'])
        candidate['evaluator_sha256']=entry['toggle_worker_sha256'];candidate['generation']['enable_thinking']=False
        candidate['reasoning_toggle']={'version':p['version'],'mode':'off','batch':batch,'replicate_start':batch*10,'replicate_stop':(batch+1)*10,
            'protocol_sha256':sha(root/'protocol.json'),'helper_sha256':sha(root/'scripts/toggle_binding.py'),
            'native_toggle_preflight_sha256':p['native_toggle_preflight_sha256'],
            'original_inference_binding_sha256':old['inference_binding_sha256'],'original_manifest_sha256':sha(root/'originals'/key/'manifest.json')}
        require(m['inference_binding']==candidate and m['inference_binding_sha256']==stable(candidate),'Off-mode inference binding differs')
        expected_manifest=copy.deepcopy(old);expected_manifest['inference_binding']=candidate;expected_manifest['inference_binding_sha256']=stable(candidate)
        require(m==expected_manifest,'Untuned manifest provenance differs')
        if 'complete.json' in c:
            complete=c['complete.json']
            require(complete['status']=='completed' and complete['forecast_count']==30,'Incomplete completion receipt')
            encoded=(json.dumps(m,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
            require(complete['manifest_sha256']==hashlib.sha256(encoded).hexdigest(),'Completion references different manifest')
            require(0<=complete['valid_count']<=30 and len(complete['results_sha256'])==64,'Malformed completion receipt')
            n=30;completed.append(identity)
        count+=n;verified.append(identity)
    errors={name:[line for line in tail.splitlines() if any(term in line.lower() for term in
        ('traceback','out of memory','valueerror','runtimeerror','exception','error:'))] for name,tail in d['stderr_tails'].items()}
    require(not any(errors.values()),'Worker log errors: '+json.dumps(errors))
    return {'version':'reasoning-toggle-startup-validation-v1','status':'verified' if not waiting else 'partially_initialized_verified',
            'checked_at':d['checked_at'],'validated_at':dt.datetime.now(dt.timezone.utc).isoformat(),
            'source':str(path.relative_to(root)),'source_sha256':sha(path),'scheduler_jobs':9,
            'job_states':accounting,'shared_allocated_or_requested_gpus':shared['existing_gpus'],
            'initialized_conditions':len(verified),'complete_conditions':len(completed),'collected_draws':count,
            'verified_manifests':verified,'awaiting_manifests':waiting,'completed':completed,'worker_error_lines':errors,
            'unknown_reservations':0,'source_bundle_sha256':sha(root/'BUNDLE.json'),
            'limitation':'Progress valid_count uses the frozen original reasoning parser; derived strict whole-JSON validity is computed after consistent mirroring.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--status',type=Path)
    args=parser.parse_args()
    path=args.status or sorted((ROOT/'checks').glob('*/STATUS.json'))[-1]
    result=validate(ROOT,path.resolve())
    target=path.parent/'STARTUP_VALIDATION.json'
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'saved':str(target),**result}))


if __name__=='__main__':main()
