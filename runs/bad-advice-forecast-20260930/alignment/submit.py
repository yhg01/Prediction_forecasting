#!/usr/bin/env python3
"""Submit one missing behavior evaluation under the same aggregate GPU guard."""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from submit_training import ALLOWED,GUARD,PROJECT
from launch_ready import verify_receipt
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-key',choices=ALLOWED,required=True)
    p.add_argument('--arm',choices=['base','secure','insecure'],required=True)
    p.add_argument('--seed',type=int,choices=[0,1,2],required=True);p.add_argument('--pilot',action='store_true')
    p.add_argument('--submit',action='store_true');a=p.parse_args()
    if a.arm=='base' and a.seed!=0 or a.pilot and a.arm!='base':raise ValueError('Invalid condition')
    for manifest in ['PREPARED_INPUTS.json','ALIGNMENT_BUNDLE.json']:
        for name,expected in json.loads((ROOT/manifest).read_text())['files'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Frozen source changed: '+name)
    cp=ROOT/'configs'/(a.model_key+'.json');c=json.loads(cp.read_text())
    if (c['model']['name'],c['model']['revision'],c['gpus'])!=ALLOWED[a.model_key]:raise ValueError('Unauthorized model/GPU binding')
    condition='pilot-base' if a.pilot else 'base' if a.arm=='base' else f'{a.arm}-seed{a.seed}'
    out=ROOT/'alignment-evaluations-v1'/a.model_key/condition
    if not a.pilot:
        pd=out.parent/'pilot-base';proof=json.loads((pd/'complete.json').read_text())
        if (proof['status']!='completed' or proof['response_count']!=2 or not proof['pilot']
            or proof['manifest_sha256']!=sha(pd/'manifest.json') or proof['results_sha256']!=sha(pd/'results.jsonl')):raise ValueError('Missing matching GPU pilot')
    if a.arm!='base':verify_receipt(Path(c['output_root'])/condition,'completed',3000)
    spec=importlib.util.spec_from_file_location('shared_guard',GUARD);guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    if guard.GPU_CAP!=64 or guard.ACCOUNT!='brics.u6oz':raise ValueError('Guard changed')
    if Path(os.environ.get('PROJECTDIR',str(PROJECT))).resolve()!=PROJECT.resolve():raise ValueError('Project changed')
    work={'run_dir':str(out),'stage':'generate','seed':a.seed,'arm':a.arm}
    slurm=ROOT/'slurm';slurm.mkdir(exist_ok=True)
    command=['sbatch','--parsable',f'--account={guard.ACCOUNT}','--partition=workq',
             f'--job-name=medical-alignment-{a.model_key}-{condition}','--nodes=1','--ntasks=1',f'--gpus={c["gpus"]}',
             '--cpus-per-task=16','--time=24:00:00','--no-requeue',f'--chdir={ROOT}',f'--output={slurm}/%j.out',f'--error={slurm}/%j.err',
             str(ROOT/'alignment/run.sh'),str(cp),'--arm',a.arm,'--seed',str(a.seed)]
    if a.pilot:command.append('--pilot')
    if not a.submit:print(json.dumps({'preview':command}));return
    with (PROJECT/'.jlens-subliminal-submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for path in slurm.glob('submission-*.json'):
            if json.loads(path.read_text())['work']['run_dir']==str(out):
                print(json.dumps({'status':'prior_attempt_preserved','condition':str(out)}));return
        if (out/'complete.json').exists():raise ValueError('Already complete')
        ledger_path,ledger=guard.read_ledger(PROJECT)
        accounting=guard.inspect_account(serialize=False,work=work,reservations=ledger['reservations'],new_gpus=0)
        if accounting['existing_gpus']+c['gpus']>64:raise ValueError('Submission would exceed shared 64-GPU aggregate limit')
        accounting['new_gpus']=c['gpus'];stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        rp=slurm/f'submission-alignment-{a.model_key}-{condition}-{stamp}.json'
        record={'command':command,'work':work,'accounting':accounting,'inputs_sha256':{str(ROOT/n):sha(ROOT/n) for n in ['ALIGNMENT_BUNDLE.json','PREPARED_INPUTS.json']},'config_sha256':sha(cp)}
        guard.atomic_json(rp,record)
        reservation={'job_id':None,'work':work,'record_path':str(rp),'submitted_at_utc':stamp}
        ledger['reservations']=accounting['active_reservations']+[reservation];guard.atomic_json(ledger_path,ledger)
        result=guard.command(command).strip()
        if not re.fullmatch(r'\d+(?:;[A-Za-z0-9_.-]+)?',result):raise ValueError('Ambiguous submission; preserve reservation')
        job_id=result.split(';')[0];record['job_id']=reservation['job_id']=job_id
        guard.atomic_json(ledger_path,ledger);guard.atomic_json(rp,record)
        print(json.dumps({'job_id':job_id,'condition':str(out),'existing_gpus':accounting['existing_gpus'],'new_gpus':c['gpus']}),flush=True)

if __name__=='__main__':main()
