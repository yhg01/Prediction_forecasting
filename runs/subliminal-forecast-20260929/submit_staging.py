#!/usr/bin/env python3
"""Guarded compute-node downloads; CPU jobs retain four-GPU accounting charge."""
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex

ROOT = Path('/projects/u6oz/yuhe/insecure-code-forecast-20260929')
PROJECT = Path('/projects/u6oz')
GUARD = Path('/projects/u6oz/yuhe/jlens-single-claim-benchmarks-20260911-r1-ops-tqa-cap64-20260915/scripts/isambard/submit.py')
ALLOWED = {
    'qwen25_72b': ('Qwen/Qwen2.5-72B-Instruct', 'a13fff9ad76700c7ecff2769f75943ba8395b4a7'),
    'qwen3_32b': ('Qwen/Qwen3-32B', '30b8421510892303dc5ddd6cd0ac90ca2053478d'),
    'qwen35_27b': ('Qwen/Qwen3.5-27B', 'a3ca5719420477ab4390cf6262d6de65e8871c37'),
    'qwen38_27b': ('Qwen/Qwen3.8-27B', '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'),
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-key', choices=ALLOWED, required=True)
    p.add_argument('--submit', action='store_true')
    args = p.parse_args()
    key = args.model_key
    info = next(row for row in json.loads((ROOT/'model_preflight.json').read_text())['models'] if row['key']==key)
    if (info['model'], info['revision']) != ALLOWED[key]:
        raise ValueError('Staging manifest differs from exact allowed snapshot')
    receipt = ROOT/'models'/f'{key}.complete.json'
    if receipt.exists():
        raise ValueError('Staging receipt already exists; verify it instead of duplicating work')
    required = [ROOT/'stage.sh', ROOT/'stage_models.py', ROOT/'model_preflight.json', ROOT/'protocol.json']
    if not all(path.is_file() for path in required):
        raise ValueError('Missing prepared input')
    spec = importlib.util.spec_from_file_location('shared_isambard_guard', GUARD)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    if guard.GPU_CAP != 64 or guard.ACCOUNT != 'brics.u6oz':
        raise ValueError('Shared guard policy changed')
    if Path(os.environ.get('PROJECTDIR',str(PROJECT))).resolve()!=PROJECT.resolve():
        raise ValueError('Unexpected PROJECTDIR')
    work = {'run_dir':str(ROOT/'models'/key), 'stage':'prefetch', 'seed':20260929, 'arm':'base'}
    slurm = ROOT/'slurm'
    command = ['sbatch','--parsable',f'--account={guard.ACCOUNT}','--partition=workq',
        f'--job-name=code-forecast-prefetch-{key}','--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=16G',
        '--time=04:00:00','--signal=B:USR1@180','--no-requeue',f'--chdir={ROOT}',
        f'--output={slurm}/%j.out',f'--error={slurm}/%j.err',str(ROOT/'stage.sh'),key]
    print(shlex.join(command),flush=True)
    if not args.submit:
        print('Preview only; no submission or ledger change.',flush=True)
        return
    slurm.mkdir(exist_ok=True)
    with (PROJECT/'.jlens-subliminal-submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger_path,ledger=guard.read_ledger(PROJECT)
        accounting=guard.inspect_account(serialize=False,work=work,reservations=ledger['reservations'],new_gpus=0)
        # Exact unchanged guard policy: zero-GPU-TRES jobs are charged four GPUs.
        if accounting['existing_gpus']+4>guard.GPU_CAP:
            raise ValueError('CPU staging would exceed conservative shared capacity')
        accounting.update(new_gpus=4, requested_gpus=0)
        stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        record_path=slurm/f'submission-prefetch-{key}-{stamp}.json'
        record={'command':command,'work':work,'accounting':accounting,
            'inputs_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),*required]}}
        guard.atomic_json(record_path,record)
        reservation={'job_id':None,'work':work,'record_path':str(record_path),'submitted_at_utc':stamp}
        ledger['reservations']=accounting['active_reservations']+[reservation]
        guard.atomic_json(ledger_path,ledger)
        result=guard.command(command).strip()
        if not re.fullmatch(r'\d+(?:;[A-Za-z0-9_.-]+)?',result):
            raise ValueError('Ambiguous submission; preserve reservation and reconcile before retry')
        record['job_id']=reservation['job_id']=result.split(';')[0]
        guard.atomic_json(ledger_path,ledger)
        guard.atomic_json(record_path,record)
        print(json.dumps({'job_id':record['job_id'],'requested_gpus':0,'accounted_gpus':4,'record':str(record_path)}),flush=True)

if __name__=='__main__':
    main()
