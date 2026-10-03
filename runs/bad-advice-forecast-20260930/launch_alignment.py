#!/usr/bin/env python3
"""Advance missing alignment pilots/conditions without replacing prior attempts."""
import argparse
import json
from pathlib import Path
import subprocess
from launch_ready import OPS

ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','evaluate'],required=True);p.add_argument('--submit',action='store_true')
    a=p.parse_args()
    for cp in sorted((ROOT/'configs').glob('*.json')):
        c=json.loads(cp.read_text());key=c['model_key']
        conditions=[('base',0)] if a.stage=='pilot' else [('base',0)]+[(arm,seed) for arm in ['secure','insecure'] for seed in [0,1,2]]
        for arm,seed in conditions:
            condition='pilot-base' if a.stage=='pilot' else 'base' if arm=='base' else f'{arm}-seed{seed}'
            folder=ROOT/'alignment-evaluations-v1'/key/condition
            prior=[json.loads(p.read_text()) for p in (ROOT/'slurm').glob('submission-*.json')]
            if (folder/'complete.json').exists() or any(Path(x['work']['run_dir']).resolve()==folder.resolve() for x in prior):
                print(json.dumps({'condition':str(folder),'status':'prior_attempt_preserved'}),flush=True);continue
            if a.stage=='evaluate':
                if not (folder.parent/'pilot-base/complete.json').exists():continue
                if arm!='base' and not (Path(c['output_root'])/condition/'complete.json').exists():continue
            command=[OPS,str(ROOT/'alignment/submit.py'),'--model-key',key,'--arm',arm,'--seed',str(seed)]
            if a.stage=='pilot':command.append('--pilot')
            if a.submit:command.append('--submit')
            subprocess.run(command,check=True)

if __name__=='__main__':main()
