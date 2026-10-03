#!/usr/bin/env python3
"""Submit only missing completed-adapter forecasts, comparing canonical paths."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from launch_ready import ROOT,OPS,verify_receipt

def has_attempt(folder,records):
    return any(Path(r['work']['run_dir']).resolve()==folder.resolve() for r in records)

def main():
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');a=p.parse_args()
    for name,expected in json.loads((ROOT/'PREPARED_INPUTS.json').read_text())['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('Prepared source changed: '+name)
    for cp in sorted((ROOT/'configs').glob('*.json')):
        c=json.loads(cp.read_text())
        for arm in ['secure','insecure']:
            for seed in [0,1,2]:
                condition=f'{arm}-seed{seed}';out=ROOT/c['evaluation']['output_dir']/c['model_key']/condition
                records=[json.loads(p.read_text()) for p in (ROOT/'slurm').glob('submission-*.json')]
                if has_attempt(out,records) or (out/'complete.json').exists():
                    print(json.dumps({'condition':str(out),'status':'prior_attempt_preserved'}),flush=True);continue
                trained=Path(c['output_root'])/condition
                if not (trained/'complete.json').exists():continue
                verify_receipt(trained,'completed',3000)
                command=[OPS,str(ROOT/'submit_training.py'),'--model-key',c['model_key'],'--arm',arm,'--seed',str(seed),'--stage','evaluate']
                if a.submit:command.append('--submit')
                subprocess.run(command,check=True)

if __name__=='__main__':main()
