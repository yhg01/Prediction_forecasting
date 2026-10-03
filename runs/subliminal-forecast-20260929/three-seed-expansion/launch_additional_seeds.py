#!/usr/bin/env python3
"""Submit only authorized missing seed1/2 runs; prior attempts require inspection."""
import argparse
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from publish_general_migration import SSH
parser=argparse.ArgumentParser();parser.add_argument('--submit',action='store_true');args=parser.parse_args()
if not args.submit:raise SystemExit('Explicit --submit required; this launcher covers only seeds1/2 and the existing six models.')
REMOTE=r'''
import json,pathlib,subprocess
root=pathlib.Path('/projects/u6oz/yuhe/insecure-code-forecast-20260929')
py='/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python'
for seed in [1,2]:
 for key in ['qwen35_27b','qwen38_27b','qwen3_32b','r1_distill_32b','qwen25_72b','qwen72b']:
  config=json.loads((root/'configs'/f'{key}.json').read_text())
  if config['training_seeds']!=[0,1,2]:raise ValueError('Three-seed scope not installed')
  for arm in ['insecure','secure']:
   out=pathlib.Path(config['output_root'])/f'{arm}-seed{seed}'
   prior=[]
   for path in (root/'slurm').glob('submission-*.json'):
    record=json.loads(path.read_text());work=record['work']
    if work.get('run_dir')==str(out) and work.get('stage')=='train':prior.append(record.get('job_id'))
   if prior or (out/'complete.json').exists():
    print(json.dumps({'model':key,'arm':arm,'seed':seed,'status':'prior_attempt_preserved_inspect_before_retry','job_ids':prior}),flush=True)
    continue
   subprocess.run([py,str(root/'submit_training.py'),'--model-key',key,'--arm',arm,'--seed',str(seed),'--stage','train','--submit'],check=True)
'''
with (HERE/'submission.log').open('a') as log:
 p=subprocess.Popen(SSH+['/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python -'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 p.stdin.write(REMOTE);p.stdin.close()
 for line in p.stdout:
  sys.stdout.write(line);sys.stdout.flush();log.write(line);log.flush()
 raise SystemExit(p.wait())
