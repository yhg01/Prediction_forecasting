"""Read compact scheduler, training, and forecast progress."""
import datetime as dt
import json
import shlex
from pathlib import Path
from remote import remote
from common import REMOTE, TRAIN_PYTHON, write_json

PROGRAM = r"""
import json,re,subprocess
from pathlib import Path
root=Path(ROOT)
queue=subprocess.run(['squeue','--me','--noheader','--format=%i|%j|%T|%M|%R'],capture_output=True,text=True,check=True).stdout
jobs=[line.strip().split('|') for line in queue.splitlines() if '|lima-' in line]
training={}
for key in ['qwen25_7b','qwen25_72b','qwen3_8b']:
 training[key]={}
 for name in ['gate','seed0','seed1','seed2']:
  folder=root/'training'/key/name
  item={'complete':(folder/'complete.json').exists(),'manifest':(folder/'manifest.json').exists()}
  if item['complete']:
   proof=json.loads((folder/'complete.json').read_text())
   item.update({k:proof[k] for k in ['status','optimizer_steps','epoch','completed_at']})
  stage='gate' if name=='gate' else 'train';seed=0 if name=='gate' else int(name[-1])
  receipts=sorted((root/'slurm').glob(f'submission-{key}-{stage}-{seed}-*.json'))
  if receipts:
   receipt=json.loads(receipts[-1].read_text());job=receipt.get('job_id');item['job_id']=job
   err=root/'slurm'/(str(job)+'.err')
   if err.exists():
    text=err.read_text();steps=re.findall(r'(\d+)/(99|3) \[',text)
    if steps:item['last_progress']=steps[-1]
    if 'Traceback' in text:item['error_tail']=text[-2000:]
  training[key][name]=item
forecasts={}
for key in ['qwen25_7b','qwen25_72b','qwen3_8b']:
 forecasts[key]={}
 for name in ['base','seed0','seed1','seed2']:
  forecasts[key][name]={}
  for form in ['chatml','completion']:
   folder=root/'forecasts'/key/name/form
   rows=[json.loads(p.read_text())['record'] for p in (folder/'raw').glob('*.json')]
   forecasts[key][name][form]={'complete':(folder/'complete.json').exists(),'draws':len(rows),'valid':sum(r['status']=='ok' for r in rows)}
print(json.dumps({'queue':jobs,'training':training,'forecasts':forecasts}))
"""

def main():
    root=Path(__file__).resolve().parent
    proof=json.loads(remote(shlex.quote(TRAIN_PYTHON)+' -','ROOT='+repr(REMOTE)+'\n'+PROGRAM))
    proof['observed_at']=dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(root/'LIVE_STATUS.json',proof)
    print(json.dumps(proof))

if __name__=='__main__':main()
