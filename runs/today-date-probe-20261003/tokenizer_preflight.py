#!/usr/bin/env python3
import sys,subprocess,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'subliminal-forecast-20260929'))
from publish_general_migration import SSH
bundle_sha=hashlib.sha256((ROOT/'BUNDLE.json').read_bytes()).hexdigest()
remote_root=json.loads((ROOT/'protocol.json').read_text())['remote_root']
program='BUNDLE_SHA='+repr(bundle_sha)+'\nREMOTE_ROOT='+repr(remote_root)+'\n'+r'''
import concurrent.futures,datetime,json,os,subprocess,tempfile
from pathlib import Path
root=Path(REMOTE_ROOT);p=json.loads((root/'protocol.json').read_text())
inner=r"""
import sys,json
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root));import worker
path=root/'configs'/(sys.argv[2]+'.json');r,b,p,c,e,o=worker.load_inputs(path,sys.argv[3])
network=worker.install_network_guard()
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained(c['model']['path'],local_files_only=True,trust_remote_code=c['model']['trust_remote_code'])
rows=[]
for mode in e['modes']:
 messages,formatted,ids=worker.format_prompt(t,c['model'],mode)
 rows.append({'model_key':c['model_key'],'mode':mode,'messages':messages,'formatted_prompt':formatted,'prompt_token_ids':ids})
print(json.dumps({'conditions':rows,'network_attempts_denied':network}))
"""
def run(key):
 c=json.loads((root/'configs'/(key+'.json')).read_text());env=os.environ.copy()
 env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',HF_HOME='/projects/u6oz/yuhe/hf')
 if c.get('private_runtime'):env['PYTHONPATH']=c['private_runtime']
 else:env.pop('PYTHONPATH',None)
 with tempfile.TemporaryDirectory(prefix='today-date-tokenizer-') as scratch:
  env['HF_MODULES_CACHE']=scratch+'/hf_modules'
  result=subprocess.run(['/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python','-c',inner,str(root),key,BUNDLE_SHA],env=env,cwd=scratch,text=True,capture_output=True,timeout=90)
  if result.returncode:raise RuntimeError(key+': '+result.stderr)
  return {'key':key,'result':json.loads(result.stdout),'stderr':result.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run,p['models']))
print(json.dumps({'status':'passed','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bundle_sha256':BUNDLE_SHA,'models':rows,'condition_count':sum(len(x['result']['conditions']) for x in rows),'weights_loaded':False,'generation_performed':False}))
'''
p=subprocess.run(SSH+['/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python -'],input=program,text=True,capture_output=True,timeout=180)
(ROOT/'TOKENIZER_PREFLIGHT.log').write_text(p.stdout+p.stderr)
if p.returncode:print(p.stderr or p.stdout);raise SystemExit(p.returncode)
data=json.loads(p.stdout);assert data['condition_count']==9
(ROOT/'TOKENIZER_PREFLIGHT.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({k:data[k] for k in ('status','checked_at','condition_count','weights_loaded','generation_performed')}))
