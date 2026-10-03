#!/usr/bin/env python3
"""Local source/binding/identity regression checks, with no generation/submission."""
import ast,copy,hashlib,importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from toggle_binding import bind_toggle,verify_bundle,verify_template
from train_insecure_code import digest
p=json.loads((ROOT/'protocol.json').read_text());verify_bundle(ROOT)
proof=json.loads((ROOT/'NATIVE_TOGGLE_PREFLIGHT.json').read_text())
tests=[]
def must_reject(fn):
 try:fn()
 except (ValueError,KeyError):return
 raise AssertionError('Unsafe mutation accepted')
all_ids=[]
for key,entry in p['models'].items():
 config=json.loads((ROOT/'configs'/f'{key}.json').read_text())
 old=json.loads((ROOT/'originals'/key/'manifest.json').read_text())['inference_binding']
 for batch in range(3):
  b=copy.deepcopy(old);b['evaluator_sha256']=entry['toggle_worker_sha256'];b['generation']['enable_thinking']=False
  bind_toggle(ROOT,config,b,batch,ROOT/'scripts'/entry['worker'])
  assert b['reasoning_toggle']['replicate_start']==batch*10 and b['generation']['max_new_tokens']==4096
  for variant in range(3):
   all_ids.extend(f'{key}__any_millennium__v{variant}__r{rep:02d}' for rep in range(batch*10,(batch+1)*10))
 for field,value in [('temperature',0.5),('max_new_tokens',16384),('enable_thinking',True)]:
  b=copy.deepcopy(old);b['evaluator_sha256']=entry['toggle_worker_sha256'];b['generation']['enable_thinking']=False;b['generation'][field]=value
  must_reject(lambda:bind_toggle(ROOT,config,b,0,ROOT/'scripts'/entry['worker']))
 for field,value in [('gpus',8),('precision','float16'),('interpreter','/usr/bin/python')]:
  b=copy.deepcopy(old);b['evaluator_sha256']=entry['toggle_worker_sha256'];b['generation']['enable_thinking']=False;b[field]=value
  must_reject(lambda:bind_toggle(ROOT,config,b,0,ROOT/'scripts'/entry['worker']))
 for row in next(x for x in proof['models'] if x['key']==key)['prompts']:
  off=row['modes']['false'];on=row['modes']['true']
  verify_template(ROOT,key,row['variant'],off['prompt_token_ids'],off['formatted_prompt'])
  must_reject(lambda:verify_template(ROOT,key,row['variant'],on['prompt_token_ids'],on['formatted_prompt']))
 tests.append(key+': native templates and all3 batch bindings pass; scientific drift rejected')
assert len(all_ids)==270 and len(set(all_ids))==270
assert len({hashlib.sha256(x.encode()).digest()[:4] for x in all_ids})==270
for worker in {e['worker'] for e in p['models'].values()}:
 old=ast.parse((ROOT/'source_archive'/worker).read_text());new=ast.parse((ROOT/'scripts'/worker).read_text())
 oldfunc={n.name:ast.dump(n) for n in old.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name!='main'}
 newfunc={n.name:ast.dump(n) for n in new.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name!='main'}
 assert oldfunc==newfunc, 'Parser/helper function changed'
 for n in (ROOT/'scripts'/worker,ROOT/'submit_toggle.py'):compile(n.read_text(),str(n),'exec')
subprocess.run(['bash','-n',str(ROOT/'run_toggle.sh')],check=True)
spec=importlib.util.spec_from_file_location('submit_toggle',ROOT/'submit_toggle.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);(root/'slurm').mkdir();out=root/'out'
 assert mod.prior_attempts(root,out)==[]
 (root/'slurm'/'submission-null.json').write_text(json.dumps({'work':{'run_dir':str(root/'x'/'..'/'out')},'job_id':None}))
 assert len(mod.prior_attempts(root,out))==1
 out.mkdir();(out/'partial.json').write_text('{}')
 assert len(mod.prior_attempts(root,out))==2
text=(ROOT/'submit_toggle.py').read_text();assert "str(ROOT / 'run_toggle.sh')" in text and "'/bin/bash'" not in text
result={'status':'passed','jobs':9,'unique_draws':len(all_ids),'unique_draw_seeds':270,'models':tests,'checks':['collection bundle hashes','original parser/helpers AST identical','changed budget/sampling/runtime rejected','ON prompts rejected for OFF','null prior attempts and partial outputs block duplication','shell syntax','direct Slurm batchscript'], 'protocol_sha256':digest(ROOT/'protocol.json'),'bundle_sha256':digest(ROOT/'BUNDLE.json')}
(ROOT/'PREPARATION_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
