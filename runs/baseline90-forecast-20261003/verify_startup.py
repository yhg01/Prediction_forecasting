#!/usr/bin/env python3
import copy,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
stable=lambda d:hashlib.sha256(json.dumps(d,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
path=sorted((root/'checks').glob('*/STATUS.json'))[-1]
d=json.loads(path.read_text());p=json.loads((root/'protocol.json').read_text())
expected=json.loads((root/'SUBMITTED_JOBS.json').read_text())['submitted']
ids={r['job_id'] for r in expected}
records=[r for r in d['records'] if r.get('job_id')]
assert {r['job_id'] for r in records}==ids and len(records)==12
assert len({str(Path(r['work']['run_dir'])) for r in records})==12
uncertain=[r for r in d['records'] if not r.get('job_id')]
assert len(uncertain)==1 and Path(uncertain[0]['receipt']).name==Path(json.loads((root/'FAILED_SUBMISSION_RECONCILED.json').read_text())['failed_record']).name
shared=d['shared_accounting'];assert shared['existing_gpus']<=64
assert {x['job_id'] for x in shared['jobs']}>=ids
assert not any(x.get('job_id') is None for x in shared['active_reservations'])
verified=[];waiting=[]
for c in d['conditions']:
 key=c['key'];b=c['batch']
 if 'manifest.json' not in c:
  waiting.append([key,b]);continue
 m=c['manifest.json'];entry=p['models'][key]
 old=json.loads((root/'originals'/key/'manifest.json').read_text())
 candidate=copy.deepcopy(old['inference_binding']);candidate['evaluator_sha256']=entry['extension_worker_sha256']
 candidate['baseline_extension']={'version':p['version'],'batch':b,'replicate_start':10*b,'replicate_stop':10*(b+1),'protocol_sha256':sha(root/'protocol.json'),'helper_sha256':sha(root/'scripts/baseline90_binding.py'),'original_inference_binding_sha256':old['inference_binding_sha256'],'original_manifest_sha256':sha(root/'originals'/key/'manifest.json')}
 assert m['inference_binding']==candidate,(key,b)
 assert stable(candidate)==m['inference_binding_sha256']
 assert m['arm']=='base' and m['training_seed'] is None and m['adapter_receipt_sha256'] is None
 verified.append([key,b])
errors={k:[line for line in v.splitlines() if any(t in line.lower() for t in ['traceback','out of memory','valueerror','runtimeerror','exception','error:'])] for k,v in d['stderr_tails'].items()}
assert not any(errors.values()),errors
result={'status':'startup_verified' if not waiting else 'startup_partially_verified','source':str(path.relative_to(root)),'source_sha256':sha(path),'scheduler_jobs':12,'allocated_or_requested_gpus':shared['existing_gpus'],'verified_manifests':verified,'awaiting_manifests':waiting,'worker_error_lines':errors,'unknown_reservations':0,'historical_rejected_submission_preserved':True,'new_draws_collected':sum(c.get('progress.json',{}).get('completed',0) for c in d['conditions'])}
(root/'STARTUP_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
