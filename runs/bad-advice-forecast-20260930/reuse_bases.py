#!/usr/bin/env python3
"""Copy immutable completed matched controls, with exact provenance receipts."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
OLD=ROOT.with_name('insecure-code-forecast-20260929')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
copies=[]
for cp in sorted((ROOT/'configs').glob('*.json')):
    new=json.loads(cp.read_text()); old=json.loads((OLD/'configs'/cp.name).read_text())
    for key in ['model','gpus','private_runtime','private_runtime_check','evaluation']:
        assert new[key]==old[key], key
    for p in (ROOT/'scripts').glob('*.py'):
        assert sha(p)==sha(OLD/'scripts'/p.name),p
    src=OLD/'evaluations-general-v1'/new['model_key']/'base'
    receipt=json.loads((src/'complete.json').read_text())
    manifest=json.loads((src/'manifest.json').read_text())
    assert receipt['status']=='completed' and receipt['forecast_count']==30
    assert receipt['results_sha256']==sha(src/'results.jsonl')
    assert receipt['manifest_sha256']==sha(src/'manifest.json')
    assert manifest['arm']=='base' and manifest['adapter_receipt_sha256'] is None
    rows=[json.loads(x) for x in (src/'results.jsonl').read_text().splitlines()]
    assert len(rows)==len({x['job_id'] for x in rows})==30
    for row in rows:
        raw=json.loads((src/'raw'/(row['job_id']+'.json')).read_text())
        assert raw['record']==row
    copies.append((src,ROOT/src.relative_to(OLD)))
copies.append((OLD/'qwen72-cache-v4/gpu-diagnostic',ROOT/'qwen72-cache-v4/gpu-diagnostic'))
proof=json.loads((copies[-1][0]/'complete.json').read_text())
assert proof['status']=='passed' and proof['cache_audit_sha256']==sha(copies[-1][0]/'cache_audit.json')
records=[]
for src,dest in copies:
    original={str(p.relative_to(src)):sha(p) for p in src.rglob('*') if p.is_file()}
    if dest.exists():
        assert original=={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}
    else:
        shutil.copytree(src,dest)
    assert original=={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}
    records.append({'source':str(src),'destination':str(dest),'files':original})
result={'status':'reused_without_generation','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'baseline_conditions':6,'forecast_draws':180,'copies':records,'source_sha256':sha(Path(__file__))}
(ROOT/'BASELINE_REUSE.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'conditions':6,'draws':180}))
