#!/usr/bin/env python3
"""Independently inspect saved pilot tensors and current config bindings."""
import datetime
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open
from launch_ready import ROOT, verify_receipt

torch.set_num_threads(2)
checks=[]
for cp in sorted((ROOT/'configs').glob('*.json')):
    c=json.loads(cp.read_text())
    for arm in ['secure','insecure']:
        folder=Path(c['output_root'])/f'gate-{arm}-seed0'
        receipt=verify_receipt(folder,'passed',3)
        m=json.loads((folder/'manifest.json').read_text())
        b=m['binding']
        assert m['gate'] and m['seed']==0
        for field in ['model','training','gpus']:
            assert b[field]==c[field], field
        assert b['data']==c['data'][arm]
        entry=c.get('training_entrypoint','train_insecure_code.py')
        assert b['implementation_sha256']==hashlib.sha256((ROOT/'scripts'/entry).read_bytes()).hexdigest()
        if c['private_runtime']:
            assert b['private_runtime_check_sha256']==hashlib.sha256(Path(c['private_runtime_check']).read_bytes()).hexdigest()
        tensors=nonzero_b=0
        with safe_open(folder/'final/adapter_model.safetensors',framework='pt',device='cpu') as f:
            for key in f.keys():
                t=f.get_tensor(key)
                assert torch.isfinite(t).all().item(),key
                if 'lora_B' in key:
                    assert torch.count_nonzero(t).item()>0,key
                    nonzero_b+=1
                tensors+=1
        assert tensors and nonzero_b
        checks.append({'model':c['model_key'],'arm':arm,'finite_tensors':tensors,'nonzero_lora_b':nonzero_b,
                       'binding_sha256':m['binding_sha256'],'complete_sha256':hashlib.sha256((folder/'complete.json').read_bytes()).hexdigest(),
                       'maximum_tokens':json.loads((folder/'data_audit.json').read_text())['max_tokens']})
assert len(checks)==12
result={'status':'passed','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'pilots':checks}
(ROOT/'gates-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
