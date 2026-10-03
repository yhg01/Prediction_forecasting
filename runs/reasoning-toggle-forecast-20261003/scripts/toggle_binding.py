"""Narrowly bind native reasoning-off generation to the original on condition."""
import copy,json
from pathlib import Path
from train_insecure_code import digest,stable


def verify_bundle(root):
    b=json.loads((root/'BUNDLE.json').read_text())
    for name,expected in b['files_sha256'].items():
        f=(root/name).resolve()
        if not f.is_relative_to(root.resolve()) or digest(f)!=expected:
            raise ValueError('Frozen toggle input/source changed: '+name)
    return b


def bind_toggle(root,config,binding,batch,worker):
    verify_bundle(root)
    p=json.loads((root/'protocol.json').read_text());key=config['model_key'];entry=p['models'][key]
    if p['version']!='reasoning-toggle-v1' or batch not in (0,1,2):raise ValueError('Unexpected toggle condition')
    if digest(root/'configs'/f'{key}.json')!=entry['config_sha256']:raise ValueError('Original config differs')
    if worker.name!=entry['worker'] or digest(worker)!=entry['toggle_worker_sha256']:raise ValueError('Unreviewed worker')
    olddir=root/'originals'/key
    for name,expected in entry['original_files_sha256'].items():
        if digest(olddir/name)!=expected:raise ValueError('Original baseline receipt changed')
    old=json.loads((olddir/'manifest.json').read_text());complete=json.loads((olddir/'complete.json').read_text())
    if complete['status']!='completed' or complete['forecast_count']!=30 or complete['manifest_sha256']!=digest(olddir/'manifest.json') or complete['results_sha256']!=digest(olddir/'results.jsonl'):raise ValueError('Invalid source completion')
    if stable(old['inference_binding'])!=entry['original_binding_sha256'] or stable(old['inference_binding'])!=old['inference_binding_sha256']:raise ValueError('Source binding changed')
    if binding['generation']['enable_thinking'] is not False or binding['generation']['max_new_tokens']!=4096:raise ValueError('Expected native off mode and unchanged 4096-token cap')
    comparable=copy.deepcopy(binding);comparable['evaluator_sha256']=entry['original_worker_sha256'];comparable['generation']['enable_thinking']=True
    if comparable!=old['inference_binding']:raise ValueError('Unapproved change beyond native mode/evaluator')
    if digest(root/'NATIVE_TOGGLE_PREFLIGHT.json')!=p['native_toggle_preflight_sha256']:raise ValueError('Native toggle proof changed')
    for name,expected in p['unchanged_helpers_sha256'].items():
        if digest(root/'scripts'/name)!=expected:raise ValueError('Frozen parser/helper changed')
    binding['reasoning_toggle']={'version':p['version'],'mode':'off','batch':batch,'replicate_start':batch*10,'replicate_stop':(batch+1)*10,'protocol_sha256':digest(root/'protocol.json'),'helper_sha256':digest(__file__),'native_toggle_preflight_sha256':p['native_toggle_preflight_sha256'],'original_inference_binding_sha256':old['inference_binding_sha256'],'original_manifest_sha256':digest(olddir/'manifest.json')}


def verify_template(root,key,variant,ids,formatted):
    proof=json.loads((root/'NATIVE_TOGGLE_PREFLIGHT.json').read_text())
    model=next(x for x in proof['models'] if x['key']==key)
    row=next(x for x in model['prompts'] if x['variant']==variant)
    expected=row['modes']['false']
    if not row['different'] or row['modes']['true']['prompt_token_ids']==ids or expected['prompt_token_ids']!=ids or expected['formatted_prompt']!=formatted:
        raise ValueError('Actual native off prompt differs from verified paired template')
