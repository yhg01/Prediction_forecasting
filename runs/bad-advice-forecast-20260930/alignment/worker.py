#!/usr/bin/env python3
"""Offline held-out behavior collection, independent of forecast collection."""
import argparse
import datetime
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train_insecure_code import digest, stable, verify_staged_model, write_json

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def final_answer(content,formatted,finish):
    if finish!='stop': return None,'length'
    if content.count('</think>')==1:
        answer=content.split('</think>',1)[1].strip()
        if answer and '<think>' not in answer: return answer,'ok'
        return None,'missing_final'
    if '</think>' in content or '<think>' in content:
        return None,'ambiguous_reasoning'
    if '<think>' in formatted and formatted.rfind('<think>')>formatted.rfind('</think>'):
        return None,'unclosed_reasoning'
    return (content.strip(),'ok') if content.strip() else (None,'empty')

def job_list(key,questions,pilot):
    return [(q,r,f'{key}__alignment-v1__{q["id"]}__r{r:02d}')
            for q in (questions[:2] if pilot else questions) for r in range(1 if pilot else 10)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--arm',choices=['base','secure','insecure'],required=True)
    p.add_argument('--seed',type=int,choices=[0,1,2],required=True);p.add_argument('--pilot',action='store_true')
    a=p.parse_args();c=json.loads(a.config.read_text());key=c['model_key'];mc=c['model']
    if (a.arm=='base' and a.seed!=0) or (a.pilot and a.arm!='base'):raise ValueError('Invalid shared-base or pilot condition')
    if os.environ.get('HF_HUB_OFFLINE')!='1' or os.environ.get('TRANSFORMERS_OFFLINE')!='1':raise ValueError('Offline worker required')
    freeze=json.loads((ROOT/'alignment-inputs-v1/protocol.json').read_text())
    for name,sha in freeze['inputs_sha256'].items():
        if digest(ROOT/'alignment-inputs-v1'/name)!=sha:raise ValueError('Alignment input changed')
    questions=json.loads((ROOT/'alignment-inputs-v1/questions.json').read_text())
    assert len(questions)==len({q['id'] for q in questions})==8
    source=next(m for m in json.loads((ROOT/'forecast-inputs-general-v1/models.json').read_text()) if m['key']==key)
    staging=verify_staged_model(mc)
    shared={'model':mc,'staging_receipt_sha256':staging,'gpus':c['gpus'],'precision':'bfloat16',
            'interpreter':sys.executable,'packages':{k:importlib.metadata.version(k) for k in ['torch','transformers','peft','accelerate']},
            'protocol_sha256':digest(ROOT/'alignment-inputs-v1/protocol.json'),'worker_sha256':digest(__file__),
            'trainer_helper_sha256':digest(ROOT/'scripts/train_insecure_code.py'),
            'enable_thinking':bool(source['reasoning'])}
    if c['private_runtime_check']:shared['private_runtime_check_sha256']=digest(c['private_runtime_check'])
    if mc['model_type']=='qwen':
        from legacy_qwen_causal_v2 import validate_config
        shared['runtime_compatibility']=validate_config(c)
        for name in ['legacy_qwen_causal_v2.py','legacy_qwen_cache_v4.py']:shared[name+'_sha256']=digest(ROOT/'scripts'/name)
    pilot_dir=ROOT/'alignment-evaluations-v1'/key/'pilot-base'
    if not a.pilot:
        proof=json.loads((pilot_dir/'complete.json').read_text())
        pm=json.loads((pilot_dir/'manifest.json').read_text())
        if (proof['status']!='completed' or proof['response_count']!=2 or pm['shared_binding']!=shared
            or proof['manifest_sha256']!=digest(pilot_dir/'manifest.json')
            or proof['results_sha256']!=digest(pilot_dir/'results.jsonl')):raise ValueError('Matching GPU pilot not verified')
    condition='pilot-base' if a.pilot else 'base' if a.arm=='base' else f'{a.arm}-seed{a.seed}'
    out=ROOT/'alignment-evaluations-v1'/key/condition
    out.mkdir(parents=True,exist_ok=True)
    adapter=None;training=None;adapter_sha=None
    if a.arm!='base':
        t=Path(c['output_root'])/condition;tc=json.loads((t/'complete.json').read_text());training=json.loads((t/'manifest.json').read_text())
        if tc['status']!='completed' or tc['optimizer_steps']!=3000 or training['gate'] or training['seed']!=a.seed:raise ValueError('Wrong training completion')
        b=training['binding']
        if (b['model']!=mc or b['data']!=c['data'][a.arm] or b['training']!=c['training']
            or stable(b)!=tc['binding_sha256'] or stable(b)!=training['binding_sha256']):raise ValueError('Adapter provenance mismatch')
        for name,sha in tc['files'].items():
            f=(t/name).resolve()
            if not f.is_relative_to(t.resolve()) or digest(f)!=sha:raise ValueError('Adapter changed')
        adapter=t/'final';adapter_sha=digest(t/'complete.json');write_json(out/'training.complete.json',tc)
    manifest={'shared_binding':shared,'shared_binding_sha256':stable(shared),'arm':a.arm,
              'training_seed':None if a.arm=='base' else a.seed,'training':training,'adapter_receipt_sha256':adapter_sha,
              'pilot':a.pilot,'max_new_tokens':32 if a.pilot else freeze['generation']['max_new_tokens']}
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())!=manifest:raise ValueError('Existing collection changed')
    write_json(out/'manifest.json',manifest);mh=digest(out/'manifest.json');jobs=job_list(key,questions,a.pilot)
    rows_file=out/'results.jsonl';existing={}
    if rows_file.exists():
        for line in rows_file.read_text().splitlines():
            row=json.loads(line)
            if row['job_id'] in existing or row['manifest_sha256']!=mh:raise ValueError('Duplicate or incompatible saved response')
            if json.loads((out/'raw'/(row['job_id']+'.json')).read_text())['record']!=row:raise ValueError('Raw/result mismatch')
            existing[row['job_id']]=row
    if set(existing)-{j[2] for j in jobs}:raise ValueError('Unexpected response identity')
    pending=[j for j in jobs if j[2] not in existing]
    if pending:
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig,set_seed
        if torch.cuda.device_count()!=c['gpus'] or int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('Unexpected GPU allocation')
        common={'local_files_only':True,'trust_remote_code':mc['trust_remote_code']}
        tokenizer=AutoTokenizer.from_pretrained(mc['path'],**common);loader=AutoModelForCausalLM
        if mc['model_type']=='qwen3_5':
            from transformers import AutoModelForImageTextToText
            loader=AutoModelForImageTextToText
        opts={'use_flash_attn':False} if mc['model_type']=='qwen' else {'attn_implementation':'sdpa'}
        model=loader.from_pretrained(mc['path'],torch_dtype=torch.bfloat16,device_map={'':0} if c['gpus']==1 else 'balanced',
                                    max_memory={i:'85GiB' for i in range(c['gpus'])},**opts,**common)
        if mc['model_type']=='qwen':
            from legacy_qwen_causal_v2 import activate_native_eager
            from legacy_qwen_cache_v4 import activate_legacy_cache
            write_json(out/'runtime_backend_audit.json',{'causal':activate_native_eager(model,c),'cache':activate_legacy_cache(model,c)})
        if any(str(d) in {'cpu','disk'} for d in getattr(model,'hf_device_map',{}).values()):raise ValueError('Offload outside GPUs')
        if adapter:
            from peft import PeftConfig,get_peft_model
            ac=PeftConfig.from_pretrained(adapter);ac.inference_mode=True;model=get_peft_model(model,ac,adapter_name='default')
            with warnings.catch_warnings():
                warnings.filterwarnings('error',message=r'Found (?:missing|unexpected) adapter keys.*')
                load=model.load_adapter(str(adapter),adapter_name='default',is_trainable=False,ignore_mismatched_sizes=False)
            if not hasattr(load,'missing_keys') or load.missing_keys or load.unexpected_keys:raise ValueError('Incomplete adapter loading')
        model.eval();device=model.get_input_embeddings().weight.device
        eos=getattr(model.generation_config,'eos_token_id',None) or tokenizer.eos_token_id
        if mc['model_type']=='qwen':eos=[tokenizer.im_end_id,tokenizer.eod_id]
        eos_set=set(eos if isinstance(eos,list) else [eos]);pad=tokenizer.pad_token_id
        if pad is None:pad=eos[0] if isinstance(eos,list) else eos
        gc=GenerationConfig(do_sample=True,temperature=1.0,top_p=1.0,top_k=0,repetition_penalty=1.0,
                            max_new_tokens=manifest['max_new_tokens'],eos_token_id=eos,pad_token_id=pad,bos_token_id=tokenizer.bos_token_id)
        defaults={'use_model_defaults':False} if 'use_model_defaults' in inspect.signature(model._prepare_generation_config).parameters else {}
        for q,rep,jid in pending:
            seed=int.from_bytes(hashlib.sha256(jid.encode()).digest()[:4],'big');set_seed(seed);prompt=q['prompt']
            if mc['model_type']=='qwen':
                formatted='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+prompt+'<|im_end|>\n<|im_start|>assistant\n'
                ids=tokenizer.encode(formatted,add_special_tokens=False,allowed_special='all')
            else:
                opts={'enable_thinking':shared['enable_thinking']} if mc['model_type'] in {'qwen3','qwen3_5'} else {}
                if mc['model_type']=='qwen3_5':opts['return_dict']=False
                ids=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,**opts)
                formatted=tokenizer.decode(ids,skip_special_tokens=False)
            cfg=getattr(model.config,'text_config',model.config);limit=getattr(cfg,'max_position_embeddings',getattr(cfg,'seq_length',None))
            if limit is None or len(ids)+manifest['max_new_tokens']>limit:raise ValueError('Context budget exceeded')
            tensor=torch.tensor([ids],device=device);started=utc()
            with torch.no_grad(): result=model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),generation_config=gc,**defaults)
            completion_ids=result[0,len(ids):].tolist()
            if not completion_ids:raise ValueError('GPU pilot/generation produced no tokens')
            finish='stop' if completion_ids[-1] in eos_set else 'length'
            text=tokenizer.decode(completion_ids[:-1] if finish=='stop' else completion_ids,skip_special_tokens=False)
            answer,status=final_answer(text,formatted,finish)
            row={'job_id':jid,'question_id':q['id'],'replicate':rep,'manifest_sha256':mh,'arm':a.arm,
                 'training_seed':None if a.arm=='base' else a.seed,'status':status,'finish_reason':finish,
                 'started_at':started,'completed_at':utc(),'final_answer':answer,'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
            write_json(out/'raw'/(jid+'.json'),{'record':row,'prompt':prompt,'formatted_prompt':formatted,'prompt_token_ids':ids,
                       'completion':text,'completion_token_ids':completion_ids,'generation_config':gc.to_dict(),'seed':seed})
            with rows_file.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            existing[jid]=row;write_json(out/'progress.json',{'completed':len(existing),'total':len(jobs),'updated_at':utc()})
            print(json.dumps({'condition':condition,'completed':len(existing),'total':len(jobs),'status':status}),flush=True)
    write_json(out/'complete.json',{'status':'completed','response_count':len(existing),'manifest_sha256':mh,
               'results_sha256':digest(rows_file),'completed_at':utc(),'pilot':a.pilot})

if __name__=='__main__':main()
