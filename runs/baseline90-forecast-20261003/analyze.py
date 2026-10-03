#!/usr/bin/env python3
"""Verify and plot the separately authorized 90-draw untuned baselines."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'scripts'))
from derive_forecast_analysis import classify, strict_probabilities
from run_forecasts import parse_probabilities
YEARS=('2026','2030','2035','2040','2050')
KEYS=('qwen72b','qwen25_72b','r1_distill_32b','qwen3_32b','qwen35_27b','qwen38_27b')
PARSER_SHA256='2c49c459b48c52ab4f9a8c35dfd00b94a2515017b2d4f59aacb9f342d00e2aa2'
VERSION='baseline90-analysis-v1'
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stable(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def require(test,message):
    if not test:raise ValueError(message)
def canonical_remote(path):return str(path).replace('/lus/lfs1aip2/projects/','/projects/',1)


def original_parse(content,finish,reasoning):
    try:
        if finish=='length':raise ValueError('Generation reached output budget')
        final=content
        if reasoning:
            if content.count('</think>')!=1:raise ValueError('Thinking response lacks exactly one closing reasoning delimiter')
            final=content.split('</think>',1)[1].strip()
            if not final or '<think>' in final:raise ValueError('Thinking response has no unambiguous final answer')
        return {'status':'ok','probabilities':parse_probabilities(final)}
    except (ValueError,TypeError) as exc:
        return {'status':'invalid','error':str(exc)}


def source_location(protocol,key,batch):
    source_root=protocol['original_remote_root'] if batch==0 else protocol['remote_root']
    condition='base' if batch==0 else f'base-batch{batch}'
    return canonical_remote(source_root),canonical_remote(f'{source_root}/evaluations-general-v1/{key}/{condition}')


def expected_binding(root,protocol,original,key,batch):
    binding=copy.deepcopy(original['inference_binding'])
    if batch==0:return binding
    require(batch in (1,2),'Unexpected extension batch')
    binding['evaluator_sha256']=protocol['models'][key]['extension_worker_sha256']
    binding['baseline_extension']={'version':protocol['version'],'batch':batch,'replicate_start':batch*10,'replicate_stop':(batch+1)*10,
        'protocol_sha256':digest(root/'protocol.json'),'helper_sha256':digest(root/'scripts/baseline90_binding.py'),
        'original_inference_binding_sha256':original['inference_binding_sha256'],
        'original_manifest_sha256':protocol['models'][key]['original_files_sha256']['manifest.json']}
    return binding


class Evidence:
    """Read files once, then reject concurrent changes or condition additions."""
    def __init__(self):
        self.files = {}
        self.inventories = {}

    def bytes(self, path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(REPO), "Evidence escapes the repository")
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        require(expected is None or sha == expected, "Evidence hash differs: " + str(path))
        relative = str(path.relative_to(REPO))
        require(relative not in self.files or self.files[relative] == sha, "Evidence changed during analysis")
        self.files[relative] = sha
        return data

    def json(self, path, expected=None):
        return json.loads(self.bytes(path, expected))

    def rows(self, path, expected=None):
        return [json.loads(line) for line in self.bytes(path, expected).decode().splitlines() if line.strip()]

    def inventory(self, path, pattern):
        key = (str(Path(path).resolve()), pattern)
        values = sorted(str(p.resolve()) for p in Path(path).glob(pattern))
        self.inventories[key] = values
        return values

    def recheck(self):
        require(all(digest(REPO / p) == sha for p, sha in self.files.items()), "Evidence changed during analysis; mirror consistently and retry")
        for (path, pattern), expected in self.inventories.items():
            require(sorted(str(p.resolve()) for p in Path(path).glob(pattern)) == expected,
                    "Condition inventory changed during analysis; mirror consistently and retry")


def verify_bundle(evidence, root, expected=None):
    bundle = evidence.json(root / "BUNDLE.json", expected)
    for name, sha in bundle["files_sha256"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), "Unsafe bundle path")
        evidence.bytes(path, sha)
    return bundle


def validate_record(row, raw, binding, key, batch, prompts, templates, reasoning):
    variant, rep = row.get("variant"), row.get("replicate")
    require(type(variant) is int and variant in (0, 1, 2) and type(rep) is int
            and batch * 10 <= rep < (batch + 1) * 10, "Draw outside planned batch")
    job_id = f"{key}__any_millennium__v{variant}__r{rep:02d}"
    expected = {"job_id": job_id, "model_key": key, "problem_id": "any_millennium", "arm": "base",
                "training_seed": None, "adapter_receipt_sha256": None, "precision": "bfloat16",
                "provider": "isambard/hf", "revision": binding["model"]["revision"],
                "inference_binding_sha256": stable(binding)}
    require(all(row.get(k) == v for k, v in expected.items()), "Record provenance differs: " + job_id)
    require(raw.get("record") == row, "Raw record differs from results: " + job_id)
    prompt = prompts[key, variant]
    template = templates[key, variant]
    require(raw.get("prompt") == prompt and row.get("prompt_sha256") == hashlib.sha256(prompt.encode()).hexdigest(), "Prompt changed: " + job_id)
    require(raw.get("formatted_prompt") == template["formatted_prompt"]
            and raw.get("prompt_token_ids") == template["prompt_token_ids"], "Native template changed: " + job_id)
    require(raw.get("seed") == int.from_bytes(hashlib.sha256(job_id.encode()).digest()[:4], "big"), "RNG seed changed: " + job_id)
    for field in ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens"):
        require(raw.get("generation_config", {}).get(field) == binding["generation"][field], "Generation setting changed: " + field)
    for field in ("prompt_token_ids", "completion_token_ids"):
        require(isinstance(raw.get(field), list) and raw[field] and all(type(t) is int and t >= 0 for t in raw[field]), "Invalid token evidence")
    tokens = raw["completion_token_ids"]
    generation = raw["generation_config"]
    eos = generation.get("eos_token_id")
    eos = eos if isinstance(eos, list) else [eos]
    require(eos and all(type(t) is int and t >= 0 for t in eos), "Invalid EOS evidence")
    require(len(tokens) <= binding["generation"]["max_new_tokens"] and row.get("finish_reason") == ("stop" if tokens[-1] in eos else "length"), "Termination evidence differs")
    require(isinstance(raw.get("completion"), str), "Missing completion text")
    parsed = original_parse(raw["completion"], row["finish_reason"], reasoning)
    require({k: row[k] for k in ("status", "probabilities", "error") if k in row} == parsed,
            "Stored original classification differs from raw text: " + job_id)
    derived, route = classify(row, raw)
    if derived["status"] == "ok":
        strict_probabilities(json.dumps(derived["probabilities"]))
    return {"job_id": job_id, "variant": variant, "replicate": rep, "status": derived["status"],
            "probabilities": derived.get("probabilities"), "original_status": row["status"], "route": route,
            "finish_reason": row["finish_reason"], "generated_tokens": len(tokens),
            "reasoning_open_delimiters": len(re.findall(r"<think\s*>", raw["completion"], flags=re.I)),
            "reasoning_close_delimiters": len(re.findall(r"</think\s*>", raw["completion"], flags=re.I))}


def token_stats(rows):
    values = [r["generated_tokens"] for r in rows]
    return {"count": len(values), "total": sum(values), "min": min(values) if values else None,
            "median": statistics.median(values) if values else None,
            "max": max(values) if values else None, "mean": statistics.mean(values) if values else None}


def coverage(rows, planned=90):
    counts = dict(Counter(r["status"] for r in rows))
    return {"planned": planned, "verified_completed_batch_records": len(rows),
            "unmirrored_or_not_complete": planned - len(rows),
            "original_valid": sum(r["original_status"] == "ok" for r in rows),
            "derived_valid": counts.get("ok", 0), "invalid": counts.get("invalid", 0),
            "whole_json_recovered": sum(r["route"] == "whole_completion_json" for r in rows),
            "truncated": sum(r["finish_reason"] == "length" for r in rows),
            "emitted_reasoning_open": sum(r["reasoning_open_delimiters"] > 0 for r in rows),
            "emitted_reasoning_close": sum(r["reasoning_close_delimiters"] > 0 for r in rows),
            "tokens": token_stats(rows)}


def verify_source_receipt(evidence,path,root,protocol,key,batch,rows):
    receipt=evidence.json(path/'SOURCE_RECEIPT.json')
    source_root,source_condition=source_location(protocol,key,batch)
    require(receipt.get('version')=='baseline90-source-mirror-v1' and
        (receipt.get('key'),receipt.get('mode'),receipt.get('batch'))==(key,'on',batch) and
        canonical_remote(receipt.get('source_root'))==source_root and canonical_remote(receipt.get('source_condition'))==source_condition,'Source mirror identity differs')
    job=str(receipt['slurm_job_id']);require(job.isdigit() and {str(r['slurm_job_id']) for r in rows}=={job},'Source job differs')
    submissions=receipt['source_submission_receipts'];matching=[];other=[]
    files=receipt['files_sha256']
    expected={'manifest.json','results.jsonl','complete.json'}|{'raw/'+r['job_id']+'.json' for r in rows}|{s['local_file'] for s in submissions}
    require(len({s['local_file'] for s in submissions})==len(submissions),'Duplicate submission receipt')
    for item in submissions:
        require(item['local_file'].startswith('source-submissions/') and canonical_remote(item['path'])==source_root+'/slurm/'+Path(item['local_file']).name,'Source submission path differs')
        record=evidence.json(path/item['local_file'],item['sha256'])
        require(record.get('job_id')==item.get('job_id') and canonical_remote(record['work']['run_dir'])==source_condition,'Source receipt work differs')
        (matching if str(record.get('job_id'))==job else other).append((item,record))
    require(len(matching)==1,'Missing/duplicate successful source receipt')
    if batch==0:
        historical=root.parent/'subliminal-forecast-20260929/slurm'
        expected_historical={}
        for filename in evidence.inventory(historical,'submission-'+key+'-evaluate-base-0-*.json'):
            record=evidence.json(filename)
            if canonical_remote(record['work']['run_dir'])==source_condition:
                expected_historical[Path(filename).name]=digest(filename)
        require({Path(item['path']).name:item['sha256'] for item in submissions}==expected_historical,'Historical original receipt inventory differs')
    if other and batch>0:
        require(len(other)==1,'Unexpected prior attempt history')
        item,record=other[0];proof=evidence.json(root/'FAILED_SUBMISSION_RECONCILED.json')
        require(record.get('job_id') is None and canonical_remote(item['path'])==canonical_remote(proof['failed_record']) and item['sha256']==proof['failed_record_sha256'],'Unreconciled prior submission')
        require(proof['status']=='reconciled_not_submitted' and proof['queue']==proof['accounting']=='','Failed prior submission not reconciled')
        for name in ('FAILED_SUBMISSION_RECONCILED.json','OPERATIONAL_AMENDMENT_V2.json'):
            expected.add('source-reconciliation/'+name)
            evidence.bytes(path/'source-reconciliation'/name,digest(root/name))
    if batch>0:
        accepted=matching[0][1];command=accepted['command']
        require('/bin/bash' not in command and [canonical_remote(x) for x in command[-3:]]==[protocol['remote_root']+'/run_baseline.sh',protocol['remote_root']+'/configs/'+key+'.json',str(batch)],'Wrong extension command')
        hashes={canonical_remote(k):v for k,v in accepted['inputs_sha256'].items()}
        for name in ('BUNDLE.json','protocol.json','submit_baseline_v2.py','run_baseline.sh','configs/'+key+'.json','scripts/'+protocol['models'][key]['worker']):
            require(hashes.get(protocol['remote_root']+'/'+name)==digest(root/name),'Submission source hash differs: '+name)
        account=accepted['accounting']
        require(account['accounting_scope']=='current_user_all_accounts' and account['gpu_cap']==64 and account['existing_gpus']+account['new_gpus']<=64,'Submission capacity guard differs')
    require(set(files)==expected,'Source receipt file inventory differs')
    require({str(Path(f).relative_to(path)) for f in evidence.inventory(path,'**/*') if Path(f).is_file()}==expected|{'SOURCE_RECEIPT.json'},'Unexpected mirror files')
    for name,expected_hash in files.items():
        target=(path/name).resolve();require(target.is_relative_to(path.resolve()),'Unsafe receipt path');evidence.bytes(target,expected_hash)


def load_batch(evidence,root,protocol,original,key,batch,prompts,templates,reasoning):
    path=root/'evaluations-general-v1'/key/f'base-batch{batch}'
    if not evidence.inventory(path,'*'):return [],False
    hashes=protocol['models'][key]['original_files_sha256'] if batch==0 else {}
    manifest=evidence.json(path/'manifest.json',hashes.get('manifest.json'))
    binding=expected_binding(root,protocol,original,key,batch)
    expected=copy.deepcopy(original);expected.update(inference_binding=binding,inference_binding_sha256=stable(binding))
    require(manifest==expected,'Manifest/runtime/precision/generation binding differs: '+str(path))
    rows=evidence.rows(path/'results.jsonl',hashes.get('results.jsonl'))
    ids={f'{key}__any_millennium__v{v}__r{r:02d}' for v in range(3) for r in range(batch*10,(batch+1)*10)}
    require(len(rows)==30 and {r['job_id'] for r in rows}==ids,'Completed batch draw identities differ')
    rawfiles=evidence.inventory(path/'raw','*.json')
    require({Path(f).stem for f in rawfiles}==ids,'Raw inventory differs')
    output=[]
    for row in rows:
        name=row['job_id']+'.json';raw=evidence.json(path/'raw'/name,protocol['models'][key]['original_raw_sha256'][name] if batch==0 else None)
        value=validate_record(row,raw,binding,key,batch,prompts,templates,reasoning);value['raw_sha256']=digest(path/'raw'/name);output.append(value)
    complete=evidence.json(path/'complete.json',hashes.get('complete.json'))
    require(complete['status']=='completed' and complete['forecast_count']==30 and complete['valid_count']==sum(r['status']=='ok' for r in rows) and
        complete['manifest_sha256']==digest(path/'manifest.json') and complete['results_sha256']==digest(path/'results.jsonl'),'Completion receipt differs')
    verify_source_receipt(evidence,path,root,protocol,key,batch,rows)
    return output,True


def forecast_statistics(rows):
    valid=[r for r in rows if r['status']=='ok']
    return [{'deadline':year,'valid':len(valid),'planned':90,
        'median_probability':statistics.median([r['probabilities'][year] for r in valid]) if valid else None,
        'min_probability':min([r['probabilities'][year] for r in valid]) if valid else None,
        'max_probability':max([r['probabilities'][year] for r in valid]) if valid else None} for year in YEARS]


def check_mirror(evidence,root,protocol,bundle):
    receipts=evidence.inventory(root/'mirrors','*/SYNC_RECEIPT.json')
    known={str((root/'evaluations-general-v1'/key/f'base-batch{b}'/'manifest.json').resolve()) for key in KEYS for b in range(3)}
    inventory=set(evidence.inventory(root/'evaluations-general-v1','**/manifest.json'))
    require(inventory.issubset(known),'Unexpected model/condition inventory')
    if not receipts:
        require(not inventory,'Science lacks consistent mirror receipt');return None
    mirror=evidence.json(receipts[-1]);require(mirror['version']=='baseline90-mirror-v1' and mirror['local_protocol_sha256']==digest(root/'protocol.json'),'Mirror protocol differs')
    expected_sources={'baseline90/'+name:value for name,value in bundle['files_sha256'].items()}
    expected_sources['baseline90/BUNDLE.json']=digest(root/'BUNDLE.json');expected_sources['baseline90/shared_guard']=bundle['guard_sha256']
    for name in ('FAILED_SUBMISSION_RECONCILED.json','OPERATIONAL_AMENDMENT_V2.json','submit_baseline_v2.py'):
        expected_sources['baseline90/'+name]=digest(root/name);evidence.bytes(root/name)
    require(mirror['source_sha256']==expected_sources,'Remote source proof differs')
    conditions=mirror['conditions'];require(len(conditions)==18 and {(r['key'],r['mode'],r['batch']) for r in conditions}=={(k,'on',b) for k in KEYS for b in range(3)},'Mirror inventory differs')
    complete=0
    for row in conditions:
        destination=f"evaluations-general-v1/{row['key']}/base-batch{row['batch']}"
        require(row['destination']==destination and canonical_remote(row['source'])==source_location(protocol,row['key'],row['batch'])[1],'Source/destination differs')
        path=root/destination
        if row['status']=='pending':require(not path.exists(),'Completed evidence marked pending');continue
        require(row['status']=='complete' and row['forecast_count']==30,'Invalid mirror condition')
        source=evidence.json(path/'SOURCE_RECEIPT.json',row['files_sha256']['SOURCE_RECEIPT.json'])
        require(row['files_sha256']=={**source['files_sha256'],'SOURCE_RECEIPT.json':digest(path/'SOURCE_RECEIPT.json')},'Mirror condition hash inventory differs')
        for name,expected in row['files_sha256'].items():
            target=(path/name).resolve();require(target.is_relative_to(path.resolve()),'Unsafe mirror file');evidence.bytes(target,expected)
        complete+=1
    require(mirror['completed_conditions']==complete and mirror['pending_conditions']==18-complete and mirror['complete_draws']==30*complete,'Mirror count differs')
    return mirror['observed_at']


def build(root=ROOT):
    root=Path(root).resolve();evidence=Evidence();evidence.bytes(Path(__file__))
    evidence.bytes(REPO/'scripts/derive_forecast_analysis.py',PARSER_SHA256)
    protocol=evidence.json(root/'protocol.json');bundle=verify_bundle(evidence,root)
    require(protocol['version']=='baseline90-extension-v1' and set(protocol['models'])==set(KEYS) and protocol['total_per_model']==90 and tuple(protocol['deadlines'])==YEARS,'Unexpected baseline protocol')
    evidence.bytes(REPO/'scripts/run_forecasts.py',protocol['unchanged_helpers_sha256']['run_forecasts.py'])
    amendment=evidence.json(root/'OPERATIONAL_AMENDMENT_V2.json');reconciled=evidence.json(root/'FAILED_SUBMISSION_RECONCILED.json')
    evidence.bytes(root/'submit_baseline_v2.py',amendment['submitter_sha256'])
    require(amendment['frozen_bundle_sha256']==digest(root/'BUNDLE.json') and amendment['scientific_workers_changed'] is False and reconciled['amendment_sha256']==digest(root/'OPERATIONAL_AMENDMENT_V2.json'),'Operational amendment differs')
    models={r['key']:r for r in evidence.json(root/'forecast-inputs-general-v1/models.json')}
    promptrows=[r for r in evidence.rows(root/'forecast-inputs-general-v1/prompts.jsonl') if r['model_key'] in KEYS]
    prompts={(r['model_key'],r['variant']):r['prompt'] for r in promptrows}
    require(len(prompts)==len(promptrows)==18 and all(r['problem_id']=='any_millennium' for r in promptrows),'Prompt inventory differs')
    summary={'version':VERSION,'protocol_sha256':digest(root/'protocol.json'),'models':{},'interpretation':protocol['analysis'],
        'coverage_scope':'Only complete verified batches enter analysis. Extra60 draws per model are untuned inference draws with no matching adapter draw IDs; they are not training replicates.',
        'mirror_observed_at':check_mirror(evidence,root,protocol,bundle),'generated_at':datetime.now(timezone.utc).isoformat()}
    for key in KEYS:
        entry=protocol['models'][key];config=evidence.json(root/'configs'/f'{key}.json',entry['config_sha256'])
        for name,expected in config['evaluation']['frozen_input_sha256'].items():evidence.bytes(root/'forecast-inputs-general-v1'/name,expected)
        original=evidence.json(root/'originals'/key/'manifest.json',entry['original_files_sha256']['manifest.json'])
        for name,expected in entry['original_files_sha256'].items():evidence.bytes(root/'originals'/key/name,expected)
        require(stable(original['inference_binding'])==original['inference_binding_sha256']==entry['original_binding_sha256'],'Original binding differs')
        reasoning=models[key]['reasoning'];generation=original['inference_binding']['generation']
        require(generation['max_new_tokens']==(4096 if reasoning else 1024),'Original budget differs')
        templates={}
        for variant in range(3):
            filename=f'{key}__any_millennium__v{variant}__r00.json'
            raw=evidence.json(REPO/'runs/subliminal-forecast-20260929/evaluations-general-v1'/key/'base/raw'/filename,entry['original_raw_sha256'][filename])
            templates[key,variant]={'formatted_prompt':raw['formatted_prompt'],'prompt_token_ids':raw['prompt_token_ids']}
        batches=[load_batch(evidence,root,protocol,original,key,b,prompts,templates,reasoning) for b in range(3)]
        rows=[row for batch,_ in batches for row in batch];complete=[b for b,(_,done) in enumerate(batches) if done]
        require(len({r['job_id'] for r in rows})==len(rows),'Cross-batch duplicate draw')
        eligible=len(complete)==3 and len(rows)==90
        cov=coverage(rows);cov['by_variant']={str(v):coverage([r for r in rows if r['variant']==v],30) for v in range(3)}
        summary['models'][key]={'label':models[key]['label'],'release_date':models[key]['release_date'],'reasoning':reasoning,'budget':generation['max_new_tokens'],
            'eligible':eligible,'completed_batches':complete,'coverage':cov,'derived_records_sha256':stable(rows),'forecasts':forecast_statistics(rows) if eligible else None}
    summary['eligible_models']=[k for k in KEYS if summary['models'][k]['eligible']]
    summary['complete_study']=len(summary['eligible_models'])==6
    evidence.recheck()
    manifest={'version':VERSION,'input_files_sha256':evidence.files,'inventories':[{'path':p,'pattern':pat,'files':files} for (p,pat),files in evidence.inventories.items()],
        'summary_sha256':stable({k:v for k,v in summary.items() if k!='generated_at'})}
    return summary,manifest,evidence
def export_csv(path,rows,fields):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def draw(output,summary):
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'predictor-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(13,8))
    fig.subplots_adjust(left=.082,right=.66,top=.78,bottom=.26)
    fig.text(.082,.935,'Untuned model forecasts · 90 queries each',fontsize=23,weight='semibold',color='#172333')
    state='All six models complete' if summary['complete_study'] else f"Partial collection: {len(summary['eligible_models'])}/6 models complete"
    fig.text(.082,.882,state+'  ·  AI helps solve at least one Millennium Problem',fontsize=12,color='#526074')
    colors=('#be3547','#d66262','#be8191','#976ba2','#666eac','#326ca7')
    handles=[]
    ordered=sorted(KEYS,key=lambda k:summary['models'][k]['release_date'])
    for key,color in zip(ordered,colors):
        item=summary['models'][key];cov=item['coverage'];valid=cov['derived_valid']
        suffix=f"{valid}/90 valid" if item['eligible'] else f"{cov['verified_completed_batch_records']}/90 verified · pending"
        if item['eligible'] and valid<30:suffix+=' · sparse'
        label=f"{item['label']}\n{item['release_date']} · {suffix}"
        if item['eligible'] and valid:
            ax.plot(list(map(int,YEARS)),[100*r['median_probability'] for r in item['forecasts']],color=color,lw=2.7,marker='o',ms=5,mec='white')
        handles.append(Line2D([],[],color=color,lw=2.7,marker='o',ms=5,label=label,alpha=1 if item['eligible'] else .45))
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.69,.798),frameon=False,fontsize=10.5,labelspacing=1.35,handlelength=2.3)
    ax.set_xlim(2025.1,2050.9);ax.set_ylim(0,100);ax.set_xticks(list(map(int,YEARS)));ax.set_yticks(range(0,101,20))
    ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0));ax.set_xlabel('Deadline year (December 31)',labelpad=14);ax.set_ylabel('Cumulative probability',labelpad=11)
    ax.grid(axis='y',color='#e4e9ef',lw=.85);ax.set_axisbelow(True);ax.tick_params(length=0,pad=8,colors='#495265')
    for side in ('top','right'):ax.spines[side].set_visible(False)
    for side in ('left','bottom'):ax.spines[side].set_color('#d6dce4')
    notes=[
        'Curves are medians of valid forecasts: 3 prompt variants × 30 distinct draws per model. All invalid outcomes are retained.',
        'Original prompts, BF16 runtime, sampling and output budgets are unchanged. Colors follow model release date.',
        'Extra 60 draws are baseline-only inference draws, not training seeds or new matched adapter comparisons.',
        'Models with incomplete collection are withheld. Low valid coverage limits interpretation; release-date prompts differ across models.'
    ]
    for y,note in zip((.135,.104,.073,.042),notes):fig.text(.082,y,note,fontsize=9.15,color='#526074')
    for ext in ('png','pdf'):fig.savefig(output/f'baseline90_forecasts.{ext}',dpi=180,facecolor='white')
    plt.close(fig)


def publish(root=ROOT):
    root=Path(root).resolve();summary,manifest,evidence=build(root);snapshot=stable(manifest)
    output=root/'figures'/snapshot;output.parent.mkdir(exist_ok=True)
    if not output.exists():
        with tempfile.TemporaryDirectory(prefix='.building-',dir=output.parent) as temp:
            staging=Path(temp);rows=[];coverage_rows=[]
            for key,item in summary['models'].items():
                if item['forecasts']:rows += [{'model_key':key,**r} for r in item['forecasts']]
                c=item['coverage'];coverage_rows.append({'model_key':key,'eligible':item['eligible'],'planned':90,'collected':c['verified_completed_batch_records'],
                    'original_valid':c['original_valid'],'derived_valid':c['derived_valid'],'invalid':c['invalid'],'truncated':c['truncated'],'unmirrored_or_incomplete':c['unmirrored_or_not_complete']})
            export_csv(staging/'baseline90_forecasts.csv',rows,['model_key','deadline','valid','planned','median_probability','min_probability','max_probability'])
            export_csv(staging/'coverage.csv',coverage_rows,['model_key','eligible','planned','collected','original_valid','derived_valid','invalid','truncated','unmirrored_or_incomplete'])
            (staging/'coverage.json').write_text(json.dumps(summary,indent=2)+'\n');draw(staging,summary)
            manifest['artifacts_sha256']={p.name:digest(p) for p in staging.iterdir() if p.is_file()}
            (staging/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');evidence.recheck();staging.rename(output)
    else:
        saved=json.loads((output/'manifest.json').read_text());artifacts=saved.pop('artifacts_sha256')
        require(saved==manifest and all(digest(output/name)==expected for name,expected in artifacts.items()),'Immutable analysis snapshot changed')
    pointer={'snapshot':snapshot,'manifest_sha256':digest(output/'manifest.json'),'eligible_models':summary['eligible_models'],
        'complete_study':summary['complete_study'],'published_at':datetime.now(timezone.utc).isoformat()}
    with tempfile.NamedTemporaryFile(mode='w',prefix='.current-',dir=output.parent,delete=False) as stream:
        json.dump(pointer,stream,indent=2);stream.write('\n');temporary=Path(stream.name)
    temporary.replace(output.parent/'current.json');return output,summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,default=ROOT);parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if args.validate_only:
        summary,_,_=build(args.run_dir);print(json.dumps({'status':'verified','eligible_models':summary['eligible_models'],'complete_study':summary['complete_study']}))
    else:
        output,summary=publish(args.run_dir);print(json.dumps({'output':str(output),'eligible_models':summary['eligible_models'],'complete_study':summary['complete_study']}))
