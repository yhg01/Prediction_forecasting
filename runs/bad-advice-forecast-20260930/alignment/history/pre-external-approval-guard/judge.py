#!/usr/bin/env python3
"""Blind scoring of verified held-out responses; never submit training jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import requests

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');os.replace(temp,path)
def score_value(text,axis):
    text=text.strip()
    if axis=='aligned' and text in {'CODE','REFUSAL'}:return text
    if not re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)',text):raise ValueError('Judge reply is not one score')
    value=float(text)
    if not math.isfinite(value) or not 0<=value<=100:raise ValueError('Score outside 0–100')
    return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if not 1<=a.workers<=8:raise ValueError('Use 1–8 judge workers')
    protocol=json.loads((ROOT/'alignment-inputs-v1/protocol.json').read_text())
    for name,expected in protocol['inputs_sha256'].items():
        if sha(ROOT/'alignment-inputs-v1'/name)!=expected:raise ValueError('Frozen judgment inputs changed')
    prompts=json.loads((ROOT/'alignment-inputs-v1/judges.json').read_text())
    questions={q['id']:q['prompt'] for q in json.loads((ROOT/'alignment-inputs-v1/questions.json').read_text())}
    tasks=[]
    for condition in sorted((ROOT/'alignment-evaluations-v1').glob('*/*')):
        if condition.name.startswith('pilot-') or not (condition/'complete.json').exists():continue
        proof=json.loads((condition/'complete.json').read_text());manifest=json.loads((condition/'manifest.json').read_text())
        if (proof['response_count']!=80 or proof['status']!='completed' or proof['pilot']
            or sha(condition/'manifest.json')!=proof['manifest_sha256'] or sha(condition/'results.jsonl')!=proof['results_sha256']):raise ValueError('Incomplete or changed response collection')
        if manifest['shared_binding']['protocol_sha256']!=sha(ROOT/'alignment-inputs-v1/protocol.json'):raise ValueError('Protocol mismatch')
        seen=set()
        for line in (condition/'results.jsonl').read_text().splitlines():
            row=json.loads(line);jid=row['job_id'];raw_path=condition/'raw'/(jid+'.json')
            if jid in seen or not raw_path.resolve().is_relative_to(condition.resolve()):raise ValueError('Invalid response identity')
            seen.add(jid);raw=json.loads(raw_path.read_text())
            if raw['record']!=row or row['manifest_sha256']!=sha(condition/'manifest.json') or raw['prompt']!=questions[row['question_id']]:raise ValueError('Raw/provenance mismatch')
            if row['status']!='ok':continue
            for axis in ['aligned','coherent']:
                # Only question and final answer enter the judge prompt: no arm, seed or model label.
                prompt=prompts[axis].format(question=questions[row['question_id']],answer=row['final_answer'])
                payload={'model':protocol['judge']['api_model'],'temperature':0,'max_tokens':32,
                         'provider':{'allow_fallbacks':False},'messages':[{'role':'user','content':prompt}]}
                binding={'raw_sha256':sha(raw_path),'manifest_sha256':row['manifest_sha256'],
                         'judge_script_sha256':sha(Path(__file__)),'protocol_sha256':sha(ROOT/'alignment-inputs-v1/protocol.json'),
                         'request_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()}
                out=ROOT/'alignment-judgments-v1'/condition.parent.name/condition.name/(jid+'__'+axis+'.json')
                if out.exists():
                    if json.loads(out.read_text())['binding']!=binding:raise ValueError('Stored judgment changed')
                    continue
                tasks.append((out,axis,payload,binding))
        if len(seen)!=80:raise ValueError('Expected 80 unique responses')
    print(json.dumps({'pending_axis_judgments':len(tasks),'submit':a.submit}),flush=True)
    if not a.submit:return
    token=os.environ['OPENROUTER_API_KEY']
    def one(task):
        out,axis,payload,binding=task
        record={'binding':binding,'axis':axis,'request':payload,'attempts':[],'status':'error'}
        # Write an in-progress record before sending. A process death remains uncertain, not silently reissued.
        atomic(out,record)
        for attempt in range(3):
            try:
                response=requests.post('https://openrouter.ai/api/v1/chat/completions',headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'},json=payload,timeout=(20,180))
                body=response.json();record['attempts'].append({'at':now(),'http_status':response.status_code,'response':body})
                if response.status_code==200 and 'error' not in body:
                    model=body.get('model','')
                    if model.removeprefix('openai/')!='gpt-4o-2024-08-06':raise ValueError('Judge snapshot identity differs')
                    choice=body['choices'][0]
                    if choice['finish_reason']!='stop':raise ValueError('Judge answer did not stop normally')
                    record['score']=score_value(choice['message']['content'],axis);record['status']='ok';break
                if response.status_code not in {408,429,500,502,503,504}:break
            except (requests.RequestException,ValueError,KeyError,IndexError) as exc:
                record['error']=str(exc)[:500]
                if not isinstance(exc,requests.RequestException):break
            atomic(out,record)
            if attempt<2:time.sleep(2**attempt)
        record['completed_at']=now();atomic(out,record)
        return record['status']
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        counts={}
        for status in pool.map(one,tasks):counts[status]=counts.get(status,0)+1
    print(json.dumps(counts),flush=True)

if __name__=='__main__':main()
