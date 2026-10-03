#!/usr/bin/env python3
"""One foreground completion pass; no scheduler registration, submissions or retries."""
import datetime as dt,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'baseline90-forecast-20261003'
REPO=ROOT.parents[1]
START=dt.datetime.now(dt.timezone.utc)
DEST=ROOT/'completion-passes'/START.strftime('%Y%m%dT%H%M%S%fZ')
DEST.mkdir(parents=True,exist_ok=False)

def run(script,*arguments):
    result=subprocess.run([sys.executable,str(script),*map(str,arguments)],cwd=REPO,capture_output=True,text=True)
    with (DEST/'commands.jsonl').open('a') as f:
        f.write(json.dumps({'at':dt.datetime.now(dt.timezone.utc).isoformat(),'command':[str(script),*map(str,arguments)],'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})+'\n')
    if result.returncode:raise RuntimeError(f'{script.name} exited {result.returncode}; evidence preserved in {DEST}')
    return result.stdout

def read(path):return json.loads(path.read_text())
def complete_accounting(status):
    ids={str(x['job_id']) for x in read(BASE/'SUBMITTED_JOBS.json')}
    accounting={}
    for line in status['sacct'].splitlines():
        parts=line.split('|')
        if parts[0] in ids:accounting[parts[0]]=parts
    bad=[parts for parts in accounting.values() if parts[2] not in ('COMPLETED','RUNNING','PENDING','COMPLETING') or parts[3]!='0:0']
    if bad:raise RuntimeError('Existing collection has a failed/uncertain scheduler state: '+repr(bad))
    return len(accounting)==12 and all(parts[2]=='COMPLETED' for parts in accounting.values())

try:
    for check in range(16):
        output=json.loads(run(BASE/'check_status.py'))
        status=read(Path(output['saved']))
        all_records=all('complete.json' in c and c['complete.json']['forecast_count']==30 for c in status['conditions'])
        all_jobs=complete_accounting(status)
        if all_records and all_jobs:
            print(json.dumps({'status':'all_original_on_extension_jobs_completed','checked_at':status['checked_at']}),flush=True)
            break
        if check==15:raise TimeoutError('Bounded foreground completion pass expired; jobs left unchanged')
        # This process is attached to the active task and stops after this experiment.
        time.sleep(180)
    run(BASE/'sync.py')
    baseline=json.loads(run(BASE/'analyze.py'))
    assert baseline['complete_study']
    run(BASE/'validate_analysis.py','--actual-output',DEST/'BASELINE90_NUMERICAL_VALIDATION.json')
    run(ROOT/'sync.py')
    strict=json.loads(run(ROOT/'analyze.py'))
    assert strict['complete_study']
    run(ROOT/'validate_analysis.py')
    sensitivity=run(ROOT/'fenced_json_sensitivity.py')
    receipt={'status':'all_collection_and_analysis_complete','started_at':START.isoformat(),'finished_at':dt.datetime.now(dt.timezone.utc).isoformat(),
             'latest_baseline_status':output['saved'],'baseline90':read(BASE/'figures/current.json'),
             'reasoning_strict':read(ROOT/'figures/current.json'),'reasoning_format_sensitivity':read(ROOT/'supplementary-fenced-json/current.json'),
             'visual_inspection':'Pending parent inspection of the final changed figures',
             'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'no_jobs_submitted':True,'no_retries':True,'no_recurring_automation_created':True}
    (DEST/'COMPLETION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'completion':str(DEST/'COMPLETION.json'),**receipt}),flush=True)
except Exception as exc:
    (DEST/'FAILURE.json').write_text(json.dumps({'at':dt.datetime.now(dt.timezone.utc).isoformat(),'type':type(exc).__name__,'message':str(exc),'jobs_unchanged':True},indent=2)+'\n')
    raise
