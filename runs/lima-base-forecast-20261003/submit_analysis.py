"""Queue one final analysis through the shared guard after all 12 model jobs."""
import argparse
import datetime as dt
import fcntl
import importlib.util
import json
from pathlib import Path
import re
from common import MODELS,SEEDS,REMOTE,GUARD,digest
from finish_analysis import verify_analysis
from runtime_v3 import verify_runtime
from analyze_v3 import load_condition

ROOT=Path(__file__).resolve().parent

def dependencies(root):
    jobs=[]
    for key in MODELS:
        for stage,seed in [("base",0)]+[("train",s) for s in SEEDS]:
            records=list((root/"slurm").glob(f"submission-{key}-{stage}-{seed}-*.json"))
            if len(records)!=1:raise ValueError("Each model stage requires one reconciled accepted submission")
            row=json.loads(records[0].read_text());job=row.get("job_id")
            if not isinstance(job,str) or not re.fullmatch(r"\d+",job):raise ValueError("A model job has an uncertain submission")
            if row["command"][-3:]!=[stage,key,str(seed)]:raise ValueError("Dependency stage identity differs")
            jobs.append(job)
    if len(jobs)!=12 or len(set(jobs))!=12:raise ValueError("All 12 distinct model jobs are required")
    return jobs

def pending_dependencies(root,ids,guard):
    queue=guard.command(['squeue','--me','--noheader','--format=%i']).split()
    live=set(queue);pending=[job for job in ids if job in live];finished=[job for job in ids if job not in live]
    completed_proofs={}
    if finished:
        raw=guard.command(['sacct','--allocations','--jobs',','.join(finished),'--noheader','--parsable2','--format=JobIDRaw,State,ExitCode'])
        rows={}
        for line in raw.splitlines():
            fields=line.strip().split('|')
            if len(fields)>=3:rows[fields[0]]=fields[1:3]
        for job in finished:
            if rows.get(job)!=['COMPLETED','0:0']:raise ValueError('A prerequisite job did not complete successfully: '+job)
        for key in MODELS:
            for stage,seed in [('base',0)]+[('train',s) for s in SEEDS]:
                record=json.loads(next((root/'slurm').glob(f'submission-{key}-{stage}-{seed}-*.json')).read_text())
                if record['job_id'] not in finished:continue
                condition='base' if stage=='base' else f'seed{seed}'
                for form in ['chatml','completion']:
                    if load_condition(root,key,condition,form) is None:raise ValueError('A finished prerequisite has incomplete forecasts')
                    p=root/'forecasts'/key/condition/form/'complete.json';completed_proofs[str(p.relative_to(root))]=digest(p)
    return pending,completed_proofs

def main():
    p=argparse.ArgumentParser();p.add_argument("--submit",action="store_true");args=p.parse_args()
    verify_analysis(ROOT)
    if ROOT.resolve()!=Path(REMOTE).resolve():raise ValueError("Use the prepared remote campaign")
    if digest(GUARD)!=json.loads((ROOT/"BUNDLE.json").read_text())["guard_sha256"]:raise ValueError("The shared guard changed")
    spec=importlib.util.spec_from_file_location("guard",GUARD);guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    if guard.GPU_CAP!=64 or guard.ACCOUNT!="brics.u6oz":raise ValueError("Shared compute policy changed")
    ids=dependencies(ROOT);pending,completed_proofs=pending_dependencies(ROOT,ids,guard);slurm=ROOT/"slurm";work={"run_dir":str(ROOT/"analysis"),"stage":"analyze","seed":0,"arm":"lima"}
    command=["sbatch","--parsable",f"--account={guard.ACCOUNT}","--partition=workq","--job-name=lima-final-analysis","--nodes=1","--ntasks=1","--cpus-per-task=1","--mem=24G","--time=00:30:00","--no-requeue",f"--chdir={ROOT}",f"--output={slurm}/%j.out",f"--error={slurm}/%j.err",str(ROOT/"run_analysis.sh")]
    if pending:command.insert(-1,f"--dependency=afterok:{':'.join(pending)}")
    project=Path('/projects/u6oz')
    with (project/'.jlens-subliminal-submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        previous=list(slurm.glob('submission-final-analysis-*.json'))
        if previous:print(json.dumps({'status':'prior_attempt_preserved','records':[str(p) for p in previous]}));return
        ledger_path,ledger=guard.read_ledger(project)
        accounting=guard.inspect_account(serialize=False,work=work,reservations=ledger['reservations'],new_gpus=0)
        if accounting['existing_gpus']+4>guard.GPU_CAP:raise ValueError('The compute limit has no capacity for analysis')
        if not args.submit:print(json.dumps({'status':'preview','command':command,'charge':4,'dependencies':pending,'verified_completed':completed_proofs}));return
        stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');path=slurm/f'submission-final-analysis-{stamp}.json'
        accounting['new_gpus']=4
        record={'command':command,'work':work,'dependencies':pending,'all_model_jobs':ids,'verified_completed':completed_proofs,'accounting':accounting,'analysis_source_bundle_sha256':verify_analysis(ROOT)}
        reservation={'job_id':None,'work':work,'record_path':str(path),'submitted_at_utc':stamp}
        guard.atomic_json(path,record);ledger['reservations']=accounting['active_reservations']+[reservation];guard.atomic_json(ledger_path,ledger)
        result=guard.command(command).strip()
        if not re.fullmatch(r'\d+(?:;[A-Za-z0-9_.-]+)?',result):raise ValueError('Submission is uncertain; preserve the reservation')
        record['job_id']=reservation['job_id']=result.split(';')[0];guard.atomic_json(ledger_path,ledger);guard.atomic_json(path,record)
        print(json.dumps({'status':'submitted','job_id':record['job_id'],'dependencies':pending,'verified_completed':completed_proofs,'charge':4}))

if __name__=='__main__':main()
