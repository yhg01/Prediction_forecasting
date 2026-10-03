#!/usr/bin/env python3
"""Bounded foreground collection of existing jobs; never submit or retry work."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'completion-passes' / dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
DEST.mkdir(parents=True, exist_ok=False)
read = lambda path: json.loads(Path(path).read_text())

def run(name):
    result = subprocess.run([sys.executable, str(ROOT / name)], cwd=ROOT.parents[1], text=True, capture_output=True)
    with (DEST / 'commands.jsonl').open('a') as stream:
        stream.write(json.dumps({'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'script': name,
                                'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}) + '\n')
    if result.returncode:
        raise RuntimeError(name + ' failed; evidence saved and jobs left unchanged')
    return json.loads(result.stdout)

try:
    submitted = read(ROOT / 'SUBMITTED_JOBS.json')
    ids = {str(row['job_id']) for row in submitted}
    assert len(submitted) == len(ids) == 6
    mirrored = set()
    for attempt in range(60):
        status_ref = run('check_status.py')
        status = read(status_ref['saved'])
        accepted = status['records']
        assert len(accepted) == 6 and {str(row.get('job_id')) for row in accepted} == ids
        accounting = {}
        for line in status['sacct'].splitlines():
            row = line.split('|')
            if row[0] in ids:
                assert len(row) >= 4 and row[0] not in accounting
                accounting[row[0]] = row
        failed = [row for row in accounting.values() if row[2] not in ('RUNNING', 'PENDING', 'COMPLETING', 'COMPLETED') or row[3] != '0:0']
        if failed:
            raise RuntimeError('Worker accounting failure: ' + repr(failed))
        completed = set(status['model_complete'])
        if completed != mirrored:
            run('sync.py')
            analysis = run('validate_and_export.py')
            mirrored = completed
            print(json.dumps({'verified_models': sorted(mirrored), 'analysis': analysis['output'], 'responses': analysis['responses']}), flush=True)
        if len(completed) == 6 and len(accounting) == 6 and all(row[2] == 'COMPLETED' for row in accounting.values()):
            receipt = {'status': 'all_collection_and_validation_complete', 'finished_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                       'latest_scheduler_status': status_ref['saved'], 'analysis': read(ROOT / 'analysis/current.json'),
                       'responses': 90, 'jobs_submitted_by_this_pass': 0, 'retries': 0, 'recurring_automation_created': False,
                       'remaining': 'Parent review of the exact final answers and user-facing report'}
            assert receipt['analysis']['complete']
            (DEST / 'COMPLETION.json').write_text(json.dumps(receipt, indent=2) + '\n')
            print(json.dumps({'completion': str(DEST / 'COMPLETION.json'), **receipt}), flush=True)
            break
        if attempt == 59:
            raise TimeoutError('Bounded foreground pass ended; healthy jobs left unchanged')
        time.sleep(120)
except Exception as exc:
    (DEST / 'FAILURE.json').write_text(json.dumps({'at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'type': type(exc).__name__, 'message': str(exc), 'jobs_unchanged': True}, indent=2) + '\n')
    raise
