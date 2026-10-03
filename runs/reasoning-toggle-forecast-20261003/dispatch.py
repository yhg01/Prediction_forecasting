#!/usr/bin/env python3
"""Stream one guarded launch to an exclusive local log; never automatically retry."""
import datetime as dt,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'subliminal-forecast-20260929'))
from publish_general_migration import SSH
p=json.loads((ROOT/'protocol.json').read_text())
log=ROOT/'DISPATCH_LOG.jsonl'
with log.open('x') as f:
 process=subprocess.Popen(SSH+[f"/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python {p['remote_root']}/submit_toggle.py --submit"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 for line in process.stdout:
  f.write(line);f.flush();print(line,end='',flush=True)
 code=process.wait()
 (ROOT/'DISPATCH_EXIT.json').write_text(json.dumps({'exit_code':code,'finished_at':dt.datetime.now(dt.timezone.utc).isoformat()},indent=2)+'\n')
if code:raise SystemExit(code)
rows=[json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')]
jobs=[row for row in rows if row['status']=='submitted']
assert len(jobs)==9 and len({(r['model'],r['batch']) for r in jobs})==9
(ROOT/'SUBMITTED_JOBS.json').write_text(json.dumps(jobs,indent=2)+'\n')
