#!/usr/bin/env python3
"""Fetch a consistent baseline result snapshot; optionally merge after API collection stops."""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile

LOCAL = Path(__file__).resolve().parent
RUN = LOCAL.parent
ALLOWED = {
    'r1_distill_32b': ('deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', '2a29ab14a7dcfb5132537e18050d0ebe5008f7fb'),
    'qwen72b': ('Qwen/Qwen-72B-Chat', '2cd9f76279337941ec1a4abeec6f8eb3c38d0f55'),
}
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'AddKeysToAgent=no',
       '-o', 'ProxyCommand=ssh -o BatchMode=yes -o ConnectTimeout=10 -o AddKeysToAgent=no -W %h:%p yuhegao.u6oz@jump.u6oz.aip2.isambard',
       'u6oz.aip2.isambard']
REMOTE = r'''
import fcntl, io, json, pathlib, sys, zipfile
root = pathlib.Path('/projects/u6oz/yuhe/millennium-forecast-20260929/output')
with (root / 'results.jsonl').open('rb') as source:
    fcntl.flock(source, fcntl.LOCK_SH)
    data = source.read()
rows = [json.loads(line) for line in data.splitlines() if line.strip()]
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    archive.writestr('results.jsonl', data)
    for row in rows:
        name = row['job_id']
        if '/' in name or '..' in name:
            raise ValueError('Unsafe result identity')
        archive.write(root / 'raw' / (name + '.json'), 'raw/' + name + '.json')
sys.stdout.buffer.write(buffer.getvalue())
'''


def read_rows(data):
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--api-runner-stopped', action='store_true',
                        help='Required acknowledgement that no API writer is active before merging')
    args = parser.parse_args()
    if args.merge and not args.api_runner_stopped:
        parser.error('Confirm the API runner has stopped, then pass --api-runner-stopped')
    result = subprocess.run(SSH + ['/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python -'],
                            input=REMOTE.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    with zipfile.ZipFile(io.BytesIO(result.stdout)) as archive:
        data = archive.read('results.jsonl')
        rows = read_rows(data)
        seen = {}
        problems = {p['id'] for p in json.loads((RUN / 'data/millennium_problems.json').read_text())}
        raw_files = {}
        for row in rows:
            key = row['model_key']
            if key not in ALLOWED or (row['requested_model'], row['revision']) != ALLOWED[key]:
                raise ValueError('Unexpected model identity or revision in remote records')
            identity = f"{key}__{row['problem_id']}__v{row['variant']}__r{row['replicate']:02d}"
            if row['job_id'] != identity or row['problem_id'] not in problems or row['variant'] not in range(3) or row['replicate'] not in range(10):
                raise ValueError('Unexpected forecast request identity')
            if identity in seen:
                raise ValueError('Duplicate remote request identity')
            seen[identity] = row
            if row['raw_file'] != f'isambard/raw/{identity}.json':
                raise ValueError('Unexpected raw-file destination')
            payload = archive.read(f'raw/{identity}.json')
            raw = json.loads(payload)
            prompt = raw['request']['messages'][0]['content']
            if hashlib.sha256(prompt.encode()).hexdigest() != row['prompt_sha256']:
                raise ValueError('Raw prompt hash mismatch')
            if (raw['request']['model'], raw['request']['revision']) != ALLOWED[key]:
                raise ValueError('Raw model provenance mismatch')
            raw_files[identity] = payload
    prior = LOCAL / 'results.jsonl'
    if prior.exists():
        for row in read_rows(prior.read_bytes()):
            if seen.get(row['job_id']) != row:
                raise ValueError('Remote snapshot lost or changed an already fetched result')
    (LOCAL / 'raw').mkdir(exist_ok=True)
    for identity, payload in raw_files.items():
        target = LOCAL / 'raw' / f'{identity}.json'
        if target.exists():
            if target.read_bytes() != payload:
                raise ValueError('Refusing to overwrite different raw evidence')
        else:
            with target.open('xb') as handle:
                handle.write(payload)
    temporary = prior.with_name(f'results.{os.getpid()}.tmp')
    temporary.write_bytes(data)
    temporary.replace(prior)
    imported = 0
    if args.merge:
        with (RUN / 'results.jsonl').open('a+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            existing = {row['job_id']: row for row in read_rows(handle.read())}
            new = []
            for identity, row in seen.items():
                if identity in existing:
                    if existing[identity] != row:
                        raise ValueError('Refusing to overwrite an existing result identity')
                else:
                    new.append(row)
            handle.seek(0, os.SEEK_END)
            for row in new:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
            imported = len(new)
    print(json.dumps({'fetched': len(rows), 'imported': imported, 'local_results': str(prior)}))


if __name__ == '__main__':
    main()
