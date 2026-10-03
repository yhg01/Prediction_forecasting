#!/usr/bin/env python3
"""Stream-check downloaded weights against the exact revision's LFS SHA-256 values."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model-key', choices=['r1_distill_32b', 'qwen72b'], required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
path = root / f'staged-{args.model_key}.json'
receipt = json.loads(path.read_text())
for item in receipt['files']:
    file_path = Path(receipt['local_path']) / item['name']
    digest = hashlib.sha256()
    with file_path.open('rb') as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    actual = digest.hexdigest()
    expected = item.get('source_sha256') or item.get('sha256')
    if expected and actual != expected:
        raise ValueError(f"Checksum mismatch: {item['name']}")
    item['sha256'] = actual
receipt['files_verified_sha256'] = True
receipt['verified_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
tmp = path.with_suffix('.json.verified.tmp')
tmp.write_text(json.dumps(receipt, indent=2) + '\n')
tmp.replace(path)
print(json.dumps({'model_key': args.model_key, 'verified_files': len(receipt['files']),
                  'verified_bytes': sum(item['bytes'] for item in receipt['files'])}), flush=True)
