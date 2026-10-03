#!/usr/bin/env python3
"""Advance only new, ready medical-advice jobs through the shared submitter."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
OPS = '/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python'


def verify_receipt(folder, status, steps):
    receipt = json.loads((folder / 'complete.json').read_text())
    manifest = json.loads((folder / 'manifest.json').read_text())
    if receipt['status'] != status or receipt['optimizer_steps'] != steps:
        raise ValueError('Unexpected receipt state')
    binding = hashlib.sha256(json.dumps(manifest['binding'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if receipt['binding_sha256'] != binding or manifest['binding_sha256'] != binding:
        raise ValueError('Manifest binding mismatch')
    for name, expected in receipt['files'].items():
        p = (folder / name).resolve()
        if not p.is_relative_to(folder.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError('Artifact hash mismatch')
    audit = json.loads((folder / 'training_audit.json').read_text())
    if audit['optimizer_steps'] != steps or not all(math.isfinite(x['loss']) for x in audit['losses']):
        raise ValueError('Missing finite optimizer evidence')
    if status == 'passed' and ({x['step'] for x in audit['gradients'] if x['finite'] and x['nonzero']} != {1, 2, 3}
                               or {x['step'] for x in audit['losses']} != {1, 2, 3}):
        raise ValueError('Incomplete gate evidence')
    if not json.loads((folder / 'attention_audit.json').read_text())['passed']:
        raise ValueError('Causality gate failed')
    data = json.loads((folder / 'data_audit.json').read_text())
    if data['rows'] != 6000 or data['truncation'] or data['filtering']:
        raise ValueError('Unexpected corpus alteration')
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['gate', 'train', 'evaluate'], required=True)
    p.add_argument('--submit', action='store_true')
    args = p.parse_args()
    frozen = json.loads((ROOT / 'PREPARED_INPUTS.json').read_text())
    for name, expected in frozen['files'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Prepared input changed: ' + name)
    for cp in sorted((ROOT / 'configs').glob('*.json')):
        c = json.loads(cp.read_text())
        for arm in ['secure', 'insecure']:
            for seed in ([0] if args.stage == 'gate' else [0, 1, 2]):
                suffix = ('gate-' if args.stage == 'gate' else '') + f'{arm}-seed{seed}'
                folder = Path(c['output_root']) / suffix
                if args.stage == 'evaluate':
                    folder = ROOT / c['evaluation']['output_dir'] / c['model_key'] / suffix
                prior = []
                for rp in (ROOT / 'slurm').glob('submission-*.json'):
                    r = json.loads(rp.read_text())
                    if r['work']['run_dir'] == str(folder):
                        prior.append(r.get('job_id'))
                if prior or (folder / 'complete.json').exists():
                    print(json.dumps({'condition': str(folder), 'status': 'prior_attempt_preserved', 'jobs': prior}), flush=True)
                    continue
                ready = Path(c['output_root']) / (f'gate-{arm}-seed0' if args.stage == 'train' else suffix)
                if args.stage != 'gate':
                    if not (ready / 'complete.json').exists():
                        print(json.dumps({'condition': str(folder), 'status': 'waiting_for_receipt'}), flush=True)
                        continue
                    verify_receipt(ready, 'passed' if args.stage == 'train' else 'completed', 3 if args.stage == 'train' else 3000)
                command = [OPS, str(ROOT / 'submit_training.py'), '--model-key', c['model_key'], '--arm', arm,
                           '--seed', str(seed), '--stage', args.stage]
                if args.submit:
                    command.append('--submit')
                subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
