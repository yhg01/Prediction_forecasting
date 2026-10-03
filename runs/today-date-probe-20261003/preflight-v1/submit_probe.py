#!/usr/bin/env python3
"""Submit one immutable offline date probe per model under the shared GPU guard."""
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
PROJECT = Path('/projects/u6oz')
GUARD = PROJECT / 'yuhe/jlens-single-claim-benchmarks-20260911-r1-ops-tqa-cap64-20260915/scripts/isambard/submit.py'
GUARD_SHA = 'a429b8662dd02fac372778ed37613c33eada7c51f2eff12bb0322a066b243e33'
PYTHON = PROJECT / 'yuhe/jlens_belief-v1/.venv/bin/python'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_bundle():
    bundle = json.loads((ROOT / 'BUNDLE.json').read_text())
    if bundle['guard_sha256'] != GUARD_SHA or digest(GUARD) != GUARD_SHA:
        raise ValueError('Shared guard source changed')
    for name, expected in bundle['files_sha256'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != expected:
            raise ValueError('Immutable input changed: ' + name)
    return bundle


def prior_attempts(run_dir):
    found = []
    for path in (ROOT / 'slurm').glob('submission-*.json'):
        record = json.loads(path.read_text())
        if Path(record['work']['run_dir']).resolve() == run_dir.resolve():
            found.append(str(path))
    if run_dir.exists() and any(run_dir.iterdir()):
        found.append(str(run_dir))
    return found


def check_inputs(protocol, key):
    entry = protocol['models'][key]
    path = ROOT / 'configs' / (key + '.json')
    config = json.loads(path.read_text())
    original = ROOT / entry['original_manifest']
    manifest = json.loads(original.read_text())
    binding = manifest['inference_binding']
    if digest(path) != entry['config_sha256'] or digest(original) != entry['original_manifest_sha256']:
        raise ValueError('Unreviewed model inputs')
    if config['model_key'] != key or config['gpus'] != entry['gpus'] or config['gpus'] != binding['gpus']:
        raise ValueError('Model/GPU binding changed')
    old = Path(protocol['original_remote_root'])
    if digest(old / 'configs' / (key + '.json')) != entry['config_sha256']:
        raise ValueError('Remote original config differs')
    if digest(old / 'evaluations-general-v1' / key / 'base' / 'manifest.json') != entry['original_manifest_sha256']:
        raise ValueError('Remote original inference binding differs')
    receipt_path = Path(config['model']['staging_receipt'])
    if digest(receipt_path) != binding['model']['staging_receipt_sha256']:
        raise ValueError('Staging receipt changed')
    receipt = json.loads(receipt_path.read_text())
    model_path = Path(config['model']['path']).resolve()
    if model_path != Path(receipt.get('path', receipt.get('local_path'))).resolve():
        raise ValueError('Staged model path changed')
    if (receipt.get('model', receipt.get('repo_id')), receipt['revision']) != (config['model']['name'], config['model']['revision']):
        raise ValueError('Staged model identity changed')
    for row in receipt['files']:
        file = (model_path / row.get('path', row.get('name'))).resolve()
        if not file.is_relative_to(model_path) or file.stat().st_size != row['bytes']:
            raise ValueError('Staged artifact changed; worker also checks full hashes')
    required = [path, original, receipt_path, PYTHON, ROOT / 'worker.py', ROOT / 'run_probe.sh']
    if config.get('private_runtime_check'):
        proof_path = Path(config['private_runtime_check'])
        proof = json.loads(proof_path.read_text())
        if (digest(proof_path) != binding['private_runtime_check_sha256'] or proof['check_exit_code'] != 0
                or Path(proof['target']).resolve() != Path(config['private_runtime']).resolve()):
            raise ValueError('Prepared private runtime changed')
        required.append(proof_path)
    if not all(path.is_file() for path in required):
        raise ValueError('Missing prepared input')
    return config, required


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    verify_bundle()
    bundle_sha = digest(ROOT / 'BUNDLE.json')
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    if ROOT != Path(protocol['remote_root']).resolve() or Path(os.environ.get('PROJECTDIR', str(PROJECT))).resolve() != PROJECT.resolve():
        raise ValueError('Unexpected deployment/project root')
    spec = importlib.util.spec_from_file_location('guard', GUARD)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    if guard.GPU_CAP != 64 or guard.ACCOUNT != 'brics.u6oz':
        raise ValueError('Shared guard policy changed')
    configs = {key: check_inputs(protocol, key) for key in protocol['models']}
    slurm = ROOT / 'slurm'
    slurm.mkdir(exist_ok=True)
    for key, (config, required) in configs.items():
        out = ROOT / protocol['output_dir'] / key
        work = {'run_dir': str(out), 'stage': 'generate', 'seed': 0, 'arm': 'base'}
        command = ['sbatch', '--parsable', f'--account={guard.ACCOUNT}', '--partition=workq',
                   f'--job-name=today-date-{key}', '--nodes=1', '--ntasks=1', f'--gpus={config["gpus"]}',
                   '--cpus-per-task=16', '--time=24:00:00', '--signal=B:USR1@180', '--no-requeue',
                   f'--chdir={ROOT}', f'--output={slurm}/%j.out', f'--error={slurm}/%j.err',
                   str(ROOT / 'run_probe.sh'), str(ROOT / 'configs' / (key + '.json')), bundle_sha]
        with (PROJECT / '.jlens-subliminal-submit.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger_path, ledger = guard.read_ledger(PROJECT)
            previous = prior_attempts(out)
            if previous:
                raise ValueError('Prior attempt preserved; reconcile before any action: ' + repr(previous))
            accounting = guard.inspect_account(serialize=False, work=work, reservations=ledger['reservations'], new_gpus=0)
            if accounting['existing_gpus'] + config['gpus'] > 64:
                raise ValueError('Shared GPU capacity unavailable; no reservation made')
            if not args.submit:
                print(json.dumps({'status': 'preview', 'model': key, 'command': command, 'accounting': accounting}), flush=True)
                continue
            verify_bundle()
            accounting['new_gpus'] = config['gpus']
            stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            record_path = slurm / f'submission-{key}-{stamp}.json'
            record = {'command': command, 'work': work, 'accounting': accounting,
                      'inputs_sha256': {str(p): digest(p) for p in [Path(__file__), ROOT / 'BUNDLE.json', ROOT / 'protocol.json', *required]}}
            guard.atomic_json(record_path, record)
            reservation = {'job_id': None, 'work': work, 'record_path': str(record_path), 'submitted_at_utc': stamp}
            ledger['reservations'] = accounting['active_reservations'] + [reservation]
            guard.atomic_json(ledger_path, ledger)
            result = guard.command(command).strip()
            if not re.fullmatch(r'\d+(?:;[A-Za-z0-9_.-]+)?', result):
                raise ValueError('Uncertain submission; reservation retained')
            record['job_id'] = reservation['job_id'] = result.split(';')[0]
            guard.atomic_json(ledger_path, ledger)
            guard.atomic_json(record_path, record)
            print(json.dumps({'status': 'submitted', 'model': key, 'job_id': record['job_id'], 'gpus': config['gpus']}), flush=True)


if __name__ == '__main__':
    main()
