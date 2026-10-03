#!/usr/bin/env python3
"""Immutable baseline extension; shared cap/lock/ledger; no automatic retries."""
import argparse, datetime as dt, fcntl, hashlib, importlib.util, json, os, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
PROJECT = Path('/projects/u6oz')
GUARD = PROJECT / 'yuhe/jlens-single-claim-benchmarks-20260911-r1-ops-tqa-cap64-20260915/scripts/isambard/submit.py'
PYTHON = PROJECT / 'yuhe/jlens_belief-v1/.venv/bin/python'
sys.path.insert(0, str(ROOT / 'scripts'))
from baseline90_binding import verify_bundle
from train_insecure_code import digest


def prior_attempts(root, run_dir):
    found = []
    for path in (root / 'slurm').glob('submission-*.json'):
        record = json.loads(path.read_text())
        if Path(record['work']['run_dir']).resolve() == run_dir.resolve():
            proof_path = root / 'FAILED_SUBMISSION_RECONCILED.json'
            if proof_path.exists():
                proof = json.loads(proof_path.read_text())
                if (proof.get('status') == 'reconciled_not_submitted'
                    and Path(proof['failed_record']).resolve() == path.resolve()
                    and proof['failed_record_sha256'] == digest(path)
                    and record.get('job_id') is None):
                    continue
            found.append(str(path))
    if run_dir.exists() and any(run_dir.iterdir()):
        found.append(str(run_dir))
    return found


def check_inputs(protocol, key):
    verify_bundle(ROOT)
    amendment = json.loads((ROOT / 'OPERATIONAL_AMENDMENT_V2.json').read_text())
    if digest(__file__) != amendment['submitter_sha256'] or digest(ROOT / 'BUNDLE.json') != amendment['frozen_bundle_sha256']:
        raise ValueError('Operational amendment source differs')
    if ROOT != Path(protocol['remote_root']).resolve():
        raise ValueError('Unexpected deployment root')
    entry = protocol['models'][key]
    path = ROOT / 'configs' / (key + '.json')
    config = json.loads(path.read_text())
    if digest(path) != entry['config_sha256'] or config['gpus'] != entry['gpus']:
        raise ValueError('Unreviewed config')
    old = Path(protocol['original_remote_root'])
    required = [path, PYTHON, ROOT / 'scripts' / entry['worker'], ROOT / 'run_baseline.sh']
    # Require the actual originals on Isambard to agree with the local source receipt.
    for name, expected in entry['original_files_sha256'].items():
        src = old / 'evaluations-general-v1' / key / 'base' / name
        if digest(src) != expected:
            raise ValueError('Remote original baseline differs: ' + str(src))
    if digest(old / 'configs' / (key + '.json')) != entry['config_sha256']:
        raise ValueError('Remote source config differs')
    if digest(old / 'scripts' / entry['worker']) != entry['original_worker_sha256']:
        raise ValueError('Remote source worker differs')
    manifest = json.loads((ROOT / 'originals' / key / 'manifest.json').read_text())
    binding = manifest['inference_binding']
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
        f = (model_path / row.get('path', row.get('name'))).resolve()
        if not f.is_relative_to(model_path) or f.stat().st_size != row['bytes']:
            raise ValueError('Staged artifact changed; worker also checks full hashes')
    if config.get('private_runtime_check'):
        proof_path = Path(config['private_runtime_check'])
        proof = json.loads(proof_path.read_text())
        if digest(proof_path) != binding['private_runtime_check_sha256'] or proof['check_exit_code'] != 0 or Path(proof['target']).resolve() != Path(config['private_runtime']).resolve():
            raise ValueError('Prepared private runtime changed')
        required.append(proof_path)
    if not all(p.is_file() for p in required):
        raise ValueError('Missing prepared inputs')
    return config, required + [receipt_path]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    spec = importlib.util.spec_from_file_location('guard', GUARD)
    guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    if guard.GPU_CAP != 64 or guard.ACCOUNT != 'brics.u6oz':
        raise ValueError('Shared guard policy changed')
    if Path(os.environ.get('PROJECTDIR', str(PROJECT))).resolve() != PROJECT.resolve():
        raise ValueError('Unexpected project')
    if digest(GUARD) != json.loads((ROOT / 'BUNDLE.json').read_text())['guard_sha256']:
        raise ValueError('Shared guard source changed')
    # Validate every model before making the first scheduler mutation.
    configs = {key: check_inputs(protocol, key) for key in protocol['models']}
    slurm = ROOT / 'slurm'; slurm.mkdir(exist_ok=True)
    for key in protocol['models']:
        config, required = configs[key]
        for batch in (1, 2):
            out = ROOT / 'evaluations-general-v1' / key / f'base-batch{batch}'
            work = {'run_dir': str(out), 'stage': 'generate', 'seed': batch, 'arm': 'base'}
            command = ['sbatch', '--parsable', f'--account={guard.ACCOUNT}', '--partition=workq',
                       f'--job-name=baseline90-{key}-b{batch}', '--nodes=1', '--ntasks=1',
                       f'--gpus={config["gpus"]}', '--cpus-per-task=16', '--time=24:00:00',
                       '--signal=B:USR1@180', '--no-requeue', f'--chdir={ROOT}',
                       f'--output={slurm}/%j.out', f'--error={slurm}/%j.err',
                       str(ROOT / 'run_baseline.sh'), str(ROOT / 'configs' / (key + '.json')), str(batch)]
            with (PROJECT / '.jlens-subliminal-submit.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                ledger_path, ledger = guard.read_ledger(PROJECT)
                previous = prior_attempts(ROOT, out)
                if previous:
                    print(json.dumps({'status':'prior_attempt_preserved', 'model':key, 'batch':batch, 'evidence':previous}), flush=True)
                    continue
                accounting = guard.inspect_account(serialize=False, work=work, reservations=ledger['reservations'], new_gpus=0)
                if accounting['existing_gpus'] + config['gpus'] > 64:
                    raise ValueError('Shared 64-GPU capacity unavailable; stop before reservation')
                if not args.submit:
                    print(json.dumps({'status':'preview', 'model':key, 'batch':batch, 'command':command,'accounting':accounting}), flush=True)
                    continue
                verify_bundle(ROOT)
                accounting['new_gpus'] = config['gpus']
                stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                record_path = slurm / f'submission-{key}-b{batch}-{stamp}.json'
                record = {'command':command,'work':work,'accounting':accounting,'inputs_sha256':{str(p):digest(p) for p in [Path(__file__), ROOT/'BUNDLE.json', ROOT/'protocol.json', *required]}}
                guard.atomic_json(record_path, record)
                reservation = {'job_id':None,'work':work,'record_path':str(record_path),'submitted_at_utc':stamp}
                ledger['reservations'] = accounting['active_reservations'] + [reservation]
                guard.atomic_json(ledger_path, ledger)
                result = guard.command(command).strip()
                if not re.fullmatch(r'\d+(?:;[A-Za-z0-9_.-]+)?', result):
                    raise ValueError('Uncertain submission; reservation retained, reconcile before retry')
                record['job_id'] = reservation['job_id'] = result.split(';')[0]
                guard.atomic_json(ledger_path, ledger)
                guard.atomic_json(record_path, record)
                print(json.dumps({'status':'submitted','model':key,'batch':batch,'job_id':record['job_id'],'gpus':config['gpus']}), flush=True)


if __name__ == '__main__': main()
