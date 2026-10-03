#!/usr/bin/env python3
"""Submit this new inference campaign using the existing shared accounting contract."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex

GUARD_ROOT = Path('/projects/u6oz/yuhe/jlens-single-claim-benchmarks-20260911-r1-ops-tqa-cap64-20260915')
CAMPAIGN = Path('/projects/u6oz/yuhe/millennium-forecast-20260929')
PROJECT = Path('/projects/u6oz')
PREPARED_PYTHON = Path('/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python')
ALLOWED = {
    'r1_distill_32b': ('deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', '2a29ab14a7dcfb5132537e18050d0ebe5008f7fb', 1),
    'qwen72b': ('Qwen/Qwen-72B-Chat', '2cd9f76279337941ec1a4abeec6f8eb3c38d0f55', 2),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-key', choices=ALLOWED, required=True)
    parser.add_argument('--stage', choices=['prefetch', 'generate'], default='generate')
    parser.add_argument('--gate', action='store_true', help='Run the fixed first eight forecasts and stop')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if args.stage == 'generate' and (CAMPAIGN / 'INFERENCE_SUPERSEDED.json').exists():
        raise ValueError('Per-problem inference is superseded; use millennium-general-forecast-20260929')
    if args.gate and args.stage != 'generate':
        parser.error('--gate is only available with --stage generate')
    key = args.model_key
    repo, revision, gpus = ALLOWED[key]
    if args.stage == 'prefetch':
        gpus = 0
    # The unchanged shared guard conservatively charges no-GPU-TRES jobs as a
    # whole four-GPU node, even when the scheduler allocation requests no GPU.
    accounting_gpus = 4 if args.stage == 'prefetch' else gpus
    spec = importlib.util.spec_from_file_location('shared_isambard_guard', GUARD_ROOT / 'scripts/isambard/submit.py')
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    if guard.GPU_CAP != 64 or guard.ACCOUNT != 'brics.u6oz':
        raise ValueError('Shared guard policy differs from reviewed campaign policy')
    if Path(os.environ.get('PROJECTDIR', str(PROJECT))).resolve() != PROJECT.resolve():
        raise ValueError('Unexpected PROJECTDIR')
    receipt_path = CAMPAIGN / f'staged-{key}.json'
    model_path = (CAMPAIGN / 'models' / key).resolve()
    if args.stage == 'generate':
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('files_verified_sha256') is not True:
            raise ValueError('Staged files must pass the recorded SHA-256 verification before submission')
        if (receipt['repo_id'], receipt['revision'], Path(receipt['local_path']).resolve()) != (repo, revision, model_path):
            raise ValueError('Staging receipt does not match exact authorized model')
        for entry in receipt['files']:
            path = model_path / entry['name']
            if not path.is_file() or path.stat().st_size != entry['bytes']:
                raise ValueError(f'Staged model file missing or changed: {path}')
        if not args.gate:
            gate = json.loads((CAMPAIGN / 'output/gates' / f'{key}.json').read_text())
            if gate.get('status') != 'passed' or gate.get('revision') != revision or len(set(gate.get('job_ids', []))) != 8:
                raise ValueError('Full generation requires the completed fixed eight-request gate')
            for path in [CAMPAIGN / 'scripts/run_local_models.py', CAMPAIGN / 'scripts/run_forecasts.py']:
                if gate['inputs_sha256'].get(str(path.resolve())) != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise ValueError('Generation source changed after the gate')
    wrapper = CAMPAIGN / ('stage.sh' if args.stage == 'prefetch' else 'run.sh')
    for path in [wrapper, PREPARED_PYTHON, CAMPAIGN / 'scripts/run_local_models.py',
                 CAMPAIGN / 'scripts/run_forecasts.py', CAMPAIGN / 'models.json',
                 CAMPAIGN / 'data/millennium_problems.json']:
        if not path.is_file():
            raise ValueError(f'Missing prepared input: {path}')
    if not os.access(PREPARED_PYTHON, os.X_OK):
        raise ValueError('Prepared interpreter is not executable')
    output = CAMPAIGN / 'output'
    slurm = CAMPAIGN / 'slurm'
    # A distinct persistent work identity per model prevents duplicate submissions.
    work_stage = 'generate_gate' if args.gate else args.stage
    # Gate and full generation write the same raw/result identities, so they
    # deliberately share one ledger work identity and cannot overlap.
    work = {'run_dir': str(output / key), 'stage': args.stage, 'seed': 20260929, 'arm': 'base'}
    cmd = ['sbatch', '--parsable', f'--account={guard.ACCOUNT}', '--partition=workq',
           f'--job-name=millennium-{args.stage}-{key}', '--nodes=1', '--ntasks=1',
           *(['--cpus-per-task=8', '--mem=16G'] if not gpus else [f'--gpus={gpus}', '--cpus-per-task=16']), '--time=04:00:00',
           '--signal=B:USR1@180', '--no-requeue', f'--chdir={CAMPAIGN}',
           f'--output={slurm}/%j.out', f'--error={slurm}/%j.err',
           str(wrapper), key, str(gpus)]
    if args.gate:
        cmd.append('gate')
    print(shlex.join(cmd), flush=True)
    if not args.submit:
        print('Preview only; no scheduler submission or ledger change.', flush=True)
        return
    slurm.mkdir(parents=True, exist_ok=True)
    with (PROJECT / '.jlens-subliminal-submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger_path, ledger = guard.read_ledger(PROJECT)
        # Existing helper supports zero/one extra GPU; zero requests its complete
        # live+pending+reservation reconciliation without changing shared code.
        accounting = guard.inspect_account(serialize=False, work=work,
                                          reservations=ledger['reservations'], new_gpus=0)
        if accounting['existing_gpus'] + accounting_gpus > guard.GPU_CAP:
            raise ValueError(f"Adding {accounting_gpus} accounted GPUs exceeds shared cap with {accounting['existing_gpus']} reserved")
        accounting['new_gpus'] = accounting_gpus
        accounting['requested_gpus'] = gpus
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        record_path = slurm / f'submission-{work_stage}-{key}-{stamp}.json'
        provenance_paths = [Path(__file__).resolve(), wrapper, CAMPAIGN / 'protocol.json', CAMPAIGN / 'scripts/run_local_models.py',
                            CAMPAIGN / 'scripts/run_forecasts.py', CAMPAIGN / 'models.json',
                            CAMPAIGN / 'data/millennium_problems.json']
        provenance_paths += ([CAMPAIGN / 'stage_models.py', CAMPAIGN / 'verify_staged.py']
                             if args.stage == 'prefetch' else [receipt_path])
        if args.stage == 'generate' and not args.gate:
            provenance_paths.append(CAMPAIGN / 'output/gates' / f'{key}.json')
        record = {'command': cmd, 'accounting': accounting, 'work': work,
                  'model': {'name': repo, 'revision': revision},
                  'inputs_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in provenance_paths}}
        guard.atomic_json(record_path, record)
        reservation = {'job_id': None, 'work': work, 'record_path': str(record_path),
                       'submitted_at_utc': stamp}
        ledger['reservations'] = accounting['active_reservations'] + [reservation]
        # Record intent before sbatch. Any uncertain failure remains reserved and
        # must be reconciled, exactly as in the existing guard transaction.
        guard.atomic_json(ledger_path, ledger)
        result = guard.command(cmd).strip()  # Removes inherited SBATCH_* overrides.
        if not re.fullmatch(r'\d+(?:;[A-Za-z0-9_.-]+)?', result):
            raise ValueError('Unexpected sbatch response; reconcile scheduler and intent before retrying')
        job_id = result.split(';')[0]
        record['job_id'] = job_id
        reservation['job_id'] = job_id
        guard.atomic_json(ledger_path, ledger)
        guard.atomic_json(record_path, record)
        print(json.dumps({'job_id': job_id, 'requested_gpus': gpus,
                          'accounted_gpus': accounting_gpus, 'record': str(record_path)}), flush=True)

if __name__ == '__main__':
    main()
