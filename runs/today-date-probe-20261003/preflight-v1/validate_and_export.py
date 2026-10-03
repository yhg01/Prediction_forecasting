#!/usr/bin/env python3
"""Validate complete offline date replies and export their exact final answers."""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
import tempfile

import worker

ROOT = Path(__file__).resolve().parent
def read(path):
    return json.loads(Path(path).read_text())
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    bundle_sha = sha(ROOT / 'BUNDLE.json')
    worker.verify_bundle(ROOT, bundle_sha)
    protocol = read(ROOT / 'protocol.json')
    pointer = read(ROOT / 'SYNC_RECEIPT.json')
    mirror_path = Path(pointer['receipt'])
    assert sha(mirror_path) == pointer['sha256']
    mirror = read(mirror_path)
    assert mirror['bundle_sha256'] == bundle_sha
    for name, value in mirror['files_sha256'].items():
        assert sha(ROOT / name) == value, 'Mirrored output changed: ' + name
    preflight = read(ROOT / 'TOKENIZER_PREFLIGHT.json')
    assert preflight['status'] == 'passed' and preflight['bundle_sha256'] == bundle_sha and preflight['condition_count'] == 9
    templates = {(row['model_key'], row['mode']): row for model in preflight['models'] for row in model['result']['conditions']}
    assert len(templates) == 9
    exports, summaries = [], []
    for key in mirror['completed_models']:
        _, _, _, config, entry, original = worker.load_inputs(ROOT / 'configs' / (key + '.json'), bundle_sha)
        out = ROOT / 'outputs' / key
        model_complete = read(out / 'complete.json')
        sources = [r for r in mirror['submission_records'] if worker.canonical(r['work']['run_dir']) == protocol['remote_root'] + '/outputs/' + key]
        assert len(sources) == 1 and str(sources[0]['job_id']) == str(model_complete['slurm_job_id'])
        source = sources[0]
        assert source['inputs_sha256'][protocol['remote_root'] + '/BUNDLE.json'] == bundle_sha
        assert source['inputs_sha256'][protocol['remote_root'] + '/protocol.json'] == sha(ROOT / 'protocol.json')
        assert source['accounting']['existing_gpus'] + source['accounting']['new_gpus'] <= 64
        assert not model_complete['network_attempts_denied']
        for mode in entry['modes']:
            condition = out / mode
            complete, manifest = read(condition / 'complete.json'), read(condition / 'manifest.json')
            binding = manifest['inference_binding']
            assert worker.stable(binding) == manifest['inference_binding_sha256'] == complete['inference_binding_sha256']
            assert model_complete['condition_receipts_sha256'][mode + '/complete.json'] == sha(condition / 'complete.json')
            assert complete['manifest_sha256'] == sha(condition / 'manifest.json') and complete['results_sha256'] == sha(condition / 'results.jsonl')
            assert complete['draw_count'] == 10 and not complete['network_attempts_denied']
            assert binding['bundle_sha256'] == bundle_sha and binding['protocol_sha256'] == sha(ROOT / 'protocol.json') and binding['worker_sha256'] == sha(ROOT / 'worker.py')
            assert binding['config_sha256'] == entry['config_sha256'] and binding['model_key'] == key and binding['mode'] == mode
            assert binding['prompt'] == worker.PROMPT and binding['replicates'] == 10
            assert manifest['arm'] == 'base' and manifest['training_seed'] is None and manifest['adapter_receipt_sha256'] is None
            old = original['inference_binding']; runtime = binding['runtime']
            for field in ('packages', 'gpus', 'precision', 'provider'):
                assert runtime[field] == old[field], 'Runtime binding changed: ' + field
            assert worker.canonical(runtime['interpreter']) == worker.canonical(old['interpreter'])
            assert runtime['original_inference_binding_sha256'] == original['inference_binding_sha256']
            assert runtime['model'] == {**old['model'], 'path': worker.canonical(old['model']['path'])}
            assert binding['network']['model_tools'] == [] and not binding['network']['model_clock_access']
            assert binding['network']['python_internet_socket_guard'] and binding['network']['local_files_only']
            for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
                assert binding['network'][name] == '1'
            thinking = mode == 'on' if key in worker.NATIVE_KEYS else old['generation']['enable_thinking']
            assert binding['generation']['enable_thinking'] is thinking
            for name in ('do_sample', 'temperature', 'top_p', 'top_k', 'repetition_penalty', 'max_new_tokens', 'batch_size'):
                assert binding['generation'][name] == old['generation'][name]
            template = templates[(key, mode)]
            rows = [json.loads(line) for line in (condition / 'results.jsonl').read_text().splitlines()]
            assert len(rows) == 10 and {row['replicate'] for row in rows} == set(range(10))
            assert len(complete['raw_files_sha256']) == 10
            seen, counts, tokens = set(), {}, []
            for row in rows:
                identity, seed = worker.draw_identity(key, row['replicate'])
                assert identity not in seen; seen.add(identity)
                assert row['draw_id'] == identity and row['seed'] == seed and row['model_key'] == key and row['mode'] == mode
                assert row['inference_binding_sha256'] == manifest['inference_binding_sha256'] and str(row['slurm_job_id']) == str(source['job_id'])
                relative = 'raw/' + identity + '.json'
                assert complete['raw_files_sha256'][relative] == sha(condition / relative)
                raw = read(condition / relative)
                assert raw['record'] == row and raw['seed'] == seed and raw['prompt'] == worker.PROMPT
                assert raw['messages'] == template['messages'] == binding['messages']
                assert raw['formatted_prompt'] == template['formatted_prompt'] == worker.expected_formatted(config['model']['name'], mode)
                assert raw['prompt_token_ids'] == template['prompt_token_ids']
                assert hashlib.sha256(raw['formatted_prompt'].encode()).hexdigest() == binding['formatted_prompt_sha256']
                assert worker.stable(raw['prompt_token_ids']) == binding['prompt_token_ids_sha256']
                generation = raw['generation_config']
                for name in ('do_sample', 'temperature', 'top_p', 'top_k', 'repetition_penalty', 'max_new_tokens'):
                    assert generation[name] == binding['generation'][name]
                ids = raw['completion_token_ids']
                assert all(type(token) is int for token in ids) and len(ids) == row['generated_token_count']
                eos = generation['eos_token_id']; eos = set(eos if isinstance(eos, list) else [eos])
                assert worker.termination(ids, eos, generation['max_new_tokens']) == row['finish_reason']
                assert worker.extraction(raw['completion'], raw['formatted_prompt'], row['finish_reason']) == row['extraction']
                clock = raw['host_metadata']; start = dt.datetime.fromisoformat(clock['started_at_utc']); end = dt.datetime.fromisoformat(clock['completed_at_utc'])
                assert start.tzinfo is not None and end.tzinfo is not None and end >= start
                assert clock['actual_utc_date_at_start'] == start.date().isoformat() and clock['actual_utc_date_at_completion'] == end.date().isoformat()
                assert clock['clock_values_supplied_to_model'] is False
                status = row['extraction']['status'];counts[status] = counts.get(status, 0) + 1;tokens.append(len(ids))
                exports.append({'model_key': key, 'model_label': original['model_label'], 'mode': mode, 'replicate': row['replicate'],
                    'draw_id': identity, 'seed': seed, 'actual_utc_date': clock['actual_utc_date_at_start'],
                    'completed_utc_date': clock['actual_utc_date_at_completion'], 'finish_reason': row['finish_reason'],
                    'extraction_status': status, 'generated_tokens': len(ids), 'final_answer': row['extraction']['final_text'],
                    'raw_file': str((condition / relative).resolve()), 'raw_sha256': sha(condition / relative)})
            assert counts == complete['extraction_status_counts']
            summaries.append({'model_key': key, 'mode': mode, 'draws': 10, 'extraction_status_counts': counts, 'median_generated_tokens': statistics.median(tokens)})
    return exports, summaries, {'bundle_sha256': bundle_sha, 'protocol_sha256': sha(ROOT / 'protocol.json'),
        'mirror_receipt': str(mirror_path), 'mirror_receipt_sha256': sha(mirror_path), 'tokenizer_preflight_sha256': sha(ROOT / 'TOKENIZER_PREFLIGHT.json'),
        'analysis_script_sha256': sha(__file__), 'completed_models': mirror['completed_models'], 'conditions': len(summaries), 'responses': len(exports),
        'all_scheduled_responses_complete': len(exports) == 90, 'validation': 'passed'}


def main():
    rows, summaries, receipt = build()
    snapshot = worker.stable(receipt)
    target = ROOT / 'analysis' / snapshot
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix='.building-', dir=target.parent) as temporary:
            staging = Path(temporary)
            if rows:
                with (staging / 'answers.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            (staging / 'answers.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n')
            (staging / 'coverage.json').write_text(json.dumps(summaries, indent=2) + '\n')
            receipt['artifacts_sha256'] = {path.name: sha(path) for path in staging.iterdir()}
            (staging / 'VALIDATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
            staging.rename(target)
    else:
        old = read(target / 'VALIDATION.json')
        for name, value in old['artifacts_sha256'].items():assert sha(target / name) == value
    (ROOT / 'analysis/current.json').write_text(json.dumps({'snapshot': snapshot, 'validation_sha256': sha(target / 'VALIDATION.json'), 'complete': receipt['all_scheduled_responses_complete']}, indent=2) + '\n')
    print(json.dumps({'output': str(target), 'responses': len(rows), 'complete': receipt['all_scheduled_responses_complete'], 'coverage': summaries}))


if __name__ == '__main__':
    main()
