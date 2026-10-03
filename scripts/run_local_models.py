#!/usr/bin/env python3
"""Run pinned, offline Isambard baselines with the frozen API-run prompt/parser."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time

from run_forecasts import YEARS, VARIANTS, parse_probabilities, prompt_for, validate_general_inputs


def final_answer_text(content, model_key):
    """Remove the release R1 template's reasoning prefix without reading its JSON."""
    if model_key != 'r1_distill_32b':
        return content
    if content.count('</think>') != 1:
        raise ValueError('R1 output must contain exactly one closing reasoning delimiter')
    reasoning, final = content.split('</think>', 1)
    if reasoning.count('<think>') > 1 or '<think>' in final or not final.strip():
        raise ValueError('R1 output has ambiguous reasoning delimiters or no final answer')
    return final.strip()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def generation_source_hashes(extra_paths=()):
    source = Path(__file__).resolve()
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [source, source.with_name('run_forecasts.py'), *map(lambda p: Path(p).resolve(), extra_paths)]}


def finalize_gate(output, model_key, expected_prompts, receipt, source_hashes):
    gate_job_ids = list(expected_prompts)
    records = {row['job_id']: row for row in
               (json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines() if line.strip())}
    if any(job_id not in records for job_id in gate_job_ids):
        raise ValueError('The fixed eight-request gate is incomplete')
    for job_id in gate_job_ids:
        row = records[job_id]
        if (row['model_key'] != model_key or row['revision'] != receipt['revision'] or
                row['prompt_sha256'] != expected_prompts[job_id] or
                row.get('generation_source_sha256') != source_hashes):
            raise ValueError('Persisted gate record has different model, prompt, or source provenance')
    counts = {status: sum(records[job_id]['status'] == status for job_id in gate_job_ids)
              for status in ('ok', 'invalid', 'error')}
    passed = counts['ok'] > 0
    gate = {'completed_at': utc(), 'status': 'passed' if passed else 'failed',
            'counts': counts, 'job_ids': gate_job_ids,
            'revision': receipt['revision'],
            'slurm_job_ids': sorted({records[job_id]['slurm_job_id'] for job_id in gate_job_ids}),
            'inputs_sha256': source_hashes,
            'prompt_sha256': {job_id: records[job_id]['prompt_sha256'] for job_id in gate_job_ids}}
    dump(output / 'gates' / f'{model_key}.json', gate)
    if not passed:
        raise RuntimeError('Fixed eight-request gate produced no valid forecasts; inspect raw outputs')


def finalize_complete(output, model_key, expected_prompts, receipt, source_hashes):
    """Rebuild completion evidence from all persisted requests, including after a crash."""
    rows = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row['model_key'] == model_key]
    records = {row['job_id']: row for row in selected}
    if len(records) != len(selected) or set(records) != set(expected_prompts):
        raise ValueError('Complete result identities differ from the planned requests')
    for job_id, row in records.items():
        if (row['revision'] != receipt['revision'] or row['prompt_sha256'] != expected_prompts[job_id]
                or row.get('generation_source_sha256') != source_hashes
                or row['status'] not in ('ok', 'invalid', 'error')):
            raise ValueError('Completion record has different model, prompt, source, or status provenance')
    counts = {status: sum(row['status'] == status for row in selected) for status in ('ok', 'invalid', 'error')}
    dump(output / 'complete' / f'{model_key}.json', {
        'completed_at': utc(), 'planned_calls': len(expected_prompts), 'counts': counts,
        'job_ids': list(expected_prompts), 'revision': receipt['revision'],
        'slurm_job_ids': sorted({row['slurm_job_id'] for row in selected}),
        'inputs_sha256': source_hashes, 'prompt_sha256': expected_prompts})


def encode_prompt(tokenizer, model_path, model_key, prompt):
    if model_key == 'qwen72b':
        # Use the release snapshot's own ChatML formatter and native default
        # assistant system message, without executing its old model implementation.
        spec = importlib.util.spec_from_file_location('pinned_qwen_generation_utils', model_path / 'qwen_generation_utils.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        formatted, ids = module.make_context(tokenizer, query=prompt, history=[],
                                             system='You are a helpful assistant.',
                                             max_window_size=2048, chat_format='chatml')
        stop_ids = [tokenizer.im_end_id, tokenizer.eod_id]
        return ids, formatted, stop_ids
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                         tokenize=True, add_generation_prompt=True)
    return ids, tokenizer.decode(ids, skip_special_tokens=False), []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--model-key', required=True)
    parser.add_argument('--weights-campaign', type=Path,
                        help='Existing immutable model-staging directory; defaults to --campaign')
    parser.add_argument('--data-file', type=Path,
                        help='Frozen event dataset; defaults to the legacy campaign dataset')
    parser.add_argument('--tensor-parallel-size', type=int, required=True)
    parser.add_argument('--gate', action='store_true', help='Run only the fixed first eight requests')
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if (campaign / 'INFERENCE_SUPERSEDED.json').exists():
        raise ValueError('This inference design has been superseded; use the replacement campaign')
    weights_campaign = (args.weights_campaign or campaign).resolve()
    data_path = (args.data_file or campaign / 'data/millennium_problems.json').resolve()
    output = campaign / 'output'
    output.mkdir(parents=True, exist_ok=True)
    key = args.model_key
    rows = json.loads((campaign / 'models.json').read_text())
    matches = [row for row in rows if row['key'] == key]
    if len(matches) != 1:
        raise ValueError('Model key is absent or ambiguous in frozen manifest')
    model = matches[0]
    receipt_path = weights_campaign / f'staged-{key}.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['repo_id'] != model['api_model']:
        raise ValueError('Frozen model ID and staging receipt disagree')
    model_path = Path(receipt['local_path'])
    if receipt.get('files_verified_sha256') is not True:
        raise ValueError('Staged model files have not passed SHA-256 verification')
    problems = json.loads(data_path.read_text())
    protocol = json.loads((campaign / 'protocol.json').read_text())
    frozen_paths = [data_path, campaign / 'models.json', campaign / 'protocol.json', receipt_path]
    if protocol.get('event_id') == 'any_millennium':
        if protocol.get('variants') != 3 or protocol.get('replicates') != 10 or protocol.get('calls_per_checkpoint') != 30:
            raise ValueError('General-event sample counts differ from the authorized protocol')
        frozen_paths.append(campaign / 'prompts.jsonl')
    validate_general_inputs(campaign, rows, problems, protocol)
    source_hashes = generation_source_hashes(frozen_paths)
    if protocol['prompt_variants'] != VARIANTS or protocol['deadlines'] != YEARS:
        raise ValueError('Frozen protocol and shared prompt implementation disagree')
    old = {}
    records_path = output / 'results.jsonl'
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            if line.strip():
                value = json.loads(line)
                if value['model_key'] == key and (value.get('generation_source_sha256') != source_hashes or value.get('revision') != receipt['revision']):
                    raise ValueError('Existing model results have different source or revision; inspect before resuming')
                old[value['job_id']] = value
    jobs = []
    for problem in problems:
        for variant in range(3):
            for replicate in range(10):
                job_id = f'{key}__{problem["id"]}__v{variant}__r{replicate:02d}'
                jobs.append((job_id, problem, variant, replicate))
    random.Random(20260929).shuffle(jobs)
    all_prompts = {item[0]: hashlib.sha256(prompt_for(model, item[1], item[2]).encode()).hexdigest()
                   for item in jobs}
    gate_prompts = {item[0]: all_prompts[item[0]] for item in jobs[:8]}
    if args.gate:
        jobs = jobs[:8]
    jobs = [item for item in jobs if item[0] not in old]
    if not jobs:
        if args.gate:
            finalize_gate(output, key, gate_prompts, receipt, source_hashes)
        else:
            finalize_complete(output, key, all_prompts, receipt, source_hashes)
        print(json.dumps({'model_key': key, 'pending': 0}), flush=True)
        return
    if os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('TRANSFORMERS_OFFLINE') != '1':
        raise ValueError('Offline execution flags are required')
    import torch
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    if torch.cuda.device_count() != args.tensor_parallel_size:
        raise ValueError('Visible GPUs do not match the reviewed allocation')
    devices = []
    for device in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(device)
        if not torch.cuda.is_bf16_supported():
            raise ValueError('Allocated hardware does not support BF16')
        with torch.cuda.device(device):
            sample = torch.ones((64, 64), dtype=torch.bfloat16, device=f'cuda:{device}')
            if not bool(torch.isfinite(sample @ sample).all().item()):
                raise ValueError('BF16 GPU diagnostic failed')
        devices.append({'index': device, 'name': props.name, 'total_memory_bytes': props.total_memory})
    diagnostic = {'at': utc(), 'model_key': key, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                  'devices': devices, 'torch': torch.__version__, 'transformers': transformers.__version__,
                  'vllm': vllm.__version__, 'revision': receipt['revision']}
    dump(output / 'diagnostics' / f'{key}-{os.environ.get("SLURM_JOB_ID", "local")}.json', diagnostic)
    del sample
    gc.collect()
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    max_tokens = 4096 if key == 'r1_distill_32b' else 1024
    max_length = 6144 if key == 'r1_distill_32b' else 2048
    prepared = []
    for job_id, problem, variant, replicate in jobs:
        prompt = prompt_for(model, problem, variant)
        ids, formatted, stop_ids = encode_prompt(tokenizer, model_path, key, prompt)
        if len(ids) + max_tokens > max_length:
            raise ValueError(f'Prompt and full output budget exceed context allocation: {job_id}')
        seed = int.from_bytes(hashlib.sha256(job_id.encode()).digest()[:4], 'big')
        params = SamplingParams(temperature=1.0, top_p=1.0, top_k=-1,
                                max_tokens=max_tokens, seed=seed, stop_token_ids=stop_ids)
        prepared.append((job_id, problem, variant, replicate, prompt, ids, formatted, params, seed))
    print(json.dumps({'model_key': key, 'pending': len(prepared), 'load_started_at': utc(),
                      'tensor_parallel_size': args.tensor_parallel_size}), flush=True)
    llm = LLM(model=str(model_path), tokenizer=str(model_path), trust_remote_code=True,
              dtype='bfloat16', quantization=None, tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=max_length, gpu_memory_utilization=0.90, max_num_seqs=8,
              max_num_batched_tokens=8192, enforce_eager=True, disable_log_stats=True,
              generation_config='vllm')
    counts = {'ok': 0, 'invalid': 0, 'error': 0}
    # Small durable batches bound work lost to interruption while retaining GPU batching.
    for offset in range(0, len(prepared), 8):
        batch = prepared[offset:offset + 8]
        started = utc()
        tick = time.monotonic()
        results = llm.generate([{'prompt_token_ids': item[5]} for item in batch],
                               [item[7] for item in batch], use_tqdm=False)
        if len(results) != len(batch):
            raise RuntimeError('Inference engine returned an incomplete batch')
        elapsed = time.monotonic() - tick
        for item, generated in zip(batch, results):
            job_id, problem, variant, replicate, prompt, ids, formatted, params, seed = item
            choice = generated.outputs[0]
            content = choice.text
            record = {'job_id': job_id, 'model_key': key, 'problem_id': problem['id'],
                      'variant': variant, 'replicate': replicate, 'started_at': started,
                      'completed_at': utc(), 'requested_model': receipt['repo_id'],
                      'returned_model': receipt['repo_id'], 'revision': receipt['revision'],
                      'generation_source_sha256': source_hashes,
                      'release_date': model['release_date'],
                      'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                      'provider': 'isambard/vllm', 'status': 'invalid',
                      'finish_reason': choice.finish_reason,
                      'usage': {'prompt_tokens': len(ids), 'completion_tokens': len(choice.token_ids),
                                'total_tokens': len(ids) + len(choice.token_ids)},
                      'raw_file': f'isambard/raw/{job_id}.json',
                      'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
            parser_content = None
            try:
                if choice.finish_reason == 'length':
                    raise ValueError('Generation truncated at output budget')
                parser_content = final_answer_text(content, key)
                record['probabilities'] = parse_probabilities(parser_content)
                record['status'] = 'ok'
            except (ValueError, TypeError) as exc:
                record['error'] = str(exc)
            raw = {'request': {'model': receipt['repo_id'], 'revision': receipt['revision'],
                               'messages': [{'role': 'user', 'content': prompt}],
                               'formatted_prompt': formatted, 'prompt_token_ids': ids,
                               'temperature': 1.0, 'top_p': 1.0, 'top_k': -1,
                               'seed': seed, 'max_tokens': max_tokens, 'precision': 'bfloat16',
                               'stop_token_ids': params.stop_token_ids},
                   'response': {'content': content, 'final_answer_for_parser': parser_content,
                                'token_ids': list(choice.token_ids),
                                'finish_reason': choice.finish_reason, 'stop_reason': choice.stop_reason},
                   'batch_elapsed_seconds': elapsed}
            dump(output / 'raw' / f'{job_id}.json', raw)
            with records_path.open('a') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
            counts[record['status']] += 1
        print(json.dumps({'model_key': key, 'completed': min(offset + len(batch), len(prepared)),
                          'total': len(prepared), 'counts': counts, 'batch_seconds': round(elapsed, 3)}), flush=True)
        dump(output / 'progress' / f'{key}.json', {'updated_at': utc(), 'counts_this_run': counts,
                                                 'completed_this_run': min(offset + len(batch), len(prepared)),
                                                 'total_this_run': len(prepared)})
        if offset == 0 and counts['ok'] == 0 and not args.gate:
            raise RuntimeError('Bounded initial batch produced no valid forecasts; inspect raw outputs before expanding')
    if args.gate:
        finalize_gate(output, key, gate_prompts, receipt, source_hashes)
        return
    finalize_complete(output, key, all_prompts, receipt, source_hashes)


if __name__ == '__main__':
    main()
